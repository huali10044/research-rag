"""
run_eval.py — Evaluation harness for the Research Paper RAG system.

Two tracks, selected by query type, because the metrics are not
interchangeable:

  content / metadata  → RAGAS
                        faithfulness, answer relevancy,
                        context precision, context recall
  stats / listing     → deterministic assertion checks
                        These queries deliberately bypass vector search
                        (adaptive_retrieve returns zero chunks and injects
                        corpus_stats.json instead), so every context-based
                        RAGAS metric is undefined for them. Scoring them
                        with RAGAS would report a fake 0.0 and drag the
                        headline number down for a correctly-working path.

Usage
-----
    # Full evaluation, default config
    python eval/run_eval.py

    # Ablation over retrieval depth — this is what produces the comparison table
    python eval/run_eval.py --sweep-top-k 3 5 8 12

    # Only the fast deterministic track (no LLM judge calls)
    python eval/run_eval.py --track exact

    # Use a different generator or judge
    python eval/run_eval.py --backend gemini --judge-backend anthropic

Configuration
-------------
Defaults come from eval/config.toml (generator/judge backend + model, top_k,
track, throttling — safe to commit). API keys come from eval/secrets.toml
(gitignored; copy from eval/secrets.toml.example) and are injected into
os.environ under the same variable names rag.py already reads, so they work
for both the generator and the RAGAS judge. Every config.toml value can be
overridden on the command line; flags win.

Free-tier note
--------------
Gemini free tier allows ~15 requests/minute. RAGAS defaults to 16 parallel
workers, which will exhaust that budget within seconds and produce a wall of
429s that look like evaluation failures. This harness pins max_workers to 1
and throttles generation by default. A full 25-question run takes roughly
10-15 minutes on free tier. Use --max-workers to raise it on a paid key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

# --- make src/ importable regardless of where this is invoked from ------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag  # noqa: E402  (path setup must precede this import)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_PATH = EVAL_DIR / "config.toml"

RAG_TRACK_TYPES = {"content", "metadata"}
EXACT_TRACK_TYPES = {"stats", "listing"}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("eval")


# ── Config file ───────────────────────────────────────────────────────────────
# Mirrors app.py's pattern: config.toml holds non-secret defaults (which
# backend/model to use), API keys live in a separate gitignored file and get
# injected into os.environ so rag.py's existing os.environ.get(...) lookups
# (and resolve_backend's availability check) work unchanged.
#
#   eval/config.toml        — defaults, safe to commit
#   eval/secrets.toml       — API keys, gitignored (copy from secrets.toml.example)

def load_eval_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load eval/config.toml if present. Returns {} if missing."""
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_eval_secrets(cfg: dict) -> None:
    """
    Inject API keys into os.environ so rag.py and the RAGAS judge builders
    can find them via os.environ.get(...), exactly like app.py does with
    st.secrets. Precedence: real environment variables win over the secrets
    file, so a key already exported in the shell is never overridden.
    """
    secrets_path = EVAL_DIR / cfg.get("secrets_file", "secrets.toml")
    if not secrets_path.exists():
        return
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    for key, val in secrets.items():
        if isinstance(val, str) and val:
            os.environ.setdefault(key, val)


# ── Golden set ────────────────────────────────────────────────────────────────

def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    data = json.loads(path.read_text())
    questions = data["questions"]

    unverified = [q["id"] for q in questions if not q.get("verified")]
    if unverified:
        print(
            f"\n  NOTE: {len(unverified)} of {len(questions)} references are still "
            f"marked verified: false.\n"
            f"  Context precision and recall are only as good as those references.\n"
        )
    return questions


# ── Generation pass ───────────────────────────────────────────────────────────

def generate_predictions(
    questions: list[dict],
    backend: str,
    model: str,
    top_k: int,
    throttle: float,
) -> list[dict]:
    """Run every question through the live RAG pipeline and record the result."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(rag.EMBED_MODEL, device="cpu")
    collection = rag.get_collection()

    records = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i:>2}/{len(questions)}] {q['id']:<9} {q['query'][:58]}")
        t0 = time.perf_counter()
        try:
            answer, chunks, query_type = rag.adaptive_answer(
                query=q["query"],
                embedder=embedder,
                collection=collection,
                top_k=top_k,
                backend=backend,
                model=model,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 — record and continue the sweep
            log.warning("%s failed: %s", q["id"], exc)
            answer, chunks, query_type, error = "", [], "error", str(exc)

        records.append(
            {
                "id": q["id"],
                "query": q["query"],
                "expected_type": q["expected_type"],
                "actual_type": query_type,
                "type_match": query_type == q["expected_type"],
                "reference": q["reference"],
                "must_contain": q.get("must_contain", []),
                "answer": answer,
                "contexts": [c["text"] for c in chunks],
                "similarities": [c["similarity"] for c in chunks],
                "n_chunks": len(chunks),
                "latency_s": round(time.perf_counter() - t0, 2),
                "error": error,
            }
        )
        if throttle:
            time.sleep(throttle)

    return records


# ── Track 1: deterministic checks ─────────────────────────────────────────────

def score_exact_track(records: list[dict]) -> dict[str, Any]:
    """
    Substring assertions for stats/listing queries.

    Deliberately crude: these queries have exactly one correct answer drawn from
    corpus_stats.json, so an LLM judge adds cost and variance without adding
    information. A missing substring is a real regression.
    """
    rows = []
    for r in records:
        required = r["must_contain"]
        answer_lc = r["answer"].lower()
        # Tolerate thousands separators: "51,234" should match "51234" too.
        hits = [
            s
            for s in required
            if s.lower() in answer_lc
            or s.lower().replace(",", "") in answer_lc.replace(",", "")
        ]
        passed = len(hits) == len(required)
        rows.append(
            {
                "id": r["id"],
                "passed": passed,
                "missing": [s for s in required if s not in hits],
                "type_match": r["type_match"],
                "latency_s": r["latency_s"],
            }
        )

    n = len(rows) or 1
    return {
        "n": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 4),
        "type_accuracy": round(sum(r["type_match"] for r in rows) / n, 4),
        "rows": rows,
    }


# ── Routing accuracy ──────────────────────────────────────────────────────────
# classify_query drives which track (and which metrics) a question gets scored
# on, so a routing regression can silently move a question between the RAGAS
# track and the exact track. That's a different failure mode than "retrieval
# got worse" or "the assertion failed," so it's tracked across *all* records
# (not just stats/listing) and reported on its own, independent of --track.
#
# Records where generation errored (e.g. a 429) are excluded entirely: those
# are an availability failure of the generator/judge API, not a classify_query
# decision, and adaptive_answer never got far enough to route anything. Folding
# them into "mismatches" would blame routing for an outage — exactly the
# confusion this metric exists to avoid. They're reported separately as
# "errored" so an outage during a run is visible without being misattributed.

def score_routing(records: list[dict]) -> dict[str, Any]:
    scorable = [r for r in records if not r["error"]]
    errored = [r["id"] for r in records if r["error"]]

    by_type: dict[str, dict[str, int]] = {}
    mismatches = []
    for r in scorable:
        expected = r["expected_type"]
        bucket = by_type.setdefault(expected, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if r["type_match"]:
            bucket["correct"] += 1
        else:
            mismatches.append(
                {"id": r["id"], "expected": expected, "actual": r["actual_type"]}
            )

    per_type = {
        t: round(b["correct"] / b["n"], 4) if b["n"] else None
        for t, b in by_type.items()
    }
    n = len(scorable) or 1
    overall = round(sum(r["type_match"] for r in scorable) / n, 4)

    return {
        "overall": overall,
        "per_type": per_type,
        "mismatches": mismatches,
        "errored": errored,
    }


# ── Judge backend resolution ──────────────────────────────────────────────────

JUDGE_ORDER = ["gemini", "anthropic", "openai", "cohere", "groq", "mistral"]


def _backend_available(name: str) -> bool:
    cfg = rag.BACKENDS[name]
    return cfg["env_key"] is None or bool(os.environ.get(cfg["env_key"]))


def resolve_judge_backend(requested: str, generator_backend: str) -> tuple[str, str]:
    """
    Return (judge_backend, judge_model).

    "auto" prefers a backend that differs from the generator (so the model
    isn't grading its own output — see README's "Judge independence" section),
    falling back to whichever supported judge backend has a key available,
    including the generator's own backend as a last resort.
    """
    if requested != "auto":
        if requested not in JUDGE_ORDER:
            raise ValueError(
                f"Unsupported judge backend {requested!r}. Use one of: "
                + ", ".join(JUDGE_ORDER) + ", auto."
            )
        if not _backend_available(requested):
            cfg = rag.BACKENDS[requested]
            raise EnvironmentError(
                f"Judge backend '{requested}' requires {cfg['env_key']} to be set."
            )
        return requested, rag.BACKENDS[requested]["default_model"]

    independent = [b for b in JUDGE_ORDER if b != generator_backend]
    same_as_gen = [b for b in JUDGE_ORDER if b == generator_backend]
    for name in independent + same_as_gen:
        if _backend_available(name):
            return name, rag.BACKENDS[name]["default_model"]

    raise EnvironmentError(
        "No judge backend available. Set one of GEMINI_API_KEY, "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, COHERE_API_KEY, GROQ_API_KEY, "
        "MISTRAL_API_KEY in eval/secrets.toml or the environment, or run with "
        "--track exact to skip the judge entirely."
    )


# ── Judge LLM ─────────────────────────────────────────────────────────────────
# The judge should ideally differ from the generator, so the model is not
# grading its own output. Default is Gemini for both because the free tier makes
# that the only zero-cost option; pass --judge-backend to separate them.

def build_judge_llm(backend: str, model: str | None):
    from ragas.llms import LangchainLLMWrapper

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                # gemini-2.0-flash was deprecated by Google; the API's own 404
                # error names gemini-3.6-flash as the direct replacement.
                model=model or "gemini-3.6-flash",
                google_api_key=os.environ["GEMINI_API_KEY"],
                temperature=0.0,
            )
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(
            ChatOpenAI(model=model or "gpt-4o-mini", temperature=0.0)
        )

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LangchainLLMWrapper(
            ChatAnthropic(model=model or "claude-haiku-4-5-20251001", temperature=0.0)
        )

    if backend == "cohere":
        from langchain_cohere import ChatCohere

        return LangchainLLMWrapper(
            ChatCohere(
                # command-r was removed by Cohere on 2025-09-15; command-r7b
                # is the current lightweight replacement on the trial tier
                # (20 req/min, 1000 calls/month — see docs.cohere.com/v2/docs/rate-limits).
                model=model or "command-r7b-12-2024",
                cohere_api_key=os.environ["COHERE_API_KEY"],
                temperature=0.0,
            )
        )

    # groq and mistral both expose OpenAI-compatible chat completions endpoints
    # (same pattern rag.py's answer_with_groq / answer_with_mistral use), so
    # ChatOpenAI with a custom base_url covers them without a new dependency.
    if backend == "groq":
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(
            ChatOpenAI(
                model=model or "openai/gpt-oss-20b",
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0,
            )
        )

    if backend == "mistral":
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(
            ChatOpenAI(
                model=model or "mistral-small-latest",
                api_key=os.environ["MISTRAL_API_KEY"],
                base_url="https://api.mistral.ai/v1",
                temperature=0.0,
            )
        )

    raise ValueError(
        f"Unsupported judge backend {backend!r}. "
        "Use one of: gemini, openai, anthropic, cohere, groq, mistral."
    )


# ── Embeddings adapter ────────────────────────────────────────────────────────
# RAGAS needs a LangChain-compatible embedder. We reuse the *same* model the
# retriever uses (all-MiniLM-L6-v2) rather than an OpenAI embedding, for two
# reasons: it is free, and answer-relevancy is then measured in the same vector
# space the system actually retrieves in, which makes the number meaningful
# rather than merely comparable to other people's numbers.

def build_embeddings():
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name, device="cpu")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.model.encode(texts, device="cpu").tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.model.encode([text], device="cpu")[0].tolist()

    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(rag.EMBED_MODEL))


# ── Track 2: RAGAS ────────────────────────────────────────────────────────────

def score_rag_track(
    records: list[dict],
    judge_backend: str,
    judge_model: str | None,
    max_workers: int,
) -> dict[str, Any]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    scorable = [r for r in records if r["contexts"] and not r["error"]]
    skipped = [r["id"] for r in records if not r["contexts"] or r["error"]]
    if skipped:
        print(f"  Skipping {len(skipped)} sample(s) with no context or an error: {skipped}")
    if not scorable:
        return {"n": 0, "skipped": skipped, "scores": {}, "rows": []}

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["query"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
                reference=r["reference"],
            )
            for r in scorable
        ]
    )

    llm = build_judge_llm(judge_backend, judge_model)
    embeddings = build_embeddings()

    # max_workers=1 keeps free-tier judges inside their rate limit; the long
    # max_wait lets tenacity ride out a 429 instead of failing the whole run.
    run_config = RunConfig(
        timeout=300,
        max_retries=10,
        max_wait=90,
        max_workers=max_workers,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_cols = [
        c
        for c in df.columns
        if c
        not in {"user_input", "retrieved_contexts", "response", "reference"}
    ]

    scores: dict[str, float | None] = {}
    for col in metric_cols:
        vals = [v for v in df[col].tolist() if v is not None and v == v]  # drop NaN
        scores[col] = round(statistics.mean(vals), 4) if vals else None

    rows = []
    for r, (_, row) in zip(scorable, df.iterrows()):
        entry = {"id": r["id"], "n_chunks": r["n_chunks"], "latency_s": r["latency_s"]}
        for col in metric_cols:
            v = row[col]
            entry[col] = round(float(v), 4) if v is not None and v == v else None
        rows.append(entry)

    return {"n": len(scorable), "skipped": skipped, "scores": scores, "rows": rows}


# ── Reporting ─────────────────────────────────────────────────────────────────

def render_markdown(runs: list[dict]) -> str:
    """Render one or more configurations as a comparison table."""
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    metric_keys: list[str] = []
    for run in runs:
        for k in run["rag_track"].get("scores", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    if metric_keys:
        lines += [
            "## Retrieval and generation quality (content + metadata queries)",
            "",
            "| Config | n | " + " | ".join(metric_keys) + " |",
            "|---|---|" + "---|" * len(metric_keys),
        ]
        for run in runs:
            s = run["rag_track"].get("scores", {})
            cells = [
                f"{s[k]:.3f}" if s.get(k) is not None else "—" for k in metric_keys
            ]
            lines.append(
                f"| {run['config']['label']} | {run['rag_track']['n']} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    lines += [
        "## Deterministic checks (stats + listing queries)",
        "",
        "| Config | n | pass rate | query-type accuracy |",
        "|---|---|---|---|",
    ]
    for run in runs:
        e = run["exact_track"]
        lines.append(
            f"| {run['config']['label']} | {e['n']} | "
            f"{e['pass_rate']:.0%} | {e['type_accuracy']:.0%} |"
        )
    lines.append("")

    # Routing accuracy across *all* query types (not just stats/listing) —
    # a classify_query regression can move a question into the wrong track
    # entirely, which is a different failure mode than a bad retrieval or a
    # failed assertion, so it's reported on its own.
    query_types: list[str] = []
    for run in runs:
        for t in run["routing"]["per_type"]:
            if t not in query_types:
                query_types.append(t)

    lines += [
        "## Routing accuracy (classify_query, all query types)",
        "",
        "| Config | overall | " + " | ".join(query_types) + " |",
        "|---|---|" + "---|" * len(query_types),
    ]
    for run in runs:
        r = run["routing"]
        cells = [
            f"{r['per_type'][t]:.0%}" if r["per_type"].get(t) is not None else "—"
            for t in query_types
        ]
        lines.append(f"| {run['config']['label']} | {r['overall']:.0%} | " + " | ".join(cells) + " |")
    lines.append("")

    routing_failures = []
    for run in runs:
        for m in run["routing"]["mismatches"]:
            routing_failures.append(
                f"- `{m['id']}` ({run['config']['label']}): expected `{m['expected']}`, got `{m['actual']}`"
            )
    if routing_failures:
        lines += ["## Routing mismatches", "", *routing_failures, ""]

    errored = []
    for run in runs:
        ids = run["routing"].get("errored", [])
        if ids:
            errored.append(f"- {run['config']['label']}: {', '.join(ids)}")
    if errored:
        lines += [
            "## Errored (excluded from routing accuracy and both scoring tracks)",
            "",
            *errored,
            "",
        ]

    failures = []
    for run in runs:
        for row in run["exact_track"]["rows"]:
            if not row["passed"]:
                failures.append(
                    f"- `{row['id']}` ({run['config']['label']}): missing {row['missing']}"
                )
    if failures:
        lines += ["## Failed assertions", "", *failures, ""]

    return "\n".join(lines)


def save_results(runs: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = RESULTS_DIR / f"eval-{stamp}.json"
    json_path.write_text(json.dumps(runs, indent=2))

    md_path = RESULTS_DIR / f"eval-{stamp}.md"
    md_path.write_text(render_markdown(runs))

    latest = RESULTS_DIR / "latest.md"
    latest.write_text(render_markdown(runs))

    return json_path, md_path


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_one_config(
    questions: list[dict],
    label: str,
    backend: str,
    model: str,
    top_k: int,
    track: str,
    judge_backend: str,
    judge_model: str | None,
    max_workers: int,
    throttle: float,
) -> dict:
    print(f"\n=== {label} ===")
    print(f"Generating answers (backend={backend}, model={model}, top_k={top_k})")

    records = generate_predictions(questions, backend, model, top_k, throttle)

    routing = score_routing(records)

    rag_records = [r for r in records if r["expected_type"] in RAG_TRACK_TYPES]
    exact_records = [r for r in records if r["expected_type"] in EXACT_TRACK_TYPES]

    exact = score_exact_track(exact_records) if track in {"all", "exact"} else {
        "n": 0, "pass_rate": 0.0, "type_accuracy": 0.0, "rows": []
    }

    if track in {"all", "rag"}:
        print(f"\nScoring {len(rag_records)} samples with RAGAS "
              f"(judge={judge_backend}, workers={max_workers})")
        ragas_scores = score_rag_track(rag_records, judge_backend, judge_model, max_workers)
    else:
        ragas_scores = {"n": 0, "skipped": [], "scores": {}, "rows": []}

    return {
        "config": {
            "label": label,
            "backend": backend,
            "model": model,
            "top_k": top_k,
            "judge_backend": judge_backend,
            "embed_model": rag.EMBED_MODEL,
        },
        "rag_track": ragas_scores,
        "exact_track": exact,
        "routing": routing,
        "records": [{k: v for k, v in r.items() if k != "contexts"} for r in records],
    }


def main() -> None:
    cfg = load_eval_config()
    load_eval_secrets(cfg)  # populate os.environ from eval/secrets.toml, if present

    eval_cfg = cfg.get("eval", {})
    llm_cfg = cfg.get("llm", {})
    judge_cfg = cfg.get("judge", {})

    p = argparse.ArgumentParser(description="Evaluate the Research Paper RAG system")
    p.add_argument("--backend", default=None,
                   help="Generator backend (default: eval/config.toml [llm].backend, else auto)")
    p.add_argument("--model", default=None, help="Generator model override")
    p.add_argument("--top-k", type=int, default=eval_cfg.get("top_k", rag.TOP_K))
    p.add_argument("--sweep-top-k", type=int, nargs="+", metavar="K",
                   help="Run the eval once per K and emit a comparison table")
    p.add_argument("--track", choices=["all", "rag", "exact"],
                   default=eval_cfg.get("track", "all"))
    p.add_argument("--judge-backend", default=judge_cfg.get("backend", "auto"),
                   choices=["auto", "gemini", "openai", "anthropic", "cohere", "groq", "mistral"],
                   help="Judge backend for RAGAS. 'auto' (default) prefers a "
                        "backend different from the generator; see eval/README.md.")
    p.add_argument("--judge-model", default=judge_cfg.get("model") or None)
    p.add_argument("--max-workers", type=int, default=eval_cfg.get("max_workers", 1),
                   help="RAGAS parallelism. Keep at 1 for free-tier judges.")
    p.add_argument("--throttle", type=float, default=eval_cfg.get("throttle", 1.0),
                   help="Seconds to sleep between generation calls")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    args = p.parse_args()

    questions = load_golden_set()
    if args.limit:
        questions = questions[: args.limit]

    backend, model = rag.resolve_backend(args.backend or llm_cfg.get("backend") or "auto")
    if args.model:
        model = args.model
    elif not args.backend and llm_cfg.get("model"):
        model = llm_cfg["model"]

    if args.track in {"all", "rag"}:
        judge_backend, judge_model = resolve_judge_backend(args.judge_backend, backend)
        if args.judge_model:
            judge_model = args.judge_model
    else:
        judge_backend, judge_model = args.judge_backend, args.judge_model

    print(f"Generator: backend={backend} model={model}")
    if args.track in {"all", "rag"}:
        print(f"Judge:     backend={judge_backend} model={judge_model}")

    top_ks = args.sweep_top_k or [args.top_k]
    runs = []
    for k in top_ks:
        runs.append(
            run_one_config(
                questions=questions,
                label=f"top_k={k}",
                backend=backend,
                model=model,
                top_k=k,
                track=args.track,
                judge_backend=judge_backend,
                judge_model=judge_model,
                max_workers=args.max_workers,
                throttle=args.throttle,
            )
        )

    json_path, md_path = save_results(runs)
    print("\n" + render_markdown(runs))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
