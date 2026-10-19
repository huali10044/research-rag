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

    # Three full runs of one configuration (each regenerates and re-judges)
    python eval/run_eval.py --top-k 8 --repeats 3

    # Re-judge saved answers three times, regenerating nothing: judge variance only
    python eval/run_eval.py --replay eval/results/eval-<stamp>.json --repeats 3

    # Refuse to report a mean over a partial question set, and stop the sweep
    python eval/run_eval.py --sweep-top-k 3 5 8 12 --strict-coverage

What each run records
---------------------
Every eval-<stamp>.json run stores the answers and the retrieved contexts, the
generator and judge model, the judge temperature, the RAGAS version (which is
the version of the judge-output parser), the harness commit and a hash of the
golden set. Each score carries a status, so a missing score says whether the
judge API refused the call, the parser could not read the output, or the
metric was undefined. Raw judge outputs go to eval-<stamp>.judge.jsonl.

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

Note that max_workers bounds *concurrency*, not *request rate* — even
sequentially, judge calls completing in a few seconds each will exceed a
20 req/min ceiling. Judge calls are therefore separately rate-limited via
--judge-rpm (per-backend defaults in _JUDGE_RPM_DEFAULTS). This matters most
for --sweep-top-k, which multiplies the judge call count by the number of
configs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any


def _neutralise_nest_asyncio() -> None:
    """
    Keep RAGAS working on Python 3.14.

    ragas 0.2.x calls nest_asyncio.apply() when ragas.executor is imported.
    nest_asyncio is unmaintained and on Python 3.14 its patch leaves
    asyncio.current_task() returning None, so every metric dies at once with
    "RuntimeError: Timeout should be used inside a task" and scores as NaN
    without a single judge call being made.

    nest_asyncio only matters when an event loop is already running (a
    notebook). This harness is a command-line script with no running loop, so
    on 3.14 the patch is turned into a no-op before ragas is imported. Earlier
    Python versions are left exactly as they were.
    """
    if sys.version_info < (3, 14) or "ragas" in sys.modules:
        return
    try:
        import nest_asyncio
    except ImportError:
        return
    nest_asyncio.apply = lambda *args, **kwargs: None


_neutralise_nest_asyncio()


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

# Every judge is built with this temperature. It is written into each result
# file so a score can be traced back to the exact judge settings.
JUDGE_TEMPERATURE = 0.0

# Time limit, in seconds, for one metric on one question. RAGAS applies it to the
# whole metric, not to each judge call, and context precision makes one judge
# call per retrieved chunk (8 at k=8, 12 at k=12) while the other metrics make
# one to three. At 300 s, precision was the only metric to time out whenever the
# judge slowed to about 40 s a call. 900 s leaves room for that; change it with
# --judge-timeout. Recorded in each result file.
JUDGE_TIMEOUT_S = 900


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


# ── Judge rate limiting ───────────────────────────────────────────────────────
# max_workers caps *concurrency*, not *request rate*. Even at max_workers=1,
# sequential judge calls finishing in 3-8s each burst well past a 20 req/min
# free-tier ceiling. A single-config run (72 judge jobs) can slip under the
# wire; a --sweep-top-k run (4 configs, ~288 jobs) will not.
#
# Observed failure mode without this: Cohere started returning 429s
# (x-trial-endpoint-call-limit: 20, x-trial-endpoint-call-remaining: 3) and
# per-job time degraded 8s -> 63s -> 106s as retries compounded, turning a
# ~60 min sweep into a 4+ hour one.
#
# Requests/minute per backend, from each provider's published free-tier limits.
# Deliberately set slightly under the documented ceiling to leave headroom for
# RAGAS's own internal retries, which also consume quota.
_JUDGE_RPM_DEFAULTS = {
    "cohere":    18,   # trial: 20 req/min  (docs.cohere.com/v2/docs/rate-limits)
    "gemini":    12,   # free tier: ~15 req/min, but see the per-day cap note in config.toml
    "groq":      25,   # free tier: ~30 req/min
    "mistral":   10,   # free tier is stricter in practice than documented
    "openai":    60,   # paid tier-1 is far higher; this is just a sane default
    "anthropic": 45,
}


def build_rate_limiter(backend: str, rpm: float | None = None):
    """
    Token-bucket rate limiter for judge calls, or None to disable.

    rpm=0 explicitly disables limiting (useful on paid keys). rpm=None uses
    the backend's default from _JUDGE_RPM_DEFAULTS.
    """
    if rpm is None:
        rpm = _JUDGE_RPM_DEFAULTS.get(backend, 15)
    if not rpm:
        return None

    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        check_every_n_seconds=0.1,
        # Allow a small burst so short gaps between calls aren't wasted, but
        # not enough to blow the per-minute window on its own.
        max_bucket_size=max(1.0, rpm / 10.0),
    )


# ── Judge LLM ─────────────────────────────────────────────────────────────────
# The judge should ideally differ from the generator, so the model is not
# grading its own output. Default is Gemini for both because the free tier makes
# that the only zero-cost option; pass --judge-backend to separate them.

def build_judge_llm(backend: str, model: str | None, judge_rpm: float | None = None):
    from ragas.llms import LangchainLLMWrapper

    rate_limiter = build_rate_limiter(backend, judge_rpm)

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=model or "gemini-3.5-flash-lite",
                google_api_key=os.environ["GEMINI_API_KEY"],
                temperature=JUDGE_TEMPERATURE,
                rate_limiter=rate_limiter,
            )
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(
            ChatOpenAI(model=model or "gpt-4o-mini", temperature=JUDGE_TEMPERATURE,
                       rate_limiter=rate_limiter)
        )

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LangchainLLMWrapper(
            ChatAnthropic(model=model or "claude-3-5-haiku-latest", temperature=JUDGE_TEMPERATURE,
                          rate_limiter=rate_limiter)
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
                temperature=JUDGE_TEMPERATURE,
                rate_limiter=rate_limiter,
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
                temperature=JUDGE_TEMPERATURE,
                rate_limiter=rate_limiter,
            )
        )

    if backend == "mistral":
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(
            ChatOpenAI(
                model=model or "mistral-small-latest",
                api_key=os.environ["MISTRAL_API_KEY"],
                base_url="https://api.mistral.ai/v1",
                temperature=JUDGE_TEMPERATURE,
                rate_limiter=rate_limiter,
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


# ── Judge tracing ─────────────────────────────────────────────────────────────
# RAGAS returns one number per (question, metric), and NaN for anything that
# went wrong. A NaN can mean the judge API refused the call, the judge answered
# but the parser could not read it, or the metric had nothing to score. Those
# are three different problems, and pooling them makes a mean over the
# surviving questions look like a mean over all of them.
#
# The trace below listens to RAGAS's own callback tree (evaluation → row →
# metric → prompt → LLM call) and records, for every judge call, which question
# and metric it belonged to, what the judge returned, and why a metric failed.

MISSING_REASONS = {
    "rate_limit":       "judge API refused the call (quota or rate limit)",
    "timeout":          "metric ran past the per-question time limit "
                        "(--judge-timeout)",
    "parse_failure":    "judge answered but the RAGAS parser could not read it",
    "judge_error":      "judge call failed for another reason",
    "harness_error":    "the metric failed before any judge call was made "
                        "(a harness or library problem, not the judge)",
    "undefined":        "metric returned no value and raised no error",
    "generation_error": "the generator failed, so there was nothing to score",
    "no_context":       "retrieval returned no chunks",
    "unrecorded":       "result file predates reason tracking",
}


def classify_judge_error(type_name: str, message: str) -> str:
    """Map an exception raised inside a RAGAS metric to a MISSING_REASONS key."""
    text = f"{type_name} {message}".lower()
    if "outputparser" in text or "failed to parse" in text:
        return "parse_failure"
    if any(s in text for s in ("429", "rate limit", "ratelimit", "quota",
                               "resource exhausted", "resourceexhausted",
                               "too many requests")):
        return "rate_limit"
    if "timeout" in type_name.lower() or "timed out" in text:
        return "timeout"
    return "judge_error"


def build_judge_trace(question_ids: list[str], keep_prompts: bool = False):
    """
    Return a LangChain callback handler that attributes every judge call to a
    (question, metric) pair. Built lazily so --track exact needs no langchain.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class JudgeTrace(BaseCallbackHandler):
        raise_error = False

        def __init__(self) -> None:
            self.chains: dict[Any, tuple] = {}   # run_id → (name, parent, type, row)
            self.calls: dict[Any, dict] = {}     # llm run_id → call record
            self.metric_errors: dict[tuple, dict] = {}  # (row, metric) → error
            self._t0: dict[Any, float] = {}      # llm run_id → monotonic start

        def _finish(self, run_id) -> dict | None:
            call = self.calls.get(run_id)
            t0 = self._t0.pop(run_id, None)
            if call is not None and t0 is not None:
                call["duration_s"] = round(time.monotonic() - t0, 2)
            return call

        def _locate(self, run_id) -> tuple[int | None, str | None, str | None]:
            """Walk up the run tree to find the row, metric and prompt names."""
            row = metric = prompt = None
            for _ in range(50):
                if run_id is None or run_id not in self.chains:
                    break
                name, parent, ctype, row_index = self.chains[run_id]
                if ctype == "ragas_prompt" and prompt is None:
                    prompt = name
                elif ctype == "metric":
                    metric = name
                elif ctype == "row":
                    row = row_index
                run_id = parent
            return row, metric, prompt

        def on_chain_start(self, serialized, inputs, *, run_id,
                           parent_run_id=None, metadata=None, **kwargs):
            metadata = metadata or {}
            ctype = metadata.get("type")
            ctype = getattr(ctype, "value", ctype)
            name = (serialized or {}).get("name") or kwargs.get("name")
            self.chains[run_id] = (name, parent_run_id, ctype, metadata.get("row_index"))

        def on_chain_error(self, error, *, run_id, **kwargs):
            entry = self.chains.get(run_id)
            if entry and entry[2] == "metric":
                row, metric, _ = self._locate(run_id)
                self.metric_errors[(row, metric)] = {
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                }

        def _start_call(self, run_id, parent_run_id, prompt_text: str) -> None:
            row, metric, prompt = self._locate(parent_run_id)
            known = row is not None and 0 <= row < len(question_ids)
            call = {
                "question_id": question_ids[row] if known else None,
                "metric": metric,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "prompt_chars": len(prompt_text),
                "output": None,
                "error": None,
                # Wall-clock time from the call's start to its end or error.
                # It includes waiting on the rate limiter and any retries the
                # client makes internally, so it is what the time limit sees.
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": None,
            }
            self._t0[run_id] = time.monotonic()
            if keep_prompts:
                call["prompt_text"] = prompt_text
            self.calls[run_id] = call

        def on_chat_model_start(self, serialized, messages, *, run_id,
                                parent_run_id=None, **kwargs):
            text = "\n".join(str(m.content) for batch in messages for m in batch)
            self._start_call(run_id, parent_run_id, text)

        def on_llm_start(self, serialized, prompts, *, run_id,
                         parent_run_id=None, **kwargs):
            self._start_call(run_id, parent_run_id, "\n".join(prompts))

        def on_llm_end(self, response, *, run_id, **kwargs):
            call = self._finish(run_id)
            if call is None:
                return
            texts = [g.text for gens in response.generations for g in gens]
            call["output"] = texts[0] if len(texts) == 1 else texts

        def on_llm_error(self, error, *, run_id, **kwargs):
            call = self._finish(run_id)
            if call is not None:
                call["error"] = f"{type(error).__name__}: {str(error)[:500]}"

    return JudgeTrace()


# ── Coverage ──────────────────────────────────────────────────────────────────
# A mean is only comparable to another mean if both cover the same questions.
# These helpers work on saved result dicts, so they also describe result files
# written before reason tracking existed (reasons show as "unrecorded").

def metric_coverage(rag_track: dict) -> dict[str, dict]:
    """Per metric: how many questions were scored, of how many, and why not."""
    rows = rag_track.get("rows", [])
    skipped = rag_track.get("skipped", [])
    skipped_reasons = rag_track.get("skipped_reasons", {})
    total = len(rows) + len(skipped)

    coverage: dict[str, dict] = {}
    for metric in rag_track.get("scores", {}):
        missing: dict[str, str] = {}
        for row in rows:
            if row.get(metric) is None:
                missing[row["id"]] = row.get("status", {}).get(metric, "unrecorded")
        for qid in skipped:
            missing[qid] = skipped_reasons.get(qid, "unrecorded")
        coverage[metric] = {
            "scored": total - len(missing),
            "total": total,
            "missing": missing,
        }
    return coverage


def is_complete(rag_track: dict) -> bool:
    return all(c["scored"] == c["total"] for c in metric_coverage(rag_track).values())


def matched_question_means(runs: list[dict], metric: str) -> tuple[list[str], list[float | None]]:
    """
    Restrict every run to the questions that *all* runs scored on this metric,
    and return (those question ids, one mean per run). This is the like-for-like
    comparison: without it, a run that lost its harder questions looks better.
    """
    scored_sets = []
    for run in runs:
        rows = run["rag_track"].get("rows", [])
        scored_sets.append({r["id"] for r in rows if r.get(metric) is not None})
    if not scored_sets:
        return [], []
    common = set.intersection(*scored_sets)

    order = [r["id"] for r in runs[0]["rag_track"].get("rows", []) if r["id"] in common]
    means: list[float | None] = []
    for run in runs:
        vals = [r[metric] for r in run["rag_track"].get("rows", []) if r["id"] in common]
        means.append(round(statistics.mean(vals), 4) if vals else None)
    return order, means


# ── Track 2: RAGAS ────────────────────────────────────────────────────────────

# The four RAGAS metrics, by the column name they report under. Only the last
# two read the reference answer, so a reference change only needs those two
# re-scored (--metrics), which is far cheaper in judge calls.
ALL_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]
REFERENCE_METRICS = ["llm_context_precision_with_reference", "context_recall"]


def build_metrics(names: list[str]) -> list:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    factory = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
    }
    return [factory[n]() for n in names]

def score_rag_track(
    records: list[dict],
    judge_backend: str,
    judge_model: str | None,
    max_workers: int,
    judge_rpm: float | None = None,
    keep_prompts: bool = False,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.run_config import RunConfig

    metrics = metrics or ALL_METRICS

    scorable = [r for r in records if r["contexts"] and not r["error"]]
    skipped_reasons = {
        r["id"]: ("generation_error" if r["error"] else "no_context")
        for r in records
        if not r["contexts"] or r["error"]
    }
    skipped = list(skipped_reasons)
    if skipped:
        print(f"  Skipping {len(skipped)} sample(s) with no context or an error: {skipped}")
    if not scorable:
        return {"n": 0, "skipped": skipped, "skipped_reasons": skipped_reasons,
                "scores": {}, "coverage": {}, "complete": not skipped,
                "rows": [], "_judge_calls": []}

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

    llm = build_judge_llm(judge_backend, judge_model, judge_rpm)
    # Only answer relevancy uses embeddings; skip loading the model otherwise.
    embeddings = build_embeddings() if "answer_relevancy" in metrics else None
    trace = build_judge_trace([r["id"] for r in scorable], keep_prompts=keep_prompts)

    # max_workers=1 keeps free-tier judges inside their rate limit; the long
    # max_wait lets tenacity ride out a 429 instead of failing the whole run.
    run_config = RunConfig(
        timeout=JUDGE_TIMEOUT_S,
        max_retries=10,
        max_wait=90,
        max_workers=max_workers,
    )

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(metrics),
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        callbacks=[trace],
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

    attempted = {(c["question_id"], c["metric"]) for c in trace.calls.values()}

    rows = []
    for i, (r, (_, row)) in enumerate(zip(scorable, df.iterrows())):
        entry: dict[str, Any] = {
            "id": r["id"], "n_chunks": r["n_chunks"], "latency_s": r["latency_s"],
        }
        status: dict[str, str] = {}
        errors: dict[str, str] = {}
        for col in metric_cols:
            v = row[col]
            if v is not None and v == v:
                entry[col] = round(float(v), 4)
                status[col] = "ok"
                continue
            entry[col] = None
            err = trace.metric_errors.get((i, col))
            if err:
                # No judge call at all means the failure is ours, not the judge's.
                status[col] = (
                    classify_judge_error(err["type"], err["message"])
                    if (r["id"], col) in attempted
                    else "harness_error"
                )
                errors[col] = f"{err['type']}: {err['message']}"
            else:
                status[col] = "undefined"
        entry["status"] = status
        if errors:
            entry["errors"] = errors
        rows.append(entry)

    track = {"n": len(scorable), "skipped": skipped, "skipped_reasons": skipped_reasons,
             "scores": scores, "rows": rows}
    track["coverage"] = metric_coverage(track)
    track["complete"] = is_complete(track)
    track["_judge_calls"] = list(trace.calls.values())

    harness_errors = sorted({
        row["errors"][col]
        for row in rows
        for col, reason in row["status"].items()
        if reason == "harness_error"
    })
    if harness_errors:
        print("\n  ERROR: some metrics failed before any judge call was made. This is a "
              "harness or library problem, not a judge result:")
        for message in harness_errors[:3]:
            print(f"    {message}")
    return track


# ── Reporting ─────────────────────────────────────────────────────────────────

def render_markdown(runs: list[dict], strict: bool = False) -> str:
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

    coverages = [metric_coverage(run["rag_track"]) for run in runs]
    any_partial = any(
        c["scored"] < c["total"] for cov in coverages for c in cov.values()
    )

    if metric_keys:
        lines += [
            "## Retrieval and generation quality (content + metadata queries)",
            "",
        ]
        if any_partial:
            lines += [
                "A cell written as `mean (scored/total)` covers only some of the "
                "questions. It is not comparable to a mean over all of them; use "
                "the same-question table below.",
                "",
            ]
        lines += [
            "| Config | n | " + " | ".join(metric_keys) + " |",
            "|---|---|" + "---|" * len(metric_keys),
        ]
        for run, cov in zip(runs, coverages):
            s = run["rag_track"].get("scores", {})
            cells = []
            for k in metric_keys:
                c = cov.get(k)
                partial = bool(c) and c["scored"] < c["total"]
                if s.get(k) is None:
                    cell = "—"
                elif partial and strict:
                    cell = "withheld"
                else:
                    cell = f"{s[k]:.3f}"
                if partial:
                    cell += f" ({c['scored']}/{c['total']})"
                cells.append(cell)
            lines.append(
                f"| {run['config']['label']} | {run['rag_track']['n']} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    if any_partial:
        lines += [
            "## Incomplete coverage",
            "",
            "| Config | Metric | Scored | Why the rest are missing |",
            "|---|---|---|---|",
        ]
        for run, cov in zip(runs, coverages):
            for k in metric_keys:
                c = cov.get(k)
                if not c or c["scored"] == c["total"]:
                    continue
                counts: dict[str, int] = {}
                for reason in c["missing"].values():
                    counts[reason] = counts.get(reason, 0) + 1
                why = "; ".join(
                    f"{n} × {MISSING_REASONS.get(reason, reason)}"
                    for reason, n in sorted(counts.items())
                )
                lines.append(
                    f"| {run['config']['label']} | {k} | {c['scored']}/{c['total']} | {why} |"
                )
        lines.append("")

    scored_runs = [run for run in runs if run["rag_track"].get("rows")]
    if any_partial and len(scored_runs) > 1:
        lines += [
            "## Same-question comparison",
            "",
            "Each metric restricted to the questions every run scored, so all "
            "runs are compared on an identical set.",
            "",
            "| Config | " + " | ".join(metric_keys) + " |",
            "|---|" + "---|" * len(metric_keys),
        ]
        matched = {k: matched_question_means(scored_runs, k) for k in metric_keys}
        lines.append(
            "| *questions in common* | "
            + " | ".join(str(len(matched[k][0])) for k in metric_keys)
            + " |"
        )
        for i, run in enumerate(scored_runs):
            cells = [
                f"{matched[k][1][i]:.3f}" if matched[k][1][i] is not None else "—"
                for k in metric_keys
            ]
            lines.append(f"| {run['config']['label']} | " + " | ".join(cells) + " |")
        lines.append("")

    if any("judge_model" in run["config"] for run in runs):
        lines += [
            "## Run configuration",
            "",
            "| Config | Mode | Generator | Judge | Judge temp. | RAGAS |",
            "|---|---|---|---|---|---|",
        ]
        for run in runs:
            c = run["config"]
            prov = run.get("provenance", {})
            lines.append(
                f"| {c['label']} | {c.get('mode', 'generate')} | "
                f"{c.get('backend')}/{c.get('model')} | "
                f"{c.get('judge_backend')}/{c.get('judge_model')} | "
                f"{c.get('judge_temperature')} | {prov.get('ragas')} |"
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


def save_results(
    runs: list[dict], stamp: str | None = None, strict: bool = False
) -> tuple[Path, Path]:
    """
    Write eval-<stamp>.json and .md, plus eval-<stamp>.judge.jsonl holding one
    line per judge call (question, metric, prompt hash, raw output, error).
    Keys starting with "_" are working data and stay out of the main JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    public = [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs]
    json_path = RESULTS_DIR / f"eval-{stamp}.json"
    json_path.write_text(json.dumps(public, indent=2))

    judge_lines = [
        json.dumps({"run": run["config"]["label"], **call})
        for run in runs
        for call in run.get("_judge_calls", [])
    ]
    if judge_lines:
        judge_path = RESULTS_DIR / f"eval-{stamp}.judge.jsonl"
        judge_path.write_text("\n".join(judge_lines) + "\n")

    md_path = RESULTS_DIR / f"eval-{stamp}.md"
    md_path.write_text(render_markdown(public, strict=strict))

    latest = RESULTS_DIR / "latest.md"
    latest.write_text(render_markdown(public, strict=strict))

    return json_path, md_path


# ── Provenance ────────────────────────────────────────────────────────────────

def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(
    question_ids: list[str], golden_set_sha256: str | None = None
) -> dict[str, Any]:
    """
    Everything needed to say which code and which libraries produced a score.

    "Dirty" means the code differs from the recorded commit. The vector store
    and the results folder are excluded: both change on every run, and neither
    is code.
    """
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 ":(exclude)data/chroma_db", ":(exclude)eval/results")
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "harness_commit": _git("rev-parse", "HEAD"),
        "harness_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "ragas": _pkg_version("ragas"),          # the judge-output parser lives here
        "langchain_core": _pkg_version("langchain-core"),
        # Which version of the reference answers was used. For a replay that
        # keeps the source run's references, this is the source run's hash.
        "golden_set_sha256": golden_set_sha256
        or hashlib.sha256(GOLDEN_SET.read_bytes()).hexdigest(),
        "question_ids": question_ids,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def score_records(
    records: list[dict],
    config: dict,
    track: str,
    max_workers: int,
    judge_rpm: float | None = None,
    keep_prompts: bool = False,
    metrics: list[str] | None = None,
    golden_set_sha256: str | None = None,
) -> dict:
    """Score a set of generated (or reloaded) records and assemble one run."""
    judge_backend = config["judge_backend"]
    judge_model = config["judge_model"]

    routing = score_routing(records)

    rag_records = [r for r in records if r["expected_type"] in RAG_TRACK_TYPES]
    exact_records = [r for r in records if r["expected_type"] in EXACT_TRACK_TYPES]

    exact = score_exact_track(exact_records) if track in {"all", "exact"} else {
        "n": 0, "pass_rate": 0.0, "type_accuracy": 0.0, "rows": []
    }

    if track in {"all", "rag"}:
        effective_rpm = (judge_rpm if judge_rpm is not None
                         else _JUDGE_RPM_DEFAULTS.get(judge_backend, 15))
        print(f"\nScoring {len(rag_records)} samples with RAGAS "
              f"(judge={judge_backend}/{judge_model}, workers={max_workers}, "
              f"rate_limit={effective_rpm or 'off'} req/min)")
        ragas_scores = score_rag_track(rag_records, judge_backend, judge_model,
                                       max_workers, judge_rpm, keep_prompts,
                                       metrics)
    else:
        ragas_scores = {"n": 0, "skipped": [], "skipped_reasons": {}, "scores": {},
                        "coverage": {}, "complete": True, "rows": [],
                        "_judge_calls": []}

    judge_calls = ragas_scores.pop("_judge_calls", [])

    if not ragas_scores.get("complete", True):
        for metric, cov in ragas_scores["coverage"].items():
            if cov["scored"] < cov["total"]:
                print(f"  WARNING {config['label']}: {metric} scored on "
                      f"{cov['scored']}/{cov['total']} questions")

    return {
        "config": {
            **config,
            "judge_temperature": JUDGE_TEMPERATURE,
            "judge_timeout_s": JUDGE_TIMEOUT_S,
            "judge_rpm": (judge_rpm if judge_rpm is not None
                          else _JUDGE_RPM_DEFAULTS.get(judge_backend, 15)),
            "embed_model": rag.EMBED_MODEL,
            "metrics": metrics or ALL_METRICS,
        },
        "provenance": collect_provenance([r["id"] for r in records],
                                         golden_set_sha256),
        "rag_track": ragas_scores,
        "exact_track": exact,
        "routing": routing,
        # Contexts are kept: three of the four RAGAS metrics need them, so a
        # judge replay (--replay) is impossible without them.
        "records": records,
        "_judge_calls": judge_calls,
    }


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
    judge_rpm: float | None = None,
    keep_prompts: bool = False,
    metrics: list[str] | None = None,
) -> dict:
    print(f"\n=== {label} ===")
    print(f"Generating answers (backend={backend}, model={model}, top_k={top_k})")

    records = generate_predictions(questions, backend, model, top_k, throttle)
    config = {
        "label": label,
        "mode": "generate",
        "backend": backend,
        "model": model,
        "top_k": top_k,
        "judge_backend": judge_backend,
        "judge_model": judge_model,
    }
    return score_records(records, config, track, max_workers, judge_rpm, keep_prompts,
                         metrics)


def load_replay_runs(path: Path, label: str | None = None) -> list[dict]:
    """
    Load saved runs to re-judge. Each must carry the retrieved contexts, which
    result files written before contexts were saved do not.
    """
    data = json.loads(path.read_text())
    runs = data if isinstance(data, list) else [data]
    if label is not None:
        runs = [r for r in runs if r["config"]["label"] == label]
        if not runs:
            raise SystemExit(f"No run labelled {label!r} in {path}")
    for run in runs:
        rag_records = [r for r in run["records"] if r["expected_type"] in RAG_TRACK_TYPES]
        if any("contexts" not in r for r in rag_records):
            raise SystemExit(
                f"{path} (run {run['config']['label']!r}) was saved without the "
                "retrieved contexts, so the judge cannot be replayed over it. "
                "Generate a fresh run with this version of the harness and "
                "replay that file instead."
            )
    return runs


def refresh_references(records: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Copy the records with each reference (and must_contain list) replaced by
    the current golden set's. Returns the copies and the ids whose reference
    changed. Answers and retrieved contexts are untouched.
    """
    golden = {q["id"]: q for q in load_golden_set(GOLDEN_SET)}
    missing = [r["id"] for r in records if r["id"] not in golden]
    if missing:
        raise SystemExit(
            f"--refresh-references: these saved questions are not in "
            f"{GOLDEN_SET.name}: {missing}"
        )
    refreshed, changed = [], []
    for r in records:
        q = golden[r["id"]]
        new = {**r, "reference": q["reference"],
               "must_contain": q.get("must_contain", [])}
        if new["reference"] != r["reference"]:
            changed.append(r["id"])
        refreshed.append(new)
    return refreshed, changed


def replay_one(
    source: dict,
    source_file: Path,
    label: str,
    track: str,
    judge_backend: str,
    judge_model: str | None,
    max_workers: int,
    judge_rpm: float | None = None,
    keep_prompts: bool = False,
    metrics: list[str] | None = None,
    refresh_refs: bool = False,
) -> dict:
    """
    Re-judge saved answers. Nothing is regenerated or re-retrieved. With
    refresh_refs, the current golden-set references replace the saved ones,
    so a reference rewrite can be measured on identical answers.
    """
    print(f"\n=== {label} ===")
    print(f"Replaying the judge over saved answers from {source_file.name}")
    src = source["config"]
    records = source["records"]
    golden_sha = source.get("provenance", {}).get("golden_set_sha256")
    changed: list[str] = []
    if refresh_refs:
        records, changed = refresh_references(records)
        golden_sha = None  # recorded from the current golden set
        print(f"Using current references: {len(changed)} of {len(records)} changed")
    config = {
        "label": label,
        "mode": "replay",
        "backend": src.get("backend"),
        "model": src.get("model"),
        "top_k": src.get("top_k"),
        "judge_backend": judge_backend,
        "judge_model": judge_model,
        "source_file": source_file.name,
        "source_label": src["label"],
        "references": "current golden set" if refresh_refs else "saved with source run",
        "references_changed": changed,
    }
    return score_records(records, config, track, max_workers,
                         judge_rpm, keep_prompts, metrics, golden_sha)


def main() -> None:
    global JUDGE_TIMEOUT_S
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
    p.add_argument("--judge-rpm", type=float, default=judge_cfg.get("rpm"),
                   help="Judge requests/minute cap. Defaults to a per-backend value "
                        "based on that provider's free-tier limit; pass 0 to disable "
                        "(paid keys). max_workers alone does NOT bound request rate.")
    p.add_argument("--judge-timeout", type=float, default=JUDGE_TIMEOUT_S, metavar="SECONDS",
                   help=f"Time limit for one metric on one question (default "
                        f"{JUDGE_TIMEOUT_S}). Context precision makes one judge call "
                        "per retrieved chunk, so it needs the most time.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    p.add_argument("--repeats", type=int, default=1, metavar="N",
                   help="Run each configuration N times. With --replay this isolates "
                        "judge variance; without it, each repeat regenerates and re-judges.")
    p.add_argument("--replay", type=Path, default=None, metavar="RESULTS_JSON",
                   help="Re-judge the saved answers and contexts in a results file "
                        "instead of generating new answers.")
    p.add_argument("--replay-label", default=None, metavar="LABEL",
                   help="With --replay: only the run with this label (default: every run in the file)")
    p.add_argument("--save-judge-prompts", action="store_true",
                   help="Also store the full judge prompts in the .judge.jsonl file "
                        "(default: raw outputs plus a prompt hash)")
    p.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=None,
                   metavar="METRIC",
                   help="Score only these RAGAS metrics (default: all four). "
                        "Choices: " + ", ".join(ALL_METRICS))
    p.add_argument("--refresh-references", action="store_true",
                   help="With --replay: judge the saved answers against the current "
                        "golden-set references instead of the ones saved with the run.")
    p.add_argument("--strict-coverage", action="store_true",
                   help="Withhold any mean that does not cover every question, and "
                        "stop the sweep after the first incomplete run (exit code 2).")
    args = p.parse_args()

    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    if args.replay and args.track == "exact":
        p.error("--replay re-runs the judge, so it cannot be combined with --track exact")
    if args.refresh_references and not args.replay:
        p.error("--refresh-references only applies to --replay")
    if args.judge_timeout <= 0:
        p.error("--judge-timeout must be positive")
    JUDGE_TIMEOUT_S = args.judge_timeout

    def repeat_label(base: str, i: int) -> str:
        return f"{base} (run {i + 1})" if args.repeats > 1 else base

    jobs: list[tuple[str, Any]] = []   # (label, zero-argument callable → run dict)

    if args.replay:
        for source in load_replay_runs(args.replay, args.replay_label):
            judge_backend, judge_model = resolve_judge_backend(
                args.judge_backend, source["config"].get("backend") or ""
            )
            if args.judge_model:
                judge_model = args.judge_model
            print(f"Judge:     backend={judge_backend} model={judge_model}")
            for i in range(args.repeats):
                label = repeat_label(f"{source['config']['label']}, judge replay", i)
                jobs.append((label, lambda source=source, label=label,
                             jb=judge_backend, jm=judge_model: replay_one(
                    source=source, source_file=args.replay, label=label,
                    track=args.track, judge_backend=jb, judge_model=jm,
                    max_workers=args.max_workers, judge_rpm=args.judge_rpm,
                    keep_prompts=args.save_judge_prompts, metrics=args.metrics,
                    refresh_refs=args.refresh_references,
                )))
    else:
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

        for k in args.sweep_top_k or [args.top_k]:
            for i in range(args.repeats):
                label = repeat_label(f"top_k={k}", i)
                jobs.append((label, lambda k=k, label=label: run_one_config(
                    questions=questions,
                    label=label,
                    backend=backend,
                    model=model,
                    top_k=k,
                    track=args.track,
                    judge_backend=judge_backend,
                    judge_model=judge_model,
                    max_workers=args.max_workers,
                    throttle=args.throttle,
                    judge_rpm=args.judge_rpm,
                    keep_prompts=args.save_judge_prompts,
                    metrics=args.metrics,
                )))

    runs = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stopped_early = False
    for label, job in jobs:
        runs.append(job())
        json_path, md_path = save_results(runs, stamp=stamp, strict=args.strict_coverage)
        print(f"\n[checkpoint] Saved {len(runs)}/{len(jobs)} run(s) to {json_path}")
        if args.strict_coverage and not runs[-1]["rag_track"].get("complete", True):
            stopped_early = True
            print(f"\n--strict-coverage: {label} did not score every question. "
                  f"Stopping before the remaining {len(jobs) - len(runs)} run(s).")
            break

    public = [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs]
    print("\n" + render_markdown(public, strict=args.strict_coverage))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    if stopped_early:
        sys.exit(2)


if __name__ == "__main__":
    main()
