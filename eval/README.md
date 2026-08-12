# Evaluation

Measures retrieval and answer quality for the Research Paper RAG system.

```bash
pip install -r eval/requirements-eval.txt

cp eval/secrets.toml.example eval/secrets.toml
# edit eval/secrets.toml and fill in the key(s) for the backend(s) you use

python eval/run_eval.py                        # full run
python eval/run_eval.py --track exact          # fast, no judge calls
python eval/run_eval.py --sweep-top-k 3 5 8 12 # ablation over retrieval depth
```

Results are written to `eval/results/` as timestamped JSON and Markdown, plus a
rolling `latest.md`.

## Configuration

Same two-file split as the main app (`config.toml` + `.streamlit/secrets.toml`):

| File | Contents | Committed? |
|---|---|---|
| `eval/config.toml` | generator/judge backend + model, `top_k`, track, throttling | yes |
| `eval/secrets.toml` | API keys | no — gitignored |

`eval/secrets.toml` is loaded at startup and injected into `os.environ` under
the same variable names `rag.py` already reads (`GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.), so a key set there works for both the generator
and the RAGAS judge. A key already exported in your shell always takes
precedence over the file.

Every value in `eval/config.toml` can be overridden on the command line —
flags win over the config file:

```bash
python eval/run_eval.py --backend anthropic --judge-backend gemini --top-k 8
```

`[judge].backend = "auto"` (the default) picks a judge backend that differs
from the generator when a key for one is available, falling back to the
generator's own backend only if no independent judge key is configured. See
"Judge independence" below for why that matters.

## Why two tracks

The system routes queries to four strategies, and only two of them retrieve
chunks:

| Query type | Retrieval | Evaluated by |
|---|---|---|
| `content` | vector search + per-paper dedup | RAGAS |
| `metadata` | vector search + corpus header | RAGAS |
| `stats` | none — answered from `corpus_stats.json` | assertions |
| `listing` | none — full paper list injected | assertions |

Every context-based RAGAS metric is undefined when `retrieved_contexts` is
empty. Scoring `stats` and `listing` queries with RAGAS would report 0.0 for a
path that is working exactly as designed, and would drag the headline number
down for the wrong reason. So those queries get deterministic substring
assertions instead — they have exactly one correct answer, drawn from a file we
control, and an LLM judge would add cost and variance without adding
information.

The harness also records whether `classify_query` routed each question to the
expected strategy, across *all* query types (not just stats/listing). Routing
accuracy is reported separately, because a routing regression and a retrieval
regression need different fixes.

## Metrics

| Metric | Question it answers | Fails when |
|---|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved chunks? | The model invents detail |
| Answer relevancy | Does the answer address the question asked? | The model answers a nearby question |
| Context precision | Are the retrieved chunks actually relevant? | Retrieval returns noise |
| Context recall | Did retrieval find everything the reference needs? | `top_k` too low, or chunking splits the answer |

Precision and recall usually trade off against `top_k`. Faithfulness and
context precision moving in opposite directions is the signal worth watching:
it usually means the generator is leaning on parametric knowledge because
retrieval did not give it enough to work with.

Answer relevancy is computed with `all-MiniLM-L6-v2` — the same embedding model
the retriever uses — so the metric lives in the same vector space the system
actually searches in.

## Free-tier rate limits

Gemini's free tier allows roughly 15 requests per minute. RAGAS defaults to 16
parallel workers, which exhausts that in seconds and produces a wall of 429s
that look like evaluation failures. The harness pins `max_workers=1` and
throttles generation. A full 26-question run takes about 10–15 minutes on free
tier. Raise `--max-workers` on a paid key.

## Judge independence

Using the same provider to generate and judge biases faithfulness upward — a
model is a lenient grader of its own output. `--judge-backend auto` (the
default) picks a different backend from the generator whenever a second key
is configured in `eval/secrets.toml`, and only falls back to the generator's
own backend if no independent judge key is available. Pin it explicitly to
force a specific pairing:

```bash
python eval/run_eval.py --backend gemini --judge-backend anthropic
```

Report which configuration produced any number you quote — the config block
in each `eval/results/*.json` run records the generator and judge backend
actually used.

## Known issues

**KI-1 — the corpus contains duplicate works.** `corpus_stats.json` reports 21
entries, but several papers are indexed twice: once from a PDF (with a
filename-derived title such as
`2003-An Adaptive Nearest Neighbor Search For A Parts Acquisition Eportal-P693-Alonso`)
and once from structured metadata (`An Adaptive Nearest Neighbor Search for a
Parts Acquisition ePortal`). Roughly 15 unique works produce 21 index entries.

Consequences:

- `stats-01` asserts 21, which is the honest index-entry count but not the
  paper count a human means when asking "how many papers."
- Duplicate content inflates context precision — two copies of the right chunk
  both count as relevant.
- The per-paper dedup cap in `_retrieve_deduped` keys on `title`, so the PDF and
  metadata versions of one paper are treated as two papers and can both occupy
  slots.

Fix before quoting precision numbers externally: add a canonical work ID during
ingestion and dedup on that rather than on title.

**KI-2 — references are unverified.** Every `reference` in `golden_set.json` is
currently marked `verified: false`. They were drafted from titles and corpus
statistics, not from reading the papers. Context precision and recall are only
as trustworthy as those references. Read through them, correct them, and flip
the flag before treating the numbers as real.
