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

## Known issues (resolved)

**KI-1 — the corpus contained duplicate works. Fixed.** Several papers were
indexed twice: once from a PDF (with a filename-derived title such as
`2003-An Adaptive Nearest Neighbor Search For A Parts Acquisition Eportal-P693-Alonso`)
and once from structured metadata (`An Adaptive Nearest Neighbor Search for a
Parts Acquisition ePortal`). 15 unique works were producing 21 index entries,
because `title` was the only identifier and it differed between the two
representations.

Fix: every chunk now carries a `work_id` (a canonical slug derived from the
clean title), assigned at ingestion. `papers_metadata.py` entries that also
exist as a PDF declare a `pdf_filename` field, which `ingest.py` uses to give
both representations the same `work_id`. `compute_and_save_corpus_stats`
(ingestion) and `_retrieve_deduped`'s per-paper cap (retrieval) both group by
`work_id` instead of exact title string. `corpus_stats.json` now reports the
true count: **15 papers, 26,890 words** (the old 51,234 double-counted every
merged paper's PDF word count plus its short metadata-blob word count).

While fixing this, the author's Google Scholar profile was used to fill in
authors/venue for 5 papers that previously had none (PDF-only entries with no
`papers_metadata.py` counterpart), and to correctly identify `pro-RAMA_cs.pdf`
as "User Modeling for Contextual Suggestion" (TREC 2014) rather than the
filename-derived "Pro-Rama Cs." All 13 PDFs in the corpus now have a matching
`papers_metadata.py` entry with verified authors and venue.

**KI-2 — references were unverified. Fixed.** All 26 `reference` fields in
`golden_set.json` are now `verified: true`, checked against the papers'
actual content and the author's Google Scholar profile.

## Evaluation Results & Ablation Analysis

Evaluated against the 26-question golden set (`eval/golden_set.json`), covering 8 deterministic questions (statistics and publication listings) and 18 retrieval questions (metadata, content synthesis, and adversarial probes).

Generator: Groq (`openai/gpt-oss-20b`) | Judge: Gemini (`gemini-3.5-flash-lite`, 12 RPM)

### 1. Retrieval Depth Ablation ($k=3, 5, 8, 12$)

| Config | n | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Assertion Pass Rate | Routing Accuracy |
|---|---|---|---|---|---|---|---|
| `top_k=3` | 18 | 0.650 | 0.802 | 0.632 | 0.528 | 100% (8/8) | 100% (26/26) |
| `top_k=5` | 18 | 0.609 | 0.858 | 0.597 | 0.583 | 100% (8/8) | 100% (26/26) |
| **`top_k=8`** | 18 | **0.767** | **0.900** | **0.655** | **0.833** | **100% (8/8)** | **100% (26/26)** |
| `top_k=12` | 18 | 0.657 | 0.827 | 0.619 | 0.639 | 100% (8/8) | 100% (26/26) |

**Key Finding — Finding the Operating Point:**
- **$k=8$ is the clear optimal operating point** for this corpus. It achieves peak context recall (0.833), highest faithfulness (0.767), and highest answer relevancy (0.900).
- **At $k=3$**, retrieval is starved of necessary context: context recall drops to 0.528, forcing the generator to omit key evidence.
- **At $k=12$**, retrieval degrades across all metrics: context precision falls from 0.655 to 0.619, and faithfulness drops from 0.767 to 0.657. This demonstrates the classic "lost in the middle" and context dilution effect: stuffing the prompt with tangential or distractor chunks confuses the generator and lowers answer grounding.

### 2. Evaluator Noise Floor ($k=5$ Repeat Run)

To distinguish meaningful parameter improvements from LLM-as-a-judge stochasticity, $k=5$ was evaluated twice under identical configurations:

| Metric | Run 1 (`02:21`) | Run 2 (`15:54`) | $\Delta$ (Noise Floor) |
|---|---|---|---|
| Faithfulness | 0.609 | 0.554 | -0.055 |
| Answer Relevancy | 0.858 | 0.864 | +0.006 |
| Context Precision | 0.597 | 0.658 | +0.061 |
| Context Recall | 0.583 | 0.625 | +0.042 |

The mean run-to-run noise floor is approximately **$\pm 0.04$** ($\sim 4\%$).

### 3. Systematic Judge Calibration vs. Run-to-Run Noise

Comparing $k=5$ evaluated by **Cohere** (`command-r7b-12-2024`) vs. **Gemini** (`gemini-3.5-flash-lite`):

| Metric | Cohere Judge | Gemini Judge (avg) | $\Delta$ (Judge Shift) | Multiple of Noise Floor |
|---|---|---|---|---|
| Faithfulness | 0.840 | 0.582 | -0.258 | **$\approx 4.7\times$** |
| Context Recall | 0.917 | 0.604 | -0.313 | **$\approx 7.5\times$** |
| Context Precision | 0.752 | 0.627 | -0.125 | **$\approx 2.1\times$** |
| Answer Relevancy | 0.777 | 0.861 | +0.084 | $\approx 2.1\times$ |

**Conclusion:** The metric movement between judge models ($\Delta \approx 0.25$–$0.31$) is **5 to 7 times larger than the noise floor**, confirming that judge variance reflects systematic model calibration (Cohere being substantially more lenient and Gemini noticeably stricter), rather than statistical noise. Meanwhile, the performance gains observed at $k=8$ (+0.25 in recall and +0.16 in faithfulness over $k=5$) far exceed the $\pm 0.04$ noise threshold, confirming $k=8$ as a genuine performance peak.
