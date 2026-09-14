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

### 1. Retrieval Depth Sweep ($k=3, 5, 8, 12$)

Initial single-pass ablation across retrieval depth configurations ($n=18$ scorable queries per run):

| Config | n (runs) | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Assertion Pass Rate | Routing Accuracy |
|---|---|---|---|---|---|---|---|
| `top_k=3` | 1 | 0.650 | 0.802 | 0.632 | 0.528 | 100% (8/8) | 100% (26/26) |
| `top_k=5` | 2 (mean) | 0.582 | 0.861 | 0.627 | 0.604 | 100% (8/8) | 100% (26/26) |
| `top_k=8` | 3 (mean) | 0.666 | 0.846 | 0.642 | 0.690 | 100% (8/8) | 100% (26/26) |
| `top_k=12` | 1 | 0.657 | 0.827 | 0.619 | 0.639 | 100% (8/8) | 100% (26/26) |

Across repeated runs, `top_k=8` achieved the highest mean score on three of the four RAGAS metrics (faithfulness, context precision, and context recall; `top_k=5` achieved highest mean answer relevancy). However, as shown below, the within-configuration variability across repeated runs was substantial, and the available data do not establish statistically significant superiority over `top_k=12` or `top_k=5`. We therefore treat `top_k=8` as the candidate operating point for confirmatory paired evaluation on a larger query set.

Only two structural observations are unambiguous:
1. **$k=3$ under-retrieves**: context recall is the lowest across every evaluated run (0.528), demonstrating context starvation when retrieval depth is too shallow.
2. **Deterministic assertions and routing show zero variance**: All 7 runs achieved 100% pass rates (8/8 assertions, 26/26 routing decisions), confirming that system logic and routing are robust while LLM-judged metrics carry nearly all the observed uncertainty.

### 2. Multi-Run Variability at the Candidate Operating Point ($n=3$ Runs at $k=8$)

Evaluating `top_k=8` across three identical runs (Groq generator, Gemini Flash Lite judge) reveals substantial stochastic drift:

| Metric | Run 1 | Run 2 | Run 3 | Mean ($\bar{x}$) | Sample Std Dev ($s$) | Observed Range $[x_\min, x_\max]$ | Range Spread |
|---|---|---|---|---|---|---|---|
| **Faithfulness** | 0.767 | 0.590 | 0.640 | 0.666 | 0.091 | $[0.590, 0.767]$ | 0.176 |
| **Answer Relevancy** | 0.900 | 0.804 | 0.835 | 0.846 | 0.049 | $[0.804, 0.900]$ | 0.096 |
| **Context Precision** | 0.655 | 0.585 | 0.687 | 0.642 | 0.052 | $[0.585, 0.687]$ | 0.102 |
| **Context Recall** | 0.833 | 0.611 | 0.625 | 0.690 | 0.124 | $[0.611, 0.833]$ | 0.222 |

**Key Finding — Noise Overwhelms Single-Run Margins:**
- The margins of `top_k=8`'s mean over `top_k=12` (+0.009 faithfulness, +0.019 relevancy, +0.023 precision, +0.051 recall) range between **0.10 and 0.44 standard deviations** of `top_k=8` itself.
- On every metric, the within-configuration range at $k=8$ ($0.096$–$0.222$) exceeds the descriptive margin between configurations ($0.009$–$0.051$). The appearance of a clean peak in single-pass sweeps was heavily influenced by Run 1 drawing high (e.g., recall 0.833 vs. 0.611–0.625 on subsequent runs).
- `[min, max]` reflects the observed sample range of this specific test run, not a formal confidence interval.

### 3. Systematic Judge Calibration vs. Run-to-Run Drift

Comparing $k=5$ evaluated by **Cohere** (`command-r7b-12-2024`, $n=1$) vs. **Gemini** (`gemini-3.5-flash-lite`, $n=2$ mean):

| Metric | Cohere Judge ($n=1$) | Gemini Judge (mean, $n=2$) | $|\Delta|$ (Judge Difference) | $k=8$ Run-to-Run Std Dev ($s$) | Ratio ($|\Delta| / s$) |
|---|---|---|---|---|---|
| Faithfulness | 0.840 | 0.582 | 0.258 | 0.091 | $\approx 2.8\times$ |
| Context Recall | 0.917 | 0.604 | 0.313 | 0.124 | $\approx 2.5\times$ |
| Context Precision | 0.752 | 0.627 | 0.125 | 0.052 | $\approx 2.4\times$ |
| Answer Relevancy | 0.777 | 0.861 | 0.084 | 0.049 | $\approx 1.7\times$ |

**Conclusion:**
1. **Judge differences vs. run variance**: The shift between Cohere and Gemini ($\Delta \approx 0.08$–$0.31$) is approximately **1.7 to 2.8 times the standard deviation** observed across repeated runs. While Cohere shows systematically more lenient grading across recall, precision, and faithfulness, having $n=1$ on Cohere means this delta represents a point comparison rather than an isolated distribution.
2. **Implications for RAG benchmarking**: Most published RAG evaluations run each configuration once and declare an optimal $k$. In reality, single-run differences are easily confounded by generator and judge stochasticity. Rigorous confirmation requires paired query-level testing across matched repetitions.

### 4. Paired Query-Level Analysis ($k=8$ Run 1 vs. $k=12$)

To evaluate retrieval depth differences without discarding query pairing, we analyze matched query pairs between `top_k=8` (Run 1) and `top_k=12` from the saved run records. 

Because evaluation metrics are bounded in $[0, 1]$ and exhibit heavy tie rates, the non-parametric **Wilcoxon signed-rank test** is pre-specified as the primary hypothesis test, with a paired Student's $t$-test included as a secondary check. Each query serves as its own control ($D_i = k_{8,i} - k_{12,i}$):

| Metric | Matched Pairs ($N$) | Retention Rate | Non-Zero Pairs ($n$) | Ties ($D_i = 0$) | Mean Diff ($\bar{D}$) | Wilcoxon Signed-Rank ($p$-value) | Wilcoxon Floor ($2^{1-n}$) | Paired $t$-test ($p$-value) |
|---|---|---|---|---|---|---|---|---|
| **Faithfulness** | 9 / 18 | 50.0% | 7 | 2 | +0.088 | $W = 9.0$, $p = 0.469$ | 0.016 | $t = 1.11$, $p = 0.300$ |
| **Answer Relevancy** | 9 / 18 | 50.0% | 8 | 1 | -0.005 | $W = 16.0$, $p = 0.844$ | 0.008 | $t = -0.23$, $p = 0.826$ |
| **Context Precision** | 8 / 18 | 44.4% | 3 | 5 | +0.132 | $W = 0.0$, **$p = 0.250$** | **0.250** | $t = 1.06$, $p = 0.324$ |
| **Context Recall** | 9 / 18 | 50.0% | 1 | 8 | +0.111 | $W = 0.0$, **$p = 1.000$** | **1.000** | $t = 1.00$, $p = 0.347$ |

**Key Findings & Methodological Insights:**

1. **High Tie Rate Reveals Retrieval Redundancy (The Real Mechanism)**:
   - On Context Recall, **8 of 9 matched queries tied exactly** ($D_i = 0$). On Context Precision, **5 of 8 tied exactly**.
   - This demonstrates that chunks ranked 9 through 12 rarely alter what the generator uses or what the judge considers relevant. For the vast majority of queries in this corpus, marginal chunks past $k=8$ are either redundant duplicates or discarded distractors.
2. **Two Tests Were Mathematically Incapable of Reaching Significance**:
   - Wilcoxon's minimum achievable two-sided $p$-value on $n$ non-zero pairs is $2^{1-n}$.
   - For Context Recall ($n=1$), $p_\text{floor} = 2^0 = 1.000$. For Context Precision ($n=3$), $p_\text{floor} = 2^{-2} = 0.250$.
   - On both metrics, every single non-zero pair favored $k=8$ (the most extreme outcome the test can produce), yet the observed $p$-values equaled the mathematical floor of the test. Saying "not statistically significant" on recall and precision reflects a lack of statistical power under heavy ties, not evidence of equivalence.
3. **Pair Loss and Survivorship Bias**:
   - Judge parse timeouts on complex multi-statement content queries reduced the evaluation set from 18 to 8–9 matched pairs (44%–50% retention).
   - Because failures clustered on complex queries, surviving pairs skewed toward simpler queries where configurations are most likely to agree, further inflating the observed tie rate.
4. **Roadmap for Formal Confirmation**:
   - **Average over matched repeated runs ($M \ge 3$)** before pairing to smooth out generator phrasing stochasticity and judge parse failures.
   - **Expand the benchmark query set ($N \ge 40$)** to guarantee sufficient non-zero differences to break below $\alpha = 0.05$ on Wilcoxon tests.
   - **Pre-specify non-parametric paired tests with Holm-Bonferroni correction** across comparative baselines ($k=8$ vs. $k=3, 5, 12$).
