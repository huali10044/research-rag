# RAG Improvement Revision Plan

## Motivation

The current app fails on several classes of questions:

1. **Corpus-level factual queries** — "how many papers did Hua publish?", "when was the last
   paper published?" — fail because semantic search returns 4 chunks from a single paper and
   the LLM cannot count or sort across the full corpus.
2. **Semantic search returns redundant chunks** — all retrieved chunks come from the same paper,
   so the LLM never sees diverse results.
3. **PDF metadata is incomplete** — PDFs ingested outside `papers_metadata.py` carry no year,
   venue, or author fields in their chunk metadata, so citations show "(?)" for year.
4. **Conversational commands routed to RAG** — phrases like "rerun last query" reach the vector
   search pipeline and return nonsensical answers ("I cannot fulfill this request").

---

## Implementation Status

| # | Change | Status |
|---|---|---|
| 1 | `ingest.py` — tiered PDF metadata extraction (regex → spaCy → LLM) | ✅ Done |
| 2 | `ingest.py` — corpus statistics saved to `data/corpus_stats.json` | ✅ Done |
| 3 | `rag.py` — query classifier (stats / listing / metadata / content / command) | ✅ Done |
| 4 | `rag.py` — adaptive per-paper deduplication (similarity-gap–based cap) | ✅ Done |
| 5 | `rag.py` — corpus context injection for stats/listing/metadata queries | ✅ Done |
| 6 | `rag.py` — `adaptive_answer()` public API | ✅ Done |
| 7 | `app.py` — use `adaptive_answer()`, surface query type in UI | ✅ Done |
| 8 | `app.py` — resolve "rerun" / "repeat" commands to prior question | ✅ Done |
| 9 | Re-ingest with `--reset` to populate `corpus_stats.json` | ⏳ Pending |

---

## Change 1 — `src/ingest.py`: Richer Metadata + Corpus Statistics

### 1a. Automatic PDF Metadata Extraction

For each PDF, extract **title, authors, year, venue** through a tiered pipeline. Results are
merged field-by-field — higher tiers overwrite lower tiers per field:

```
Tier 4 (always):   Regex / filename heuristics
  - Year:  YYYY- prefix in filename, or copyright/proceedings patterns in first 2 pages

Tier 2 (optional): spaCy NLP  [--no-spacy to disable]
  - NER with en_core_web_sm/md/lg
  - PERSON entities → authors, DATE entities → year, ORG/pattern matching → venue
  - Rule-based patterns: "Proceedings of ...", "In: ...", IEEE/ACM org names
  - Runs offline, no API cost

Tier 3 (optional): LLM extraction  [--no-llm to disable]
  - JSON prompt: { title, authors, year, venue }
  - Tries GEMINI_API_KEY → ANTHROPIC_API_KEY → GROQ_API_KEY in order
  - Cached to data/pdf_metadata_cache.json — re-ingest does not re-call the API
```

New chunk metadata fields added for PDF-sourced chunks: `year`, `venue`, `authors`.

CLI flags: `python src/ingest.py --no-spacy` / `--no-llm` to control which tiers run.

### 1b. Corpus Statistics File

After every ingest run, writes `data/corpus_stats.json`:

```json
{
  "generated_at": "2025-01-01T00:00:00",
  "total_papers": 13,
  "year_range": { "min": 2003, "max": 2015 },
  "total_words": 68400,
  "papers": [
    {
      "title": "Adaptive Interest Modeling ...",
      "authors": ["Hua Li", "..."],
      "year": 2014,
      "venue": "MILCOM 2014",
      "word_count": 5200,
      "source": "metadata"
    }
  ]
}
```

Deduplication: when the same paper exists as both a PDF and a `papers_metadata.py` entry,
the `metadata` source (manually curated) wins.

### 1c. New Optional Dependencies

```
spacy>=3.7
# python -m spacy download en_core_web_sm
```

Code degrades gracefully if spaCy is absent — falls through to LLM / regex tiers.

---

## Change 2 — `src/rag.py`: Adaptive Query Routing

### 2a. Query Classifier

`classify_query(query) -> str` returns one of five types:

| Type | Trigger examples | Routing |
|---|---|---|
| `command` | "rerun last query", "try again", "repeat that", "redo" | no RAG; app.py resolves by re-running prior question |
| `stats` | "how many papers", "when was the last paper", "total word count" | skip vector search; answer from `corpus_stats.json` |
| `listing` | "list all papers", "what papers did Hua publish" | skip vector search; inject full paper list as context |
| `metadata` | "who wrote X", "where was X published", "what venue" | vector search + paper list header for grounding |
| `content` | "how does user modeling work", "what methods were used" | vector search with per-paper deduplication |

Classification is checked in the order above (`command` first) so meta-commands are never
forwarded to the embedding pipeline.

### 2b. Adaptive Per-Paper Deduplication

Fetches `top_k × 4` raw candidates (up to 40), then applies a **similarity-gap–based cap**:

- Papers whose best chunk score is within **0.15** of the top score are treated as co-dominant
  and can contribute up to `top_k` chunks (effectively uncapped).
- Papers further below the top score are capped at **2 chunks** to prevent low-relevance papers
  from crowding out the dominant one.
- If all retrieved chunks come from a single paper, no cap is applied at all.

The total chunks returned is always `top_k` (enforced by an early-exit `break`). The per-paper
cap is a ceiling on each paper's *contribution* to that fixed total, not an additive budget.

| Scenario | Dominant paper cap | Other paper cap | Total returned |
|---|---|---|---|
| One paper dominates (gap > 0.15) | `top_k` | 2 | `top_k` |
| Several co-relevant papers (gap ≤ 0.15) | `top_k` | `top_k` | `top_k` |
| Only one unique paper in results | — | — | `top_k` |

Effect:
- *Single-topic query* (one paper far ahead in similarity) → that paper fills all `top_k` slots;
  lower-scoring papers may get 0–2 slots depending on where they fall in the sorted order.
- *Multi-topic query* (several papers score closely) → results flow in natural similarity order
  with no artificial cap suppressing any of them.

Constants `SIMILARITY_GAP = 0.15` and `FALLBACK_CAP = 2` are defined at the top of
`_retrieve_deduped()` for easy tuning.

### 2c. Corpus Context Injection

`build_corpus_context(stats, verbose)` formats `corpus_stats.json` as a plain-text block:

```
CORPUS OVERVIEW
Total papers: 13
Publication years: 2003 – 2015
Total words across all papers: 68,400

PAPERS (sorted by year):
- [2003] An Adaptive Nearest Neighbor Search ... | KDD 2003 | 4,200 words
  Authors: Rafael Alonso, Jeffrey A. Bloom, Hua Li
...
```

Injected at the top of LLM context for `stats`, `listing`, and `metadata` queries.

### 2d. Adaptive Retrieval Routing Table

| Query type | `chunks` | `extra_context` |
|---|---|---|
| `command` | `[]` | `""` (app handles it) |
| `stats` | `[]` | full corpus context |
| `listing` | `[]` | full corpus context |
| `metadata` | deduped vector results | abbreviated paper list |
| `content` | deduped vector results | `""` |

### 2e. Context Length

`MAX_CONTEXT_LEN` raised from 4 000 → 12 000 characters. The corpus overview block alone
is ~1 500 chars; 4 000 was too small to fit both it and source chunks.

### 2f. Public API

```python
def adaptive_answer(
    query: str, embedder, collection, top_k: int, backend: str, model: str
) -> tuple[str, list[dict], str]:
    """Returns (answer, chunks, query_type)."""
```

---

## Change 3 — `app.py`: Command Handling + UI Updates

- **Command resolution**: before calling `adaptive_answer`, checks `classify_query(prompt)`.
  If `"command"`, scans session history for the last non-command user message and re-runs that
  instead. Falls back to the original prompt if no prior question exists.
- **Single entry point**: replaced the 3-step `retrieve → build_context → generate_answer`
  with a single `adaptive_answer()` call.
- **Query type in UI**: sources expander now shows the classification —
  `Sources · stats query`, `Sources · content query · 4 chunks`,
  `Sources · stats query` (with note: "Answer derived from corpus metadata").
- **`query_type` stored** in `st.session_state.messages` for consistent history display.

---

## File Change Summary

| File | Change type | Key additions / modifications |
|---|---|---|
| `src/ingest.py` | Rewrite | `extract_pdf_metadata()`, `_extract_with_spacy()`, `_extract_with_llm()`, `compute_and_save_corpus_stats()` |
| `src/rag.py` | Additions + edits | `classify_query()` (5 types), `adaptive_retrieve()`, `build_corpus_context()`, `adaptive_answer()`, `MAX_CONTEXT_LEN` → 12 000 |
| `app.py` | Targeted edits | command resolution, `adaptive_answer()`, query type in expander |
| `data/corpus_stats.json` | Generated | created / updated on every ingest run |
| `data/pdf_metadata_cache.json` | Generated | LLM extraction cache (avoids repeat API calls) |
| `docs/revision-plan.md` | New | this file |

---

## Rollout

1. `python src/ingest.py --reset` — rebuild vector store, extract metadata, generate `corpus_stats.json`
2. Restart Streamlit — `rag.py` and `app.py` changes take effect immediately, no re-ingest needed
