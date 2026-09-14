# Research Paper RAG — Dr. Hua Li

A retrieval-augmented generation (RAG) system that lets you ask questions about your own published research. Built as a portfolio project demonstrating modern AI engineering skills.

## What it does

- Ingests your research papers (PDFs or structured metadata)
- Chunks and embeds them using `sentence-transformers`
- Stores vectors in ChromaDB (local, no infra needed)
- Retrieves semantically relevant chunks for any query
- Generates grounded answers using Claude or GPT-4o-mini

## Stack

| Layer | Technology | Why |
|---|---|---|
| Embedding | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, fast, high quality |
| Vector store | ChromaDB | Zero infra — runs in-process |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Overlap-aware, preserves context |
| LLM Generator | Multi-backend (Groq, Gemini, Anthropic Claude, OpenAI, Mistral, Cohere, local) | Resilient with automatic provider fallback |
| Evaluation | RAGAS + deterministic substring assertions | Two-track quality and routing verification |
| UI | Streamlit | Fast to build, easy to demo |

## Setup

```bash
# 1. Clone and create environment
git clone https://github.com/huali10044/research-rag.git
cd research-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (pick one)
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# 4. (Optional) Add PDFs
cp /path/to/your/papers/*.pdf data/papers/

# 5. Run ingestion
python src/ingest.py

# 6. Query from command line
python src/rag.py --query "What work have I done on user modeling?"

# Or start interactive mode
python src/rag.py --interactive

# Or launch the web UI
streamlit run app.py
```

## Note on the vector store

`data/chroma_db/` (along with `data/corpus_stats.json` and `data/pdf_metadata_cache.json`) is committed to this repo rather than gitignored. That's intentional: this app deploys straight to Streamlit Cloud with no separate build/ingest step, so the vector store needs to already be present in the repo for the deployed app to work. If you fork this and add your own papers, just re-run `python src/ingest.py --reset` locally and commit the regenerated files.

## Project structure

```
research-rag/
├── app.py                    # Streamlit web UI
├── requirements.txt
├── src/
│   ├── ingest.py             # Ingestion pipeline (load → chunk → embed → store)
│   ├── rag.py                # Retrieval + generation pipeline
│   └── papers_metadata.py    # Structured paper metadata
├── data/
│   ├── papers/               # Drop your PDFs here
│   ├── processed/            # Intermediate outputs
│   └── chroma_db/            # Vector store (auto-created)
└── notebooks/
    └── exploration.ipynb     # End-to-end walkthrough + evaluation
```

## Sample queries

```bash
python src/rag.py --query "What work have I done on user modeling?"
python src/rag.py --query "Summarize my DARPA-related publications"
python src/rag.py --query "How does my 2014 MILCOM paper relate to RAG systems?"
python src/rag.py --query "What methods did I use for adaptive information retrieval?"
python src/rag.py --query "Connect my early KDD work to modern embedding-based search"
```

## Adding more papers

Open `src/papers_metadata.py` and add entries to the `PAPERS` list:

```python
{
    "title":    "Your Paper Title",
    "authors":  ["Author 1", "Author 2"],
    "year":     2024,
    "venue":    "Conference Name",
    "doi":      "10.xxxx/xxxxx",
    "abstract": "Full abstract text...",
    "keywords": ["keyword1", "keyword2"],
    "summary":  "Your own notes about this paper.",
},
```

Then re-run ingestion:

```bash
python src/ingest.py   # incremental — only adds new papers
# or
python src/ingest.py --reset   # rebuild from scratch
```

## Architecture & Query Routing

Rather than sending every query blindly to vector search, the system inspects intent and routes queries through specialized strategies:

```
User Query
  │
  ├─► Classify Query Intent (src/rag.py)
  │     │
  │     ├─► "stats"    ──► 0 chunks retrieved; exact corpus facts from corpus_stats.json
  │     ├─► "listing"  ──► 0 chunks retrieved; verified bibliography injected
  │     ├─► "metadata" ──► Vector search + deduplication + corpus overview header
  │     └─► "content"  ──► Vector search + per-paper deduplication cap + corpus overview
  │
  ▼
Embedding & Search
  └─ Embed query (all-MiniLM-L6-v2)
       └─ Cosine search → ChromaDB (top-k candidates)
            └─ Per-paper deduplication (prevents single-paper monopolization)
                 └─ Assemble context prompt with citations
                      └─ LLM Generation (Groq / Gemini / Claude / OpenAI) with automatic fallback
```

### Design decisions worth explaining in interviews

1. **Query-type routing avoids RAG anti-patterns** — Asking "how many papers did I publish?" or "list all papers from 2012" in standard vector RAG frequently fails because chunks can only represent localized passages. Directing corpus-level aggregation queries to deterministic data structures guarantees 100% accuracy and eliminates unnecessary vector search and LLM calls.
2. **Chunk size 800 / overlap 150** — Tested on academic text; larger chunks preserve cohesive argument structure within sections, while overlap prevents splitting mid-equation or mid-citation.
3. **Canonical Work Identification (Deduplication)** — Real-world academic corpora frequently duplicate works across pre-prints, camera-ready PDFs, and metadata entries. Ingesting both would artificially inflate corpus statistics and pollute vector search with duplicate passages. Assigning a canonical `work_id` groups representations of the same work together.
4. **Per-paper retrieval caps** — Prevents long or keyword-dense papers from monopolizing the retrieved top-k context, ensuring topic diversity across Dr. Li's corpus.
5. **Provider resilience with automatic fallback** — Generation falls back gracefully across Groq, Gemini, Anthropic, OpenAI, Mistral, Cohere, and local pipelines if any upstream provider encounters rate limits or outages.

## Evaluation & The Replication Finding

Evaluation is tracked against a 26-question golden set ([eval/golden_set.json](eval/golden_set.json)) covering corpus statistics, publication listings, metadata attribution, technical content synthesis, and adversarial out-of-corpus probes.

Because context-based metrics are undefined when zero chunks are retrieved, evaluation is split into two tracks:
- **Exact track (8 queries)**: Deterministic substring assertions for corpus statistics and paper listings.
- **RAGAS track (18 queries)**: LLM-as-a-judge scoring with `all-MiniLM-L6-v2` embeddings for content and metadata questions.

### 1. Retrieval Depth Sweep ($k=3, 5, 8, 12$)

Generator: Groq (`openai/gpt-oss-20b`) | Judge: Gemini Flash Lite (`gemini-3.5-flash-lite`)

| Retrieval Depth | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Assertion Pass Rate | Routing Accuracy |
|---|---|---|---|---|---|---|
| `top_k=3` ($n=1$) | 0.650 | 0.802 | 0.632 | 0.528 | 100% (8/8) | 100% (26/26) |
| `top_k=5` ($n=2$ mean) | 0.582 | 0.861 | 0.627 | 0.604 | 100% (8/8) | 100% (26/26) |
| `top_k=8` ($n=3$ mean) | 0.666 | 0.846 | 0.642 | 0.690 | 100% (8/8) | 100% (26/26) |
| `top_k=12` ($n=1$) | 0.657 | 0.827 | 0.619 | 0.639 | 100% (8/8) | 100% (26/26) |

### 2. The Replication Finding: Why Standard RAG Ablations Deceive

Most published RAG projects execute a single sweep across $k$, pick the configuration with the highest numbers, and declare an optimal operating point. In our initial single-pass run, `top_k=8` appeared as a sharp, textbook inverted-U optimum:
- Context Recall reached **0.833** (vs. 0.583 at $k=5$ and 0.639 at $k=12$).
- Faithfulness peaked at **0.767** (vs. 0.609 at $k=5$ and 0.657 at $k=12$).

However, **when replicated across multiple independent runs, the apparent peak dissolved into measurement noise**:

| Metric at `top_k=8` | Run 1 | Run 2 | Run 3 | Mean | Sample Std Dev ($s$) | Observed Spread $[x_\min, x_\max]$ |
|---|---|---|---|---|---|---|
| **Faithfulness** | 0.767 | 0.590 | 0.640 | 0.666 | 0.091 | $[0.590, 0.767]$ |
| **Answer Relevancy** | 0.900 | 0.804 | 0.835 | 0.846 | 0.049 | $[0.804, 0.900]$ |
| **Context Precision** | 0.655 | 0.585 | 0.687 | 0.642 | 0.052 | $[0.585, 0.687]$ |
| **Context Recall** | 0.833 | 0.611 | 0.625 | 0.690 | 0.124 | $[0.611, 0.833]$ |

- **Within-config variance overwhelms between-config margins**: The standard deviation at $k=8$ ($s = 0.049\text{--}0.124$) is larger than the descriptive margins between $k=5, 8, 12$ ($0.009\text{--}0.051$). The apparent peak in single-pass evaluation was simply a high draw on Run 1.
- **Paired query testing reveals retrieval redundancy**: Paired comparison between $k=8$ and $k=12$ on the identical queries showed that **8 of 9 queries tied exactly on Context Recall** ($D_i = 0$), and **5 of 8 tied on Context Precision**. The underlying retrieval set for this corpus is largely saturated by $k=8$; marginal chunks ranked 9 through 12 rarely change what the generator uses or what the judge considers relevant.
- **The code vs. judge contrast**: While LLM-judged metrics showed substantial run-to-run drift, **deterministic assertions and routing accuracy stayed at 100% across all 7 evaluation runs with zero variance**.
- **Systematic Judge Calibration**: Switching from Cohere (`command-r7b-12-2024`) to Gemini (`gemini-3.5-flash-lite`) shifted metrics by **$1.7\times\text{--}2.8\times$ the run-to-run standard deviation** (Cohere being systematically lenient; Gemini Flash Lite systematically strict). The choice of evaluation judge moved scores more than the retrieval parameter being evaluated.

For full statistical tables, Wilcoxon test calculations, and reproducibility guidelines, see [eval/README.md](eval/README.md).

## Portfolio deployment

```bash
# Deploy UI to Streamlit Cloud (free)
# 1. Push this repo to GitHub
# 2. Go to share.streamlit.io
# 3. Connect repo, set ANTHROPIC_API_KEY as a secret
# 4. Share the URL in your portfolio / LinkedIn
```

---

*Built as a portfolio project demonstrating RAG engineering: semantic retrieval, vector stores, LLM integration, and production-style Python architecture.*
