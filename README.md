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
| LLM | Anthropic Claude / OpenAI (fallback: local) | Best answer quality |
| UI | Streamlit | Fast to build, easy to demo |

## Setup

```bash
# 1. Clone and create environment
git clone https://github.com/YOUR_USERNAME/research-rag.git
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

## Architecture notes

```
Query
  └─ Embed query (sentence-transformers)
       └─ Cosine search → ChromaDB (top-k chunks)
            └─ Build context string (formatted citations + text)
                 └─ LLM prompt (system + context + question)
                      └─ Grounded answer with citations
```

### Design decisions worth explaining in interviews

1. **Chunk size 800 / overlap 150** — tested on academic text; larger chunks preserve more context per paper section, overlap prevents splitting mid-argument.
2. **`all-MiniLM-L6-v2`** — good quality/speed tradeoff. Swap for `bge-large-en-v1.5` or `text-embedding-3-small` for better retrieval at higher cost.
3. **ChromaDB** — in-process persistence means zero ops cost for a demo. Production upgrade path: Pinecone, Weaviate, or pgvector.
4. **Metadata filtering** — ChromaDB supports filtering by year, venue, keyword. Not yet exposed in the UI — good next feature.
5. **Similarity threshold** — currently returns top-k regardless of score. Adding a min-similarity cutoff (~0.3) prevents hallucination from off-topic queries.

## Next improvements (good interview talking points)

- [ ] Hybrid search: combine BM25 lexical with semantic similarity (reciprocal rank fusion)
- [ ] Query rewriting: use LLM to expand/rephrase the query before retrieval
- [ ] Re-ranking: use a cross-encoder to re-rank retrieved chunks for precision
- [ ] Metadata filters: expose year/venue filter in the UI
- [ ] Evaluation: RAGAS framework for retrieval + faithfulness + answer relevance scores
- [ ] Citation graph: build a network of papers connected by shared themes

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
