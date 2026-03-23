"""
ingest.py — Load, chunk, embed, and store research papers into a vector store.

Supports:
  - PDF ingestion (from /data/papers/)
  - Manual metadata ingestion (from papers_metadata.py)
  - ChromaDB as the local vector store (zero infra, runs in-process)
  - sentence-transformers for embeddings (free, no API key needed)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
PAPERS_DIR      = DATA_DIR / "papers"
CHROMA_DIR      = DATA_DIR / "chroma_db"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim

CHUNK_SIZE      = 800   # characters per chunk
CHUNK_OVERLAP   = 150   # overlap to preserve context across chunks


def get_embedder() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print(f"Reset collection: {COLLECTION_NAME}")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, source_meta: dict) -> list[dict]:
    """Split text into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    results = []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{source_meta['title']}_chunk_{i}".encode()).hexdigest()
        results.append({
            "id":       doc_id,
            "text":     chunk,
            "metadata": {**source_meta, "chunk_index": i, "total_chunks": len(chunks)},
        })
    return results


def ingest_pdfs(collection, embedder) -> int:
    """Ingest all PDFs from /data/papers/."""
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR} — skipping PDF ingestion.")
        return 0

    count = 0
    for pdf_path in pdf_files:
        print(f"  Ingesting: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        full_text = "\n\n".join(p.page_content for p in pages)

        meta = {
            "source":    pdf_path.name,
            "title":     pdf_path.stem.replace("_", " ").title(),
            "type":      "pdf",
        }
        chunks = chunk_text(full_text, meta)
        _upsert_chunks(collection, embedder, chunks)
        count += len(chunks)

    return count


def ingest_metadata_papers(collection, embedder) -> int:
    """Ingest papers from structured metadata (papers_metadata.py)."""
    from src.papers_metadata import PAPERS   # lazy import so CLI still works without it

    count = 0
    for paper in PAPERS:
        print(f"  Ingesting: {paper['title'][:60]}...")
        text = _build_paper_text(paper)
        chunks = chunk_text(text, {
            "source":   "metadata",
            "title":    paper["title"],
            "year":     str(paper.get("year", "")),
            "venue":    paper.get("venue", ""),
            "authors":  ", ".join(paper.get("authors", [])),
            "doi":      paper.get("doi", ""),
            "abstract": paper.get("abstract", ""),
            "keywords": ", ".join(paper.get("keywords", [])),
        })
        _upsert_chunks(collection, embedder, chunks)
        count += len(chunks)

    return count


def _build_paper_text(paper: dict) -> str:
    """Construct a rich text blob from paper metadata for embedding."""
    parts = [
        f"Title: {paper['title']}",
        f"Authors: {', '.join(paper.get('authors', []))}",
        f"Year: {paper.get('year', 'unknown')}",
        f"Venue: {paper.get('venue', '')}",
        f"Abstract: {paper.get('abstract', '')}",
        f"Keywords: {', '.join(paper.get('keywords', []))}",
        f"Summary: {paper.get('summary', '')}",
    ]
    return "\n\n".join(p for p in parts if p.split(": ", 1)[1].strip())


def _upsert_chunks(collection, embedder, chunks: list[dict]):
    """Embed and upsert a batch of chunks into ChromaDB."""
    if not chunks:
        return
    texts     = [c["text"] for c in chunks]
    ids       = [c["id"]   for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def run_ingestion(reset: bool = False):
    print("=== Research RAG — Ingestion ===")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedder   = get_embedder()
    collection = get_chroma_collection(reset=reset)

    existing = collection.count()
    print(f"Existing vectors in store: {existing}")

    total = 0
    total += ingest_pdfs(collection, embedder)
    total += ingest_metadata_papers(collection, embedder)

    print(f"\nIngestion complete. Chunks added/updated: {total}")
    print(f"Total vectors in store: {collection.count()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the vector store")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)
