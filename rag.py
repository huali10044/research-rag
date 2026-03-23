"""
rag.py — Retrieve relevant paper chunks and generate answers using an LLM.

Two modes:
  1. API mode  — uses OpenAI or Anthropic API (best quality, requires key)
  2. Local mode — uses a small local model via transformers (no API key needed)

Run:
  python src/rag.py --query "What work have I done on user modeling?"
  python src/rag.py --query "Summarize my DARPA-related publications"
  python src/rag.py --interactive
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
CHROMA_DIR      = DATA_DIR / "chroma_db"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"

TOP_K           = 5     # number of chunks to retrieve
MAX_CONTEXT_LEN = 4000  # chars — trim if hitting token limits


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, embedder: SentenceTransformer, collection, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-k most similar chunks with metadata."""
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":       doc,
            "metadata":   meta,
            "similarity": round(1 - dist, 4),   # cosine distance → similarity
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Assemble retrieved chunks into a formatted context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = f"{meta.get('title', 'Unknown')} ({meta.get('year', '?')}) — {meta.get('venue', '')}"
        parts.append(
            f"[Source {i}] {source}\n"
            f"Similarity: {chunk['similarity']}\n\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(parts)
    return context[:MAX_CONTEXT_LEN]


# ── LLM Generation ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant helping Dr. Hua Li explore and discuss his own published research.

You have access to excerpts from his papers. Answer questions accurately based only on the provided context.
If the context doesn't fully answer the question, say so clearly — don't hallucinate details.

When discussing papers, cite them by title and year. Highlight connections between papers where relevant.
Keep answers focused and grounded in the retrieved text."""


def build_prompt(query: str, context: str) -> str:
    return f"""Here are relevant excerpts from Dr. Hua Li's research papers:

{context}

---

Question: {query}

Please answer based on the context above. Cite specific papers by title and year."""


def answer_with_anthropic(query: str, context: str) -> str:
    """Generate answer using Claude via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, context)}],
    )
    return message.content[0].text


def answer_with_openai(query: str, context: str) -> str:
    """Generate answer using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_local(query: str, context: str) -> str:
    """Fallback: generate answer with a small local model (no API key needed)."""
    try:
        from transformers import pipeline
        print("[Using local model — slower, lower quality than API]")
        generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        prompt = f"{SYSTEM_PROMPT}\n\n{build_prompt(query, context)}\n\nAnswer:"
        result = generator(prompt, max_new_tokens=300, do_sample=False)
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        return f"[Local model unavailable: {e}]\n\nRelevant context retrieved:\n\n{context}"


def generate_answer(query: str, context: str) -> str:
    """Pick the best available LLM backend."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return answer_with_anthropic(query, context)
    elif os.environ.get("OPENAI_API_KEY"):
        return answer_with_openai(query, context)
    else:
        print("No API key found — falling back to local model.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY for best results.\n")
        return answer_local(query, context)


# ── Main interface ─────────────────────────────────────────────────────────────

def query(question: str, show_sources: bool = True) -> str:
    embedder   = SentenceTransformer(EMBED_MODEL)
    collection = get_collection()

    print(f"\nQuery: {question}")
    print("Retrieving relevant chunks...")
    chunks = retrieve(question, embedder, collection)

    if show_sources:
        print(f"\nTop {len(chunks)} sources retrieved:")
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            print(f"  {i}. [{c['similarity']:.3f}] {meta.get('title', '?')[:60]} ({meta.get('year', '?')})")

    context = build_context(chunks)
    print("\nGenerating answer...\n")
    answer = generate_answer(question, context)
    return answer


def interactive_mode():
    print("=== Research RAG — Interactive Mode ===")
    print("Ask questions about your research. Type 'quit' to exit.\n")
    embedder   = SentenceTransformer(EMBED_MODEL)
    collection = get_collection()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        chunks  = retrieve(question, embedder, collection)
        context = build_context(chunks)
        answer  = generate_answer(question, context)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your research paper RAG system")
    parser.add_argument("--query",       type=str, help="Single question to answer")
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A session")
    parser.add_argument("--no-sources",  action="store_true", help="Hide source citations")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        answer = query(args.query, show_sources=not args.no_sources)
        print(f"\nAnswer:\n{answer}")
    else:
        parser.print_help()
