"""
rag.py — Retrieve relevant paper chunks and generate answers using an LLM.

Query routing:
  stats    → answer directly from corpus_stats.json (no vector search)
  listing  → inject full paper list as context (no vector search)
  metadata → vector search + paper list header
  content  → vector search with per-paper deduplication

Supported backends (--backend flag):
  Free tier:
    gemini      Google Gemini 2.0 Flash     — aistudio.google.com     GEMINI_API_KEY
    groq        Llama 3 via Groq            — console.groq.com        GROQ_API_KEY
    mistral     Mistral Small               — console.mistral.ai      MISTRAL_API_KEY
    openrouter  Mistral-7B free model       — openrouter.ai           OPENROUTER_API_KEY
    cohere      Command-R                   — dashboard.cohere.com    COHERE_API_KEY
    ollama      Local Llama (no key)        — ollama.com              (none)
  Paid:
    anthropic   Claude Haiku                — console.anthropic.com   ANTHROPIC_API_KEY
    openai      GPT-4o-mini                 — platform.openai.com     OPENAI_API_KEY
  Fallback:
    local       TinyLlama via transformers  — (no key, slow)
"""

import logging
import os
import re
import json
import time
import tomllib
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
CHROMA_DIR      = DATA_DIR / "chroma_db"
STATS_FILE      = DATA_DIR / "corpus_stats.json"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"

TOP_K           = 5
MAX_CONTEXT_LEN = 12000   # characters; increased from 4000 to fit corpus overview + chunks

# ── Backend registry ──────────────────────────────────────────────────────────
AUTO_ORDER = ["gemini", "groq", "mistral", "openrouter", "cohere", "anthropic", "openai", "ollama", "local"]

BACKENDS = {
    "gemini": {
        "env_key":       "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "free_tier":     True,
        "notes":         "Free tier (15 RPM, 1M tokens/day). Get key at aistudio.google.com",
    },
    "groq": {
        "env_key":       "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
        "free_tier":     True,
        "notes":         "Free tier (rate-limited). Get key at console.groq.com",
    },
    "mistral": {
        "env_key":       "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
        "free_tier":     True,
        "notes":         "Free tier for small models. Get key at console.mistral.ai",
    },
    "openrouter": {
        "env_key":       "OPENROUTER_API_KEY",
        "default_model": "mistralai/mistral-7b-instruct:free",
        "free_tier":     True,
        "notes":         "Some models are free. Get key at openrouter.ai",
    },
    "cohere": {
        "env_key":       "COHERE_API_KEY",
        "default_model": "command-r",
        "free_tier":     True,
        "notes":         "Free trial key. Get key at dashboard.cohere.com",
    },
    "ollama": {
        "env_key":       None,
        "default_model": "llama3.2",
        "free_tier":     True,
        "notes":         "Local inference, completely free. Install from ollama.com, then: ollama pull llama3.2",
    },
    "anthropic": {
        "env_key":       "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
        "free_tier":     False,
        "notes":         "Paid. Get key at console.anthropic.com (new accounts get $5 free credit)",
    },
    "openai": {
        "env_key":       "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "free_tier":     False,
        "notes":         "Paid. Get key at platform.openai.com",
    },
    "local": {
        "env_key":       None,
        "default_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "free_tier":     True,
        "notes":         "Local transformers model (slow, low quality). No key needed.",
    },
}


# ── ChromaDB access ───────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


# ── Corpus stats ──────────────────────────────────────────────────────────────

def load_corpus_stats() -> dict | None:
    """Load pre-computed corpus stats from data/corpus_stats.json, or None if missing."""
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            pass
    return None


def build_corpus_context(stats: dict, verbose: bool = True) -> str:
    """Format corpus stats as a plain-text block for LLM context injection."""
    papers = stats.get("papers", [])
    yr     = stats.get("year_range", {})

    lines = ["CORPUS OVERVIEW"]
    lines.append(f"Total papers: {stats['total_papers']}")
    if yr.get("min") and yr.get("max"):
        lines.append(f"Publication years: {yr['min']} – {yr['max']}")
    if stats.get("total_words"):
        lines.append(f"Total words across all papers: {stats['total_words']:,}")
    lines.append("")
    lines.append("PAPERS (sorted by year):")

    for p in sorted(papers, key=lambda x: (x.get("year") or 0)):
        year  = p.get("year", "?")
        title = p.get("title", "Unknown")
        venue = p.get("venue", "")
        wc    = p.get("word_count", 0)

        authors = p.get("authors", "")
        if isinstance(authors, list):
            auth_str = ", ".join(authors)
        else:
            auth_str = authors or ""

        line = f"- [{year}] {title}"
        if venue:
            line += f" | {venue}"
        if verbose and wc:
            line += f" | {wc:,} words"
        lines.append(line)
        if auth_str:
            lines.append(f"  Authors: {auth_str}")

    return "\n".join(lines)


# ── Query classification ──────────────────────────────────────────────────────

_STATS_PATTERNS = [
    r"\bhow many\b",
    r"\btotal (number|count|papers|publications|works)\b",
    r"\bnumber of (papers|publications|works)\b",
    r"\bword count\b",
    r"\bhow (long|many words)\b",
    r"\b(last|first|oldest|newest|latest|most recent|earliest) paper\b",
    r"\bwhen (was|did).{0,40}publish",
    r"\bwhat year\b",
    r"\bpublication (date|year)\b",
    r"\bmost prolific\b",
]

_LISTING_PATTERNS = [
    r"\blist (all|your|my|the|his)\b",
    r"\ball (papers|publications|works|research)\b",
    r"\b(give|show) me (all|a list|the list)\b",
    r"\bwhat (papers|publications|works) (did|has|have)\b",
    r"\bwhich papers\b",
    r"\bfull list\b",
    r"\bcomplete list\b",
    r"\blist of (papers|publications)\b",
    r"\bpublication list\b",
]

_METADATA_PATTERNS = [
    r"\bwho (wrote|authored|are the authors of)\b",
    r"\b(author|authors) of\b",
    r"\bpublished (in|at|by)\b",
    r"\b(venue|conference|journal) (for|of)\b",
    r"\bwhere was\b",
    r"\bdoi\b",
]

# Conversational commands that should never hit the RAG pipeline
_COMMAND_PATTERNS = [
    r"^rerun\b",
    r"\brerun (last|that|the last|previous|it)\b",
    r"\brun (that|it|last|the last) again\b",
    r"^run again\b",
    r"\brepeat (that|last|the last|previous|it|the question)\b",
    r"^repeat$",
    r"\btry again\b",
    r"^redo\b",
    r"\bsame question\b",
    r"\bsame query\b",
]


def classify_query(query: str) -> str:
    """Classify query as 'command', 'stats', 'listing', 'metadata', or 'content'."""
    q = query.lower().strip()
    for pat in _COMMAND_PATTERNS:
        if re.search(pat, q):
            return "command"
    for pat in _STATS_PATTERNS:
        if re.search(pat, q):
            return "stats"
    for pat in _LISTING_PATTERNS:
        if re.search(pat, q):
            return "listing"
    for pat in _METADATA_PATTERNS:
        if re.search(pat, q):
            return "metadata"
    return "content"


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _extract_proper_noun_phrases(text: str) -> list[str]:
    """Return sequences of 2+ consecutive title-case words (likely person names)."""
    tokens = text.split()
    phrases, i = [], 0
    while i < len(tokens):
        word = tokens[i].rstrip("?,.")
        if i > 0 and word and word[0].isupper() and word.replace("-", "").isalpha():
            j = i
            while j < len(tokens):
                w = tokens[j].rstrip("?,.")
                if w and w[0].isupper() and w.replace("-", "").isalpha():
                    j += 1
                else:
                    break
            if j - i >= 2:
                phrases.append(" ".join(t.rstrip('?.,') for t in tokens[i:j]))
            i = j
        else:
            i += 1
    return phrases


def retrieve(query: str, embedder: SentenceTransformer, collection, top_k: int = TOP_K) -> list[dict]:
    """Raw vector search — returns top_k chunks sorted by similarity."""
    query_embedding = embedder.encode([query], device="cpu")[0].tolist()
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
            "similarity": round(1 - dist, 4),
        })
    logger.debug("chunks=" + str(chunks))
    return chunks


MIN_CHUNK_LEN = 120  # discard page-header/footer fragments shorter than this


def _retrieve_deduped(
    query: str, embedder: SentenceTransformer, collection, top_k: int
) -> list[dict]:
    """
    Retrieve top_k diverse chunks with per-paper caps.

    Pipeline:
      1. Fetch a large candidate pool.
      2. Drop exact-text duplicates.
      3. Drop chunks shorter than MIN_CHUNK_LEN (page headers / reference noise).
      4. Apply per-paper cap so no single paper monopolizes results.
    """
    SIMILARITY_GAP = 0.10   # margin to be treated as co-dominant
    DOMINANT_CAP   = 2      # max chunks from any single paper (even top-ranked)
    FALLBACK_CAP   = 1      # max chunks for papers outside the dominant band

    raw = retrieve(query, embedder, collection, top_k=min(top_k * 8, 80))
    if not raw:
        return []

    # 1. Remove exact-text duplicates
    seen_text: set[str] = set()
    deduped = []
    for chunk in raw:
        if chunk["text"] not in seen_text:
            seen_text.add(chunk["text"])
            deduped.append(chunk)

    # 2. Drop short noise chunks (page headers, reference list fragments, etc.)
    meaningful = [c for c in deduped if len(c["text"].strip()) >= MIN_CHUNK_LEN]
    if not meaningful:
        meaningful = deduped  # fall back if everything is short

    logger.debug(
        "deduped=%d meaningful=%d (dropped %d short)",
        len(deduped), len(meaningful), len(deduped) - len(meaningful),
    )

    # 3. Group by paper; track each paper's best (first) similarity score
    papers: dict[str, float] = {}
    for chunk in meaningful:
        title = chunk["metadata"].get("title", "")
        if title not in papers:
            papers[title] = chunk["similarity"]

    top_score = meaningful[0]["similarity"]
    caps = {
        title: DOMINANT_CAP if (top_score - best) <= SIMILARITY_GAP else FALLBACK_CAP
        for title, best in papers.items()
    }

    # 4. Apply caps
    seen_count: dict[str, int] = {}
    result = []
    for chunk in meaningful:
        title = chunk["metadata"].get("title", "")
        count = seen_count.get(title, 0)
        if count < caps[title]:
            result.append(chunk)
            seen_count[title] = count + 1
        if len(result) >= top_k:
            break

    logger.debug("final chunks=%d", len(result))

    # 5. Keyword supplement: for proper noun phrases in the query (e.g. person names),
    #    do a keyword-filtered vector search and append any chunks not already in the
    #    result. This handles queries like "did Jeff Lau coauthor?" where the vector
    #    similarity between the question and an author-list chunk is too low to survive
    #    the cap logic above.
    phrases = _extract_proper_noun_phrases(query)
    if phrases:
        query_embedding = embedder.encode([query], device="cpu")[0].tolist()
        existing = {(c["metadata"].get("source"), c["metadata"].get("chunk_index"))
                    for c in result}
        for phrase in phrases:
            try:
                kw = collection.query(
                    query_embeddings=[query_embedding],
                    where_document={"$contains": phrase},
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(kw["documents"][0], kw["metadatas"][0], kw["distances"][0]):
                    key = (meta.get("source"), meta.get("chunk_index"))
                    if key not in existing and len(doc.strip()) >= MIN_CHUNK_LEN:
                        existing.add(key)
                        result.append({"text": doc, "metadata": meta,
                                       "similarity": round(1 - dist, 4)})
            except Exception:
                pass

    return result


def adaptive_retrieve(
    query: str, embedder: SentenceTransformer, collection, top_k: int = TOP_K
) -> tuple[list[dict], str, str]:
    """
    Route the query based on its type. Returns (chunks, query_type, extra_context).

    - command         → no vector search; app.py resolves by re-running prior query
    - stats / listing → no vector search; answer from corpus_stats.json
    - metadata        → vector search + paper list header for grounding
    - content         → vector search with per-paper deduplication
    """
    query_type = classify_query(query)
    stats      = load_corpus_stats()

    if query_type == "command":
        return [], query_type, ""

    if query_type in ("stats", "listing"):
        extra = build_corpus_context(stats, verbose=True) if stats else ""
        return [], query_type, extra

    if query_type == "metadata":
        chunks = _retrieve_deduped(query, embedder, collection, top_k)
        extra  = build_corpus_context(stats, verbose=False) if stats else ""
        return chunks, query_type, extra

    # content — also inject corpus metadata so the LLM can fall back to it
    # (e.g. bibliography chunks retrieved by vector search + author list = enough to answer coauthor questions)
    chunks = _retrieve_deduped(query, embedder, collection, top_k)
    extra  = build_corpus_context(stats, verbose=False) if stats else ""
    return chunks, query_type, extra


# ── Context and prompt assembly ───────────────────────────────────────────────

def build_context(chunks: list[dict], extra_context: str = "") -> str:
    """Assemble LLM context from retrieved chunks plus optional structured header."""
    parts = []
    if extra_context:
        parts.append(extra_context)
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk["metadata"]
        source = (
            f"{meta.get('title', 'Unknown')} "
            f"({meta.get('year', '?')}) — {meta.get('venue', '')}"
        )
        parts.append(
            f"[Source {i}] {source}\n"
            f"Similarity: {chunk['similarity']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_LEN]


SYSTEM_PROMPT = """You are a research assistant helping Dr. Hua Li explore and discuss his own published research.

You have access to a structured overview of all his papers and/or excerpts from the papers themselves.
Answer questions accurately based only on the provided context.
If the context doesn't fully answer the question, say so clearly — don't hallucinate details.

When discussing papers, cite them by title and year. Highlight connections between papers where relevant.
Keep answers focused and grounded in the retrieved context."""


def build_prompt(query: str, context: str) -> str:
    return (
        "Here is context from Dr. Hua Li's research papers:\n\n"
        f"{context}\n\n---\n\n"
        f"Question: {query}\n\n"
        "Please answer based on the context above. Cite specific papers by title and year."
    )


# ── Backend implementations ───────────────────────────────────────────────────

def answer_with_gemini(query: str, context: str, model: str) -> str:
    from google import genai
    client   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\n{build_prompt(query, context)}",
    )
    return response.text


def answer_with_groq(query: str, context: str, model: str) -> str:
    from openai import OpenAI
    client   = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_with_mistral(query: str, context: str, model: str) -> str:
    from openai import OpenAI
    client   = OpenAI(api_key=os.environ["MISTRAL_API_KEY"], base_url="https://api.mistral.ai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_with_openrouter(query: str, context: str, model: str) -> str:
    from openai import OpenAI
    client   = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_with_cohere(query: str, context: str, model: str) -> str:
    import cohere
    client   = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
    )
    return response.message.content[0].text


def answer_with_ollama(query: str, context: str, model: str) -> str:
    from openai import OpenAI
    client   = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_with_anthropic(query: str, context: str, model: str) -> str:
    import anthropic
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(query, context)}],
    )
    return message.content[0].text


def answer_with_openai(query: str, context: str, model: str) -> str:
    from openai import OpenAI
    client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, context)},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_local(query: str, context: str, model: str) -> str:
    try:
        from transformers import pipeline
        print("[Using local model — slower, lower quality than API]")
        generator = pipeline("text-generation", model=model)
        prompt    = f"{SYSTEM_PROMPT}\n\n{build_prompt(query, context)}\n\nAnswer:"
        result    = generator(prompt, max_new_tokens=300, do_sample=False)
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        return f"[Local model unavailable: {e}]\n\nRelevant context retrieved:\n\n{context}"


_BACKEND_FN = {
    "gemini":     answer_with_gemini,
    "groq":       answer_with_groq,
    "mistral":    answer_with_mistral,
    "openrouter": answer_with_openrouter,
    "cohere":     answer_with_cohere,
    "ollama":     answer_with_ollama,
    "anthropic":  answer_with_anthropic,
    "openai":     answer_with_openai,
    "local":      answer_local,
}


# ── Backend selection ─────────────────────────────────────────────────────────

def resolve_backend(requested: str) -> tuple[str, str]:
    """Return (backend_name, model). Raises if backend unavailable."""
    if requested == "auto":
        for name in AUTO_ORDER:
            cfg = BACKENDS[name]
            if cfg["env_key"] is None or os.environ.get(cfg["env_key"]):
                return name, cfg["default_model"]
        return "local", BACKENDS["local"]["default_model"]

    if requested not in BACKENDS:
        raise ValueError(f"Unknown backend '{requested}'. Use --list-backends to see options.")

    cfg = BACKENDS[requested]
    if cfg["env_key"] and not os.environ.get(cfg["env_key"]):
        raise EnvironmentError(
            f"Backend '{requested}' requires {cfg['env_key']} to be set.\n  {cfg['notes']}"
        )
    return requested, cfg["default_model"]


def _is_retriable(exc: Exception) -> bool:
    msg = str(exc)
    return any(tok in msg for tok in ("503", "UNAVAILABLE", "429", "overload", "high demand", "rate limit"))


def _available_backends() -> list[str]:
    return [
        name for name in AUTO_ORDER
        if BACKENDS[name]["env_key"] is None or os.environ.get(BACKENDS[name]["env_key"])
    ]


def generate_answer(query: str, context: str, backend: str, model: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return _BACKEND_FN[backend](query, context, model)
        except Exception as e:
            if _is_retriable(e) and attempt < 2:
                time.sleep(2 ** attempt)
                last_exc = e
                continue
            if not _is_retriable(e):
                raise
            last_exc = e
            break

    # Retries exhausted — try other available backends in priority order
    tried = {backend}
    for fallback in _available_backends():
        if fallback in tried or fallback == "local":
            continue
        tried.add(fallback)
        fallback_model = BACKENDS[fallback]["default_model"]
        try:
            result = _BACKEND_FN[fallback](query, context, fallback_model)
            return f"*({backend} unavailable, answered via {fallback})*\n\n{result}"
        except Exception:
            continue

    if last_exc:
        raise last_exc
    return answer_local(query, context, BACKENDS["local"]["default_model"])


# ── High-level interface ──────────────────────────────────────────────────────

def adaptive_answer(
    query: str,
    embedder: SentenceTransformer,
    collection,
    top_k: int,
    backend: str,
    model: str,
) -> tuple[str, list[dict], str]:
    """
    End-to-end adaptive query answering.

    Returns:
        answer     — LLM-generated response string
        chunks     — retrieved source chunks (empty for stats/listing queries)
        query_type — one of 'stats', 'listing', 'metadata', 'content'
    """
    chunks, query_type, extra_context = adaptive_retrieve(query, embedder, collection, top_k)
    context = build_context(chunks, extra_context)
    answer  = generate_answer(query, context, backend, model)
    return answer, chunks, query_type


def query_once(question: str, backend: str, model: str, show_sources: bool = True) -> str:
    embedder   = SentenceTransformer(EMBED_MODEL, device="cpu")
    collection = get_collection()

    print(f"\nQuery: {question}")
    chunks, query_type, extra_context = adaptive_retrieve(question, embedder, collection)
    print(f"Query type: {query_type} | Retrieved chunks: {len(chunks)}")

    if show_sources and chunks:
        print(f"\nTop {len(chunks)} sources:")
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            print(f"  {i}. [{c['similarity']:.3f}] {meta.get('title', '?')[:60]} ({meta.get('year', '?')})")

    context = build_context(chunks, extra_context)
    print(f"\nGenerating answer (backend: {backend}, model: {model})...\n")
    return generate_answer(question, context, backend, model)


def interactive_mode(backend: str, model: str):
    print("=== Research RAG — Interactive Mode ===")
    print(f"Backend: {backend}  |  Model: {model}")
    print("Ask questions about your research. Type 'quit' to exit.\n")

    embedder   = SentenceTransformer(EMBED_MODEL, device="cpu")
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

        chunks, query_type, extra_context = adaptive_retrieve(question, embedder, collection)
        print(f"[{query_type} query, {len(chunks)} chunks]")
        context = build_context(chunks, extra_context)
        answer  = generate_answer(question, context, backend, model)
        print(f"\nAssistant: {answer}\n")


def list_backends():
    print("\nAvailable backends:\n")
    print(f"  {'NAME':<12} {'FREE':<6} {'DEFAULT MODEL':<40} NOTES")
    print("  " + "-" * 100)
    for name, cfg in BACKENDS.items():
        key_set = "✓" if (cfg["env_key"] is None or os.environ.get(cfg["env_key"])) else " "
        free    = "✓ free" if cfg["free_tier"] else "  paid"
        avail   = f"[key set {key_set}]"
        print(f"  {name:<12} {free:<6}  {cfg['default_model']:<40} {avail}  {cfg['notes']}")
    print()


if __name__ == "__main__":
    _config_path = Path(__file__).parent.parent / "config.toml"
    _cfg_backend = "auto"
    _cfg_model   = None
    if _config_path.exists():
        with open(_config_path, "rb") as _f:
            _cfg = tomllib.load(_f)
        _cfg_backend = _cfg.get("llm", {}).get("backend", "auto")
        _cfg_model   = _cfg.get("llm", {}).get("model") or None

    parser = argparse.ArgumentParser(description="Query your research paper RAG system")
    parser.add_argument("--query",         type=str,  help="Single question to answer")
    parser.add_argument("--interactive",   action="store_true")
    parser.add_argument("--no-sources",    action="store_true")
    parser.add_argument("--backend",       type=str,  default=None)
    parser.add_argument("--model",         type=str,  default=None)
    parser.add_argument("--list-backends", action="store_true")
    args = parser.parse_args()

    backend_arg = args.backend or _cfg_backend
    model_arg   = args.model   or _cfg_model

    if args.list_backends:
        list_backends()
    elif args.interactive or args.query:
        try:
            backend, model = resolve_backend(backend_arg)
        except (ValueError, EnvironmentError) as e:
            print(f"Error: {e}")
            raise SystemExit(1)

        if model_arg:
            model = model_arg

        if args.interactive:
            interactive_mode(backend, model)
        else:
            answer = query_once(args.query, backend, model, show_sources=not args.no_sources)
            print(f"\nAnswer:\n{answer}")
    else:
        parser.print_help()
