"""
rag.py — Retrieve relevant paper chunks and generate answers using an LLM.

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

Run:
  python src/rag.py --query "What work have I done on user modeling?"
  python src/rag.py --interactive
  python src/rag.py --interactive --backend gemini
  python src/rag.py --interactive --backend groq --model llama-3.3-70b-versatile
  python src/rag.py --list-backends
"""

import os
import tomllib
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
CHROMA_DIR      = DATA_DIR / "chroma_db"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"

TOP_K           = 5
MAX_CONTEXT_LEN = 4000

# ── Backend registry ──────────────────────────────────────────────────────────
# auto-detection tries backends in this order
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


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, embedder: SentenceTransformer, collection, top_k: int = TOP_K) -> list[dict]:
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
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk["metadata"]
        source = f"{meta.get('title', 'Unknown')} ({meta.get('year', '?')}) — {meta.get('venue', '')}"
        parts.append(
            f"[Source {i}] {source}\n"
            f"Similarity: {chunk['similarity']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_LEN]


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant helping Dr. Hua Li explore and discuss his own published research.

You have access to excerpts from his papers. Answer questions accurately based only on the provided context.
If the context doesn't fully answer the question, say so clearly — don't hallucinate details.

When discussing papers, cite them by title and year. Highlight connections between papers where relevant.
Keep answers focused and grounded in the retrieved text."""


def build_prompt(query: str, context: str) -> str:
    return (
        f"Here are relevant excerpts from Dr. Hua Li's research papers:\n\n"
        f"{context}\n\n---\n\n"
        f"Question: {query}\n\n"
        f"Please answer based on the context above. Cite specific papers by title and year."
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
    "gemini":      answer_with_gemini,
    "groq":        answer_with_groq,
    "mistral":     answer_with_mistral,
    "openrouter":  answer_with_openrouter,
    "cohere":      answer_with_cohere,
    "ollama":      answer_with_ollama,
    "anthropic":   answer_with_anthropic,
    "openai":      answer_with_openai,
    "local":       answer_local,
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
            f"Backend '{requested}' requires {cfg['env_key']} to be set.\n"
            f"  {cfg['notes']}"
        )
    return requested, cfg["default_model"]


def generate_answer(query: str, context: str, backend: str, model: str) -> str:
    return _BACKEND_FN[backend](query, context, model)


# ── Main interface ─────────────────────────────────────────────────────────────

def query_once(question: str, backend: str, model: str, show_sources: bool = True) -> str:
    embedder   = SentenceTransformer(EMBED_MODEL, device="cpu")
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

        chunks  = retrieve(question, embedder, collection)
        context = build_context(chunks)
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
    # Load config.toml defaults (CLI flags override these)
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
    parser.add_argument("--interactive",   action="store_true", help="Start interactive Q&A session")
    parser.add_argument("--no-sources",    action="store_true", help="Hide source citations")
    parser.add_argument("--backend",       type=str,  default=None,
                        help="LLM backend to use. Use --list-backends to see options.")
    parser.add_argument("--model",         type=str,  default=None,
                        help="Override the default model for the selected backend.")
    parser.add_argument("--list-backends", action="store_true", help="Show all backends and their status")
    args = parser.parse_args()

    # CLI > config.toml > hardcoded default
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
