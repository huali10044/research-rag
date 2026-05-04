"""
ingest.py — Load, chunk, embed, and store research papers into a vector store.

Metadata extraction pipeline (tiered, results merged field-by-field):
  1. Cache          — data/pdf_metadata_cache.json (instant, avoids repeat API calls)
  2. spaCy NLP      — NER for persons/orgs/dates + rule-based patterns (offline, free)
  3. LLM extraction — structured JSON prompt via Gemini / Anthropic / Groq
  4. Regex/filename — year from YYYY- filename prefix or copyright patterns (always available)

Corpus statistics are saved to data/corpus_stats.json after ingestion for fast query answering.
"""

import os
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
PAPERS_DIR      = DATA_DIR / "papers"
CHROMA_DIR      = DATA_DIR / "chroma_db"
STATS_FILE      = DATA_DIR / "corpus_stats.json"
META_CACHE_FILE = DATA_DIR / "pdf_metadata_cache.json"
COLLECTION_NAME = "research_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"

CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150
MIN_CHUNK_LEN  = 150  # discard tiny fragments (e.g. bibliography entries)


def get_embedder() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Reset collection: {COLLECTION_NAME}")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Tier 4: Regex / filename fallbacks ───────────────────────────────────────

def _year_from_filename(path: Path) -> int | None:
    m = re.match(r"^(20\d{2}|199\d)[^0-9]", path.stem)
    return int(m.group(1)) if m else None


def _year_from_text(text: str) -> int | None:
    """Extract publication year from the first ~3 000 chars of paper text."""
    patterns = [
        r"(?:©|\bCopyright\b)\s*(20\d{2}|199\d)",
        r"(?:Proceedings|Proc\.)\s+\w[^\n]{0,60}?(20\d{2}|199\d)",
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(20\d{2}|199\d)",
        r"\b(20\d{2}|199\d)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text[:3000], re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# ── Tier 2: spaCy NLP extraction ──────────────────────────────────────────────

def _extract_with_spacy(first_pages_text: str) -> dict:
    """Use spaCy NER + rule-based patterns. Returns partial {title, authors, year, venue}."""
    result: dict = {"title": None, "authors": [], "year": None, "venue": None}
    try:
        import spacy
    except ImportError:
        return result

    # Load the smallest available model
    nlp = None
    for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            nlp = spacy.load(model_name)
            break
        except OSError:
            continue
    if nlp is None:
        return result

    # Work on the first 2 000 chars where the header/authors/venue usually appear
    header = first_pages_text[:2000]
    doc    = nlp(header)

    # Authors: PERSON entities in the top 800 chars (below the title)
    persons = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "PERSON" and ent.start_char < 800
    ]
    if persons:
        result["authors"] = list(dict.fromkeys(persons))  # deduplicate, preserve order

    # Year: DATE entities that look like a 4-digit year
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            m = re.search(r"\b(20\d{2}|199\d)\b", ent.text)
            if m:
                result["year"] = int(m.group(1))
                break

    # Venue: look for "Proceedings of ...", "In: ...", "Journal of ..."
    venue_patterns = [
        r"(?:Proceedings|Proc\.)\s+(?:of\s+)?(?:the\s+)?([^\n]{5,80})",
        r"(?:In:|In\s+Proceedings[^:]*:)\s*([^\n]{5,80})",
        r"(?:Journal of|IEEE|ACM)\s+[A-Z][^\n]{5,60}",
    ]
    for pat in venue_patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            venue = (m.group(1) if m.lastindex else m.group(0)).strip().rstrip(".,;")
            if len(venue) > 5:
                result["venue"] = venue
                break

    # ORG entities near known venue keywords as secondary fallback
    if not result["venue"]:
        venue_keywords = {"IEEE", "ACM", "AAAI", "IJCAI", "NIPS", "NeurIPS", "ICML",
                          "ICDM", "KDD", "CIKM", "SIGIR", "WWW", "ICCBR", "MILCOM", "ADMA"}
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(kw in ent.text for kw in venue_keywords):
                result["venue"] = ent.text.strip()
                break

    return result


# ── Tier 3: LLM extraction ────────────────────────────────────────────────────

def _call_extraction_llm(prompt: str) -> str | None:
    """Try available LLM APIs for a one-shot structured extraction. Returns raw text or None."""
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            client   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"    [LLM] Gemini failed: {e}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg    = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"    [LLM] Anthropic failed: {e}")

    if os.environ.get("GROQ_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    [LLM] Groq failed: {e}")

    return None


def _parse_llm_json(text: str) -> dict | None:
    """Extract and parse a JSON object from an LLM response (handles markdown fences)."""
    text = text.strip()
    if "```" in text:
        for block in text.split("```")[1::2]:   # text inside fences
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except Exception:
                continue
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_with_llm(first_pages_text: str) -> dict:
    """Use an LLM to extract structured metadata. Returns partial {title, authors, year, venue}."""
    prompt = (
        "Extract metadata from this academic paper. "
        "Return ONLY a JSON object — no other text:\n"
        '{\n  "title": "full paper title",\n'
        '  "authors": ["First Last", "First Last"],\n'
        '  "year": 2014,\n'
        '  "venue": "conference or journal name"\n}\n'
        "Use null for any field you cannot find.\n\n"
        f"Paper text (first pages):\n{first_pages_text[:3000]}"
    )
    raw = _call_extraction_llm(prompt)
    if not raw:
        return {}
    parsed = _parse_llm_json(raw)
    if not parsed:
        print("    [LLM] Could not parse JSON from response")
        return {}
    return parsed


# ── Unified extraction pipeline ───────────────────────────────────────────────

def load_meta_cache() -> dict:
    if META_CACHE_FILE.exists():
        try:
            return json.loads(META_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_meta_cache(cache: dict):
    META_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def extract_pdf_metadata(
    pdf_path: Path,
    first_pages_text: str,
    cache: dict,
    use_spacy: bool = True,
    use_llm: bool = True,
) -> dict:
    """
    Return {title, authors, year, venue} for a PDF.

    Extraction order (results merged field-by-field, higher tiers win):
      regex/filename → spaCy NLP → LLM → cached result on next run
    """
    cache_key = pdf_path.name
    if cache_key in cache:
        return cache[cache_key]

    # Tier 4: cheap always-available fallbacks
    meta: dict = {
        "title":   None,
        "authors": [],
        "year":    _year_from_filename(pdf_path) or _year_from_text(first_pages_text),
        "venue":   None,
    }

    # Tier 2: spaCy NLP
    if use_spacy:
        spacy_result = _extract_with_spacy(first_pages_text)
        for field, val in spacy_result.items():
            if val and not meta.get(field):
                meta[field] = val

    # Tier 3: LLM
    if use_llm:
        llm_result = _extract_with_llm(first_pages_text)
        for field in ("title", "authors", "year", "venue"):
            val = llm_result.get(field)
            if val:
                meta[field] = val  # LLM always wins when it returns something

    title_preview = str(meta["title"])[:55] if meta["title"] else "(no title)"
    print(f"    → {title_preview} | year={meta['year']} | venue={str(meta['venue'])[:40]}")

    cache[cache_key] = meta
    return meta


# ── Corpus statistics ─────────────────────────────────────────────────────────

def compute_and_save_corpus_stats(papers: list[dict]):
    """Deduplicate papers, compute aggregate stats, and write corpus_stats.json."""
    seen: dict[str, dict] = {}
    for p in papers:
        key = p.get("title", "").lower().strip()
        if not key:
            continue
        # Prefer manually-curated metadata-source entries over raw PDF extractions
        if key not in seen or p.get("source") == "metadata":
            seen[key] = p

    unique = sorted(seen.values(), key=lambda x: (x.get("year") or 0, x.get("title", "")))
    years  = [p["year"] for p in unique if isinstance(p.get("year"), int)]

    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_papers": len(unique),
        "year_range": {
            "min": min(years) if years else None,
            "max": max(years) if years else None,
        },
        "total_words": sum(p.get("word_count", 0) for p in unique),
        "papers": unique,
    }

    STATS_FILE.write_text(json.dumps(stats, indent=2))
    yr = stats["year_range"]
    print(
        f"Corpus stats → {STATS_FILE.name}: "
        f"{stats['total_papers']} unique papers, "
        f"years {yr['min']}–{yr['max']}, "
        f"{stats['total_words']:,} total words"
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_meta: dict) -> list[dict]:
    """Split text into overlapping chunks at natural boundaries."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = chunk.rfind(sep)
                if idx > CHUNK_SIZE // 2:
                    chunk = chunk[: idx + len(sep)]
                    break
        chunks.append(chunk.strip())
        start += max(len(chunk) - CHUNK_OVERLAP, 1)

    results = []
    for i, chunk in enumerate(chunks):
        if not chunk or len(chunk) < MIN_CHUNK_LEN:
            continue
        doc_id = hashlib.md5(f"{source_meta['title']}_chunk_{i}".encode()).hexdigest()
        results.append({
            "id":       doc_id,
            "text":     chunk,
            "metadata": {**source_meta, "chunk_index": i, "total_chunks": len(chunks)},
        })
    return results


# ── Ingestion functions ───────────────────────────────────────────────────────

def ingest_pdfs(
    collection,
    embedder,
    use_spacy: bool = True,
    use_llm: bool = True,
) -> tuple[int, list[dict]]:
    """
    Ingest PDFs with tiered metadata extraction. Returns (chunk_count, papers).

    Args:
        use_spacy: Enable spaCy NLP extraction (requires spacy + model installed).
        use_llm:   Enable LLM-based extraction (requires at least one API key).
    """
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {PAPERS_DIR}")
        return 0, []

    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed — skipping PDF ingestion. Run: pip install pypdf")
        return 0, []

    meta_cache       = load_meta_cache()
    papers_collected = []
    chunk_count      = 0

    for pdf_path in pdf_files:
        print(f"  Ingesting: {pdf_path.name}")
        reader     = PdfReader(str(pdf_path))
        full_text  = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        first_text = "\n\n".join(
            (reader.pages[i].extract_text() or "")
            for i in range(min(2, len(reader.pages)))
        )
        word_count = len(full_text.split())

        extracted = extract_pdf_metadata(
            pdf_path, first_text, meta_cache,
            use_spacy=use_spacy, use_llm=use_llm,
        )

        title   = extracted.get("title") or pdf_path.stem.replace("_", " ").title()
        authors = extracted.get("authors") or []
        year    = extracted.get("year")
        venue   = extracted.get("venue") or ""

        chunk_meta = {
            "source":  pdf_path.name,
            "title":   title,
            "type":    "pdf",
            "year":    str(year) if year else "",
            "venue":   venue,
            "authors": ", ".join(authors) if isinstance(authors, list) else (authors or ""),
        }
        chunks = chunk_text(full_text, chunk_meta)
        _upsert_chunks(collection, embedder, chunks)
        chunk_count += len(chunks)

        papers_collected.append({
            "title":      title,
            "authors":    authors if isinstance(authors, list) else [],
            "year":       year,
            "venue":      venue,
            "word_count": word_count,
            "source":     "pdf",
        })

    save_meta_cache(meta_cache)
    return chunk_count, papers_collected


def ingest_metadata_papers(collection, embedder) -> tuple[int, list[dict]]:
    """Ingest structured metadata from papers_metadata.py. Returns (chunk_count, papers)."""
    from papers_metadata import PAPERS

    papers_collected = []
    chunk_count      = 0

    for paper in PAPERS:
        print(f"  Ingesting: {paper['title'][:60]}...")
        text       = _build_paper_text(paper)
        word_count = len(text.split())
        authors    = paper.get("authors", [])

        chunks = chunk_text(text, {
            "source":   "metadata",
            "title":    paper["title"],
            "year":     str(paper.get("year", "")),
            "venue":    paper.get("venue", ""),
            "authors":  ", ".join(authors),
            "doi":      paper.get("doi", ""),
            "abstract": paper.get("abstract", ""),
            "keywords": ", ".join(paper.get("keywords", [])),
        })
        _upsert_chunks(collection, embedder, chunks)
        chunk_count += len(chunks)

        papers_collected.append({
            "title":      paper["title"],
            "authors":    authors,
            "year":       paper.get("year"),
            "venue":      paper.get("venue", ""),
            "word_count": word_count,
            "source":     "metadata",
        })

    return chunk_count, papers_collected


def _build_paper_text(paper: dict) -> str:
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
    if not chunks:
        return
    texts      = [c["text"]     for c in chunks]
    ids        = [c["id"]       for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=16).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion(reset: bool = False, use_spacy: bool = True, use_llm: bool = True):
    print("=== Research RAG — Ingestion ===")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embedder   = get_embedder()
    collection = get_chroma_collection(reset=reset)
    print(f"Existing vectors in store: {collection.count()}")

    print("\n--- PDF ingestion ---")
    pdf_chunks, pdf_papers = ingest_pdfs(
        collection, embedder, use_spacy=use_spacy, use_llm=use_llm
    )

    print("\n--- Metadata ingestion ---")
    meta_chunks, meta_papers = ingest_metadata_papers(collection, embedder)

    total = pdf_chunks + meta_chunks
    print(f"\nIngestion complete. Chunks added/updated: {total}")
    print(f"Total vectors in store: {collection.count()}\n")

    compute_and_save_corpus_stats(pdf_papers + meta_papers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest research papers into the RAG vector store")
    parser.add_argument("--reset",       action="store_true", help="Wipe and rebuild the vector store")
    parser.add_argument("--no-spacy",    action="store_true", help="Disable spaCy NLP extraction")
    parser.add_argument("--no-llm",      action="store_true", help="Disable LLM-based extraction")
    args = parser.parse_args()
    run_ingestion(
        reset=args.reset,
        use_spacy=not args.no_spacy,
        use_llm=not args.no_llm,
    )
