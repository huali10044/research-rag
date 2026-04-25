"""
app.py — Streamlit web interface for the Research Paper RAG system.

Run:  streamlit run app.py
"""

import os
import sys
import tomllib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from sentence_transformers import SentenceTransformer
from rag import adaptive_answer, classify_query, get_collection, resolve_backend, BACKENDS

# ── Inject Streamlit secrets into os.environ (for Streamlit Cloud) ────────────
for _key, _val in st.secrets.items():
    if isinstance(_val, str):
        os.environ.setdefault(_key, _val)

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(_CONFIG_PATH, "rb") as _f:
    _CONFIG = tomllib.load(_f)

_DEFAULT_BACKEND = _CONFIG["llm"]["backend"]
_DEFAULT_MODEL   = _CONFIG["llm"]["model"] or None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Assistant — Dr. Hua Li",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research RAG")
    st.markdown("**Dr. Hua Li — Publication Assistant**")
    st.divider()

    backend_options = list(BACKENDS.keys())
    default_idx = backend_options.index(_DEFAULT_BACKEND) if _DEFAULT_BACKEND in backend_options else 0
    selected_backend = st.selectbox("LLM Backend", backend_options, index=default_idx)

    backend_cfg = BACKENDS[selected_backend]
    default_model_for_backend = _DEFAULT_MODEL if (
        _DEFAULT_BACKEND == selected_backend and _DEFAULT_MODEL
    ) else backend_cfg["default_model"]

    selected_model = st.text_input(
        "Model",
        value=default_model_for_backend,
        key=f"model_{selected_backend}_{default_model_for_backend}",
    )

    st.caption(backend_cfg["notes"])
    st.divider()

    top_k = st.slider("Sources to retrieve", min_value=2, max_value=8, value=4)
    show_sources = st.toggle("Show source details", value=True)

    st.divider()
    st.markdown("**Sample questions:**")
    sample_questions = [
        "What work have I done on user modeling?",
        "Summarize my DARPA-related publications",
        "How does my 2014 MILCOM paper relate to RAG systems?",
        "What are the themes across my KDD and CIKM papers?",
        "What methods did I use for information retrieval?",
        "How does my work connect to modern LLM engineering?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    collection = get_collection()
    return embedder, collection


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Research Paper Assistant")
st.caption("Ask questions about Dr. Hua Li's published research. Answers are grounded in actual paper content.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_sources:
            srcs       = message.get("sources", [])
            qtype      = message.get("query_type", "")
            label      = f"Sources · {qtype} query" if qtype else "Sources"
            if srcs:
                label += f" · {len(srcs)} chunks"
            with st.expander(label):
                if not srcs:
                    st.caption("Answer derived from corpus metadata — no text chunks retrieved.")
                for i, src in enumerate(srcs, 1):
                    meta = src["metadata"]
                    st.markdown(
                        f"**{i}. {meta.get('title', 'Unknown')}** ({meta.get('year', '?')})  \n"
                        f"*{meta.get('venue', '')}*  \n"
                        f"Similarity: `{src['similarity']:.3f}` · Chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}  \n\n"
                        f"> {src['text'][:300]}..."
                    )

# Prefill from sidebar button clicks
prefill = st.session_state.pop("prefill", "")

# Input
if prompt := st.chat_input("Ask a question about your research...", key="chat_input") or prefill:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Resolve "rerun" / "repeat" commands to the last real user question
    effective_prompt = prompt
    if classify_query(prompt) == "command":
        prior_questions = [
            m["content"] for m in st.session_state.messages
            if m["role"] == "user" and classify_query(m["content"]) != "command"
        ]
        effective_prompt = prior_questions[-1] if prior_questions else prompt

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            try:
                embedder, collection = load_resources()
                backend, model = resolve_backend(selected_backend)
                if selected_model:
                    model = selected_model
                answer, chunks, query_type = adaptive_answer(
                    effective_prompt, embedder, collection, top_k, backend, model
                )
                st.markdown(answer)

                if show_sources:
                    label = f"Sources · {query_type} query"
                    if chunks:
                        label += f" · {len(chunks)} chunks"
                    with st.expander(label):
                        if not chunks:
                            st.caption("Answer derived from corpus metadata — no text chunks retrieved.")
                        for i, src in enumerate(chunks, 1):
                            meta = src["metadata"]
                            st.markdown(
                                f"**{i}. {meta.get('title', 'Unknown')}** ({meta.get('year', '?')})  \n"
                                f"*{meta.get('venue', '')}*  \n"
                                f"Similarity: `{src['similarity']:.3f}`  \n\n"
                                f"> {src['text'][:300]}..."
                            )
            except Exception as e:
                answer     = f"Error: {e}\n\nMake sure you've run `python src/ingest.py` first."
                chunks     = []
                query_type = ""
                st.error(answer)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "sources":    chunks,
        "query_type": query_type,
    })
