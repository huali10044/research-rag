"""
app.py — Streamlit web interface for the Research Paper RAG system.

Run:  streamlit run app.py
"""

import logging
import os
import sys
import tomllib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

    top_k = st.slider("Sources to retrieve", min_value=2, max_value=16, value=4)
    show_sources = st.toggle("Show source details", value=True)

    st.divider()

    # ── Sample question browser (5 pages × 10 questions) ──────────────────────
    QUESTION_PAGES = [
        {
            "label": "Overview & Research Arc",
            "icon": "🗺️",
            "questions": [
                "Give me a high-level overview of your research portfolio.",
                "What are the main research themes across all publications?",
                "How has your research focus evolved over the years?",
                "What is the most frequently studied topic in this collection?",
                "Which application domains appear most often in the papers?",
                "How does the early work (pre-2010) compare to later work?",
                "What real-world problems does this research address?",
                "What is the broadest research contribution in this corpus?",
                "How would you characterize the overall research methodology?",
                "What is the single most important contribution in this body of work?",
            ],
        },
        {
            "label": "Metadata & Publications",
            "icon": "📋",
            "questions": [
                "How many papers are in this collection and what years do they span?",
                "Which publication venues appear most frequently?",
                "List all DARPA-related publications and their years.",
                "Which papers were published at KDD and what are they about?",
                "What papers appeared at MILCOM and in what years?",
                "Which co-authors appear most frequently across publications?",
                "What CIKM papers are included and what are their topics?",
                "List all papers published after 2015.",
                "Are there journal papers vs. conference papers in this collection?",
                "Which year had the most publications?",
            ],
        },
        {
            "label": "Algorithms & Methods",
            "icon": "⚙️",
            "questions": [
                "What specific machine learning algorithms are used across the papers?",
                "Describe the information retrieval methods used in the research.",
                "What neural network architectures appear in the papers?",
                "How is collaborative filtering applied in the user modeling work?",
                "What feature engineering techniques are described?",
                "Explain the ranking algorithms used in any of the papers.",
                "What probabilistic or Bayesian methods are employed?",
                "How are word embeddings or language representations used?",
                "What optimization techniques are described in the algorithms?",
                "Describe the evaluation metrics used across different papers.",
            ],
        },
        {
            "label": "Deep Technical Dives",
            "icon": "🔬",
            "questions": [
                "Explain the user modeling approach in technical detail.",
                "How is adversarial training or robustness addressed?",
                "What is the approach to handling temporal dynamics in user behavior?",
                "Describe any graph-based methods used in the research.",
                "How is multi-task learning applied across papers?",
                "What datasets are commonly used for experiments?",
                "How does the research handle cold-start problems?",
                "Describe any knowledge graph or ontology usage.",
                "What attention mechanisms or transformers are discussed?",
                "How is active learning or semi-supervised learning used?",
            ],
        },
        {
            "label": "Connections & Modern Relevance",
            "icon": "🔗",
            "questions": [
                "How does this research connect to modern large language models?",
                "What parallels exist between this work and modern RAG systems?",
                "How does the user modeling work relate to modern recommender systems?",
                "Which techniques from these papers are now standard in NLP/ML?",
                "How does the information retrieval work connect to neural IR?",
                "How would the described algorithms scale to today's data volumes?",
                "What open problems are identified that remain unsolved?",
                "How does this work compare to recent transformer-based approaches?",
                "What would a follow-up research agenda look like?",
                "How does my 2014 MILCOM paper relate to current RAG systems?",
            ],
        },
    ]

    if "q_page" not in st.session_state:
        st.session_state["q_page"] = 0

    page = st.session_state["q_page"]
    page_data = QUESTION_PAGES[page]

    # Page header
    st.markdown(f"**{page_data['icon']} {page_data['label']}**")

    # Questions
    for q in page_data["questions"]:
        if st.button(q, use_container_width=True, key=f"sq_{page}_{q[:40]}"):
            st.session_state["prefill"] = q

    # Navigation
    col_prev, col_indicator, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀", disabled=(page == 0), use_container_width=True, key="q_prev"):
            st.session_state["q_page"] -= 1
            st.rerun()
    with col_indicator:
        st.caption(f"Page {page + 1} of {len(QUESTION_PAGES)}", )
    with col_next:
        if st.button("▶", disabled=(page == len(QUESTION_PAGES) - 1), use_container_width=True, key="q_next"):
            st.session_state["q_page"] += 1
            st.rerun()

    # Page selector dots
    dot_cols = st.columns(len(QUESTION_PAGES))
    for i, col in enumerate(dot_cols):
        with col:
            marker = "●" if i == page else "○"
            if col.button(marker, key=f"q_dot_{i}", use_container_width=True):
                st.session_state["q_page"] = i
                st.rerun()

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
                logger.debug("query=%r backend=%s model=%s top_k=%d", effective_prompt, backend, model, top_k)
                answer, chunks, query_type = adaptive_answer(
                    effective_prompt, embedder, collection, top_k, backend, model
                )
                logger.debug("query_type=%s chunks=%d", query_type, len(chunks))
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
