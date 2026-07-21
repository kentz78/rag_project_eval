"""Stage 1 — Classic RAG. Streamlit UI: Ingest, Ask, Settings."""

import streamlit as st

from src.config import DEFAULT_CONFIG, DOCUMENTS_DIR, EMBEDDING_MODEL, LLM_MODEL, UPLOADS_DIR
from src.ingestion import collection_count, ingest_directory, ingest_document, reset_collection
from src.query import ask

st.set_page_config(page_title="RAG Workshop — Stage 1", layout="wide")

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = DEFAULT_CONFIG.chunk_size
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = DEFAULT_CONFIG.chunk_overlap
if "search_k" not in st.session_state:
    st.session_state.search_k = DEFAULT_CONFIG.search_k

# Model transparency (PRD §6.2, R0.1) — visible on every view.
st.sidebar.markdown("### Active models")
st.sidebar.markdown(f"**LLM:** `{LLM_MODEL}` (Ollama)")
st.sidebar.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}` (Ollama)")
st.sidebar.divider()

view = st.sidebar.radio("View", ["Ingest", "Ask", "Settings"])

st.title("Stage 1 — Classic RAG")

if view == "Ingest":
    st.header("Ingest documents")
    st.caption(f"Chunks currently stored: **{collection_count()}**")

    st.subheader("Sample corpus")
    st.write(f"Ingest the bundled sample documents from `{DOCUMENTS_DIR.relative_to(DOCUMENTS_DIR.parent.parent)}`.")
    if st.button("Ingest sample corpus"):
        with st.spinner("Chunking and embedding sample documents..."):
            results = ingest_directory(
                DOCUMENTS_DIR,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )
        st.success(f"Ingested {sum(results.values())} chunks from {len(results)} files.")
        st.table({"file": list(results.keys()), "chunks": list(results.values())})

    st.divider()
    st.subheader("Upload your own")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest uploaded files"):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Chunking and embedding uploaded documents..."):
            counts = {}
            for uploaded in uploaded_files:
                dest = UPLOADS_DIR / uploaded.name
                dest.write_bytes(uploaded.getvalue())
                counts[uploaded.name] = ingest_document(
                    dest,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
        st.success(f"Ingested {sum(counts.values())} chunks from {len(counts)} files.")
        st.table({"file": list(counts.keys()), "chunks": list(counts.values())})

    st.divider()
    if st.button("Reset store (delete all chunks)", type="secondary"):
        reset_collection()
        st.success("Store cleared. Re-ingest to query again.")

elif view == "Ask":
    st.header("Ask a question")
    if collection_count() == 0:
        st.warning("No documents ingested yet — go to the Ingest view first.")
    question = st.text_input("Question")
    if st.button("Ask", disabled=not question):
        with st.spinner("Retrieving and generating..."):
            result = ask(question, search_k=st.session_state.search_k)
        st.markdown("### Answer")
        st.write(result.answer)
        st.markdown(f"### Source chunks (top {st.session_state.search_k})")
        for i, chunk in enumerate(result.source_chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            with st.expander(f"[{i}] {source}"):
                st.write(chunk.page_content)

elif view == "Settings":
    st.header("Settings")
    st.caption("Changes apply on the next ingest/query — no restart needed.")

    st.subheader("Chunking (takes effect on next ingestion)")
    st.session_state.chunk_size = st.number_input(
        "chunk_size", min_value=100, max_value=4000, step=100, value=st.session_state.chunk_size
    )
    st.session_state.chunk_overlap = st.number_input(
        "chunk_overlap", min_value=0, max_value=1000, step=50, value=st.session_state.chunk_overlap
    )
    if st.session_state.chunk_overlap >= st.session_state.chunk_size:
        st.error("chunk_overlap must be smaller than chunk_size.")

    st.subheader("Retrieval (takes effect on next query)")
    st.session_state.search_k = st.number_input(
        "search_k", min_value=1, max_value=20, step=1, value=st.session_state.search_k
    )

    st.divider()
    st.subheader("Active models")
    st.write(f"**LLM:** `{LLM_MODEL}` via Ollama, temperature=0")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}` via Ollama")
    st.caption("Model selection is config-driven, not editable here — see src/config.py.")
