"""Stage 3 — Measured agentic RAG. Streamlit UI: Ingest, Ask, Settings, Trace, Evaluate.

Extends Stage 2: adds a golden-set evaluation harness (Ragas, LLM-as-judge)
so changes to Stage 1/2 become measurable instead of vibes-based.
"""

import streamlit as st

from src.agent import ask as agentic_ask
from src.config import (
    COHERE_API_KEY,
    DEFAULT_CONFIG,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    JUDGE_MODEL,
    LLM_MODEL,
    RERANK_MODEL,
    UPLOADS_DIR,
    WIDE_RETRIEVAL_K,
)
from src.evaluation import evaluate_single, run_golden_set
from src.ingestion import collection_count, ingest_directory, ingest_document, reset_collection

st.set_page_config(page_title="RAG Workshop — Stage 3", layout="wide", initial_sidebar_state="expanded")

for key in ("chunk_size", "chunk_overlap", "search_k", "score_threshold", "max_rewrites", "rerank_top_n"):
    if key not in st.session_state:
        st.session_state[key] = getattr(DEFAULT_CONFIG, key)
if "trace_history" not in st.session_state:
    st.session_state.trace_history = []

# Model transparency (PRD §6.2, R0.1/R0.3) — visible on every view.
st.sidebar.markdown("### Active models")
st.sidebar.markdown(f"**LLM:** `{LLM_MODEL}` (Ollama)")
st.sidebar.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}` (Ollama)")
if COHERE_API_KEY:
    st.sidebar.markdown(f"**Reranker:** `{RERANK_MODEL}` (Cohere)")
else:
    st.sidebar.markdown("**Reranker:** not configured (`COHERE_API_KEY` missing — falling back to wide-search order)")
if GOOGLE_API_KEY:
    st.sidebar.markdown(f"**Judge:** `{JUDGE_MODEL}` (Gemini)")
else:
    st.sidebar.markdown("**Judge:** not configured (`GOOGLE_API_KEY` missing — Evaluate view will error)")
st.sidebar.divider()

view = st.sidebar.radio("View", ["Ingest", "Ask", "Settings", "Trace", "Evaluate"])

st.title("Stage 3 — Measured Agentic RAG")

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
    st.caption(
        "Runs the Stage 2 adaptive pipeline: wide retrieval → conditional rewrite → "
        "rerank → answer, then scores the answer live with Ragas (faithfulness, answer "
        "relevance, context precision, context recall). See the Trace view for exactly what fired."
    )
    if collection_count() == 0:
        st.warning("No documents ingested yet — go to the Ingest view first.")
    question = st.text_input("Question")
    if st.button("Ask", disabled=not question):
        with st.spinner("Retrieving, adapting, and generating..."):
            result = agentic_ask(
                question,
                score_threshold=st.session_state.score_threshold,
                max_rewrites=st.session_state.max_rewrites,
                rerank_top_n=st.session_state.rerank_top_n,
            )
        st.session_state.trace_history.insert(0, result.trace)

        st.markdown("### Answer")
        st.write(result.answer)
        st.markdown(f"### Source chunks (top {st.session_state.rerank_top_n}, reranked)")
        for i, chunk in enumerate(result.source_chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            with st.expander(f"[{i}] {source}"):
                st.write(chunk.page_content)
        st.info("Full decision timeline for this question is in the Trace view.")

        st.markdown("### Real-time eval (Ragas)")
        if not GOOGLE_API_KEY:
            st.caption("`GOOGLE_API_KEY` is not set in `.env` — skipping live scoring for this answer.")
        else:
            try:
                with st.spinner(f"Scoring this answer with Ragas (judge: {JUDGE_MODEL})..."):
                    scores = evaluate_single(
                        question,
                        result.answer,
                        [c.page_content for c in result.source_chunks],
                    )
            except Exception as exc:
                st.warning(f"Live scoring failed ({type(exc).__name__}: {exc}) — the answer above is unaffected.")
            else:
                cols = st.columns(len(scores))
                for col, (metric, score) in zip(cols, scores.items()):
                    col.metric(metric.replace("_", " "), f"{score:.2f}")
                st.caption(
                    "There's no golden-set ground truth for an ad-hoc question, so context precision/recall use "
                    "the generated answer itself as a stand-in reference — treat those two as directional, not "
                    "authoritative. Faithfulness and answer relevance don't need a reference and aren't affected. "
                    "For trustworthy precision/recall, use the Evaluate view's golden-set run."
                )

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

    st.subheader("Retrieval (Stage 1, unused now that Ask runs the Stage 2 pipeline)")
    st.session_state.search_k = st.number_input(
        "search_k", min_value=1, max_value=20, step=1, value=st.session_state.search_k
    )
    st.caption("Kept for reference/future direct use of src.query.ask(); the live Ask view uses rerank_top_n instead.")

    st.subheader("Adaptive retrieval (Stage 2, takes effect on next query)")
    st.session_state.score_threshold = st.slider(
        "score_threshold", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.score_threshold
    )
    st.caption(f"Wide candidate pool is fixed at top-{WIDE_RETRIEVAL_K} (not configurable, per the workshop design).")
    st.session_state.max_rewrites = st.number_input(
        "max_rewrites", min_value=0, max_value=5, step=1, value=st.session_state.max_rewrites
    )
    st.session_state.rerank_top_n = st.number_input(
        "rerank_top_n", min_value=1, max_value=10, step=1, value=st.session_state.rerank_top_n
    )

    st.divider()
    st.subheader("Active models")
    st.write(f"**LLM:** `{LLM_MODEL}` via Ollama, temperature=0")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}` via Ollama")
    if COHERE_API_KEY:
        st.write(f"**Reranker:** `{RERANK_MODEL}` via Cohere")
    else:
        st.write(f"**Reranker:** `{RERANK_MODEL}` via Cohere — **not active**, `COHERE_API_KEY` is not set in `.env`")
    if GOOGLE_API_KEY:
        st.write(f"**Judge (Stage 3):** `{JUDGE_MODEL}` via Gemini")
    else:
        st.write(f"**Judge (Stage 3):** `{JUDGE_MODEL}` via Gemini — **not active**, `GOOGLE_API_KEY` is not set in `.env`")
    st.caption("Model selection is config-driven, not editable here — see src/config.py.")

elif view == "Trace":
    st.header("Trace")
    st.caption("Every decision the Stage 2 agent made, most recent question first.")

    if not st.session_state.trace_history:
        st.info("No queries yet — ask something in the Ask view first.")
    for trace in st.session_state.trace_history:
        with st.expander(f'{trace.timestamp} — "{trace.question}"', expanded=(trace is st.session_state.trace_history[0])):
            st.markdown(f"**Answer:** {trace.answer}")
            st.divider()
            for step in trace.steps:
                st.markdown(f"**{step.label.upper()}**")
                if step.reason:
                    st.markdown(f"- Reason: {step.reason}")
                st.markdown(f"- Act: {step.act}")
                if step.observe:
                    st.markdown(f"- Observe: {step.observe}")
                st.markdown("")

elif view == "Evaluate":
    st.header("Evaluate")
    st.caption(
        "Runs the 10-question golden set through the Stage 2 agent and scores every "
        "answer with Ragas (LLM-as-judge). Uses whatever score_threshold/max_rewrites/"
        "rerank_top_n are currently set in Settings."
    )
    if GOOGLE_API_KEY:
        st.write(f"**Judge model:** `{JUDGE_MODEL}` via Gemini")
    else:
        st.error("`GOOGLE_API_KEY` is not set in `.env` — the judge model can't run.")

    if st.button("Run evaluation", disabled=not GOOGLE_API_KEY):
        with st.spinner(
            "Running golden set through the agent (paced to respect Cohere's trial rate limit), "
            "then scoring with Ragas (~40 LLM calls) — budget a few minutes..."
        ):
            rows, aggregate = run_golden_set(
                score_threshold=st.session_state.score_threshold,
                max_rewrites=st.session_state.max_rewrites,
                rerank_top_n=st.session_state.rerank_top_n,
            )
        st.session_state.eval_result = {"rows": rows, "aggregate": aggregate}

    if "eval_result" in st.session_state:
        rows = st.session_state.eval_result["rows"]
        aggregate = st.session_state.eval_result["aggregate"]

        st.subheader("Scorecard")
        cols = st.columns(len(aggregate))
        weakest_metric = min(aggregate, key=aggregate.get)
        for col, (metric, score) in zip(cols, aggregate.items()):
            label = metric.replace("_", " ")
            col.metric(label, f"{score:.2f}", help="Weakest metric" if metric == weakest_metric else None)
        st.info(
            f"**Weakest metric: {weakest_metric.replace('_', ' ')}** ({aggregate[weakest_metric]:.2f}) — "
            f"{'improve retrieval (recall/precision)' if 'context' in weakest_metric else 'improve generation (prompt/grounding)'} "
            "is the next place to invest."
        )

        st.subheader("Per-question breakdown")
        st.dataframe(
            [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "ground_truth": r.ground_truth,
                    **{k: round(v, 2) for k, v in r.scores.items()},
                }
                for r in rows
            ],
            use_container_width=True,
        )
    else:
        st.info("No evaluation run yet — click 'Run evaluation' above.")
