"""Runtime configuration for the RAG pipeline.

Defaults live here; the Streamlit Settings view overrides them at runtime
via st.session_state (see app.py). Nothing in ingestion.py, query.py, or
agent.py should hardcode these values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
COLLECTION_NAME = "rag_docs"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
RERANK_MODEL = "rerank-v3.5"
JUDGE_MODEL = "gemini-flash-lite-latest"

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_set.json"

# Wide candidate pool searched before reranking (Stage 2). Fixed per the
# workshop design, not user-configurable — only the final rerank_top_n is.
WIDE_RETRIEVAL_K = 20

COHERE_API_KEY = os.environ.get("COHERE_API_KEY") or None
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or None


@dataclass
class RuntimeConfig:
    # Stage 1
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 3
    # Stage 2
    score_threshold: float = 0.5
    max_rewrites: int = 2
    rerank_top_n: int = 3


DEFAULT_CONFIG = RuntimeConfig()
