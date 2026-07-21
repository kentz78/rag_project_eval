"""Runtime configuration for the RAG pipeline.

Defaults live here; the Streamlit Settings view overrides them at runtime
via st.session_state (see app.py). Nothing in ingestion.py or query.py
should hardcode these values.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
COLLECTION_NAME = "rag_docs"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


@dataclass
class RuntimeConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 3


DEFAULT_CONFIG = RuntimeConfig()
