"""Stage 1 indexing phase: Loader -> Splitter -> Embeddings (Ollama) -> Chroma."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

LOADERS_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    loader_cls = LOADERS_BY_SUFFIX.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    loader = loader_cls(str(path), encoding="utf-8") if loader_cls is TextLoader else loader_cls(str(path))
    return loader.load()


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        # Cosine space gives relevance scores that behave like the deck's
        # weak/strong examples (~0.3 vs ~0.8); Chroma's default L2 space
        # compresses everything below ~0.4, making the score_threshold
        # (Stage 2) meaningless.
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_document(path: Path, chunk_size: int, chunk_overlap: int) -> int:
    """Load, chunk, embed, and store a single document. Returns the chunk count added."""
    docs = load_document(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def ingest_directory(directory: Path, chunk_size: int, chunk_overlap: int) -> dict[str, int]:
    """Ingest every supported file in a directory. Returns {filename: chunk_count}."""
    results = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in LOADERS_BY_SUFFIX:
            results[path.name] = ingest_document(path, chunk_size, chunk_overlap)
    return results


def collection_count() -> int:
    """Number of chunks currently stored in the collection."""
    return get_vectorstore()._collection.count()


def reset_collection() -> None:
    """Delete all stored chunks — used when re-ingesting with different chunk settings."""
    get_vectorstore().delete_collection()
