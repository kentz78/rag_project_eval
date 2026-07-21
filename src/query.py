"""Stage 1 query phase: Question -> Embed -> top-k chunks -> Prompt -> Ollama -> Answer.

Fixed chain, no tools, no loop, no memory — every question stands alone.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_MODEL
from src.ingestion import get_vectorstore

SYSTEM_PROMPT = (
    "You answer using only the provided context. "
    "If the context doesn't cover the question, say so plainly instead of guessing."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class QueryResult:
    answer: str
    source_chunks: list[Document]


def format_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {c.metadata.get('source', 'unknown')}) {c.page_content}"
        for i, c in enumerate(chunks)
    )


def ask(question: str, search_k: int) -> QueryResult:
    chunks = get_vectorstore().similarity_search(question, k=search_k)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(chunks), "question": question})
    return QueryResult(answer=answer, source_chunks=chunks)
