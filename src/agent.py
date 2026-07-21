"""Stage 2 query phase — adaptive retrieval.

Question -> Embed -> search Chroma (wide, top-20)
  -> top score < score_threshold?  yes -> rewrite (LLM), re-search, recheck (bounded)
                                     no  -> proceed
  -> Cohere rerank (wide -> rerank_top_n)
  -> Prompt + chunks -> Ollama -> Answer   (identical chain to Stage 1)

The loop (conditional rewrite) is what makes this "agentic" rather than a
fixed chain — behaviour depends on retrieval quality, per query.
"""

from dataclasses import dataclass

from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import COHERE_API_KEY, LLM_MODEL, RERANK_MODEL, WIDE_RETRIEVAL_K
from src.ingestion import get_vectorstore
from src.query import generate_answer
from src.trace import Trace

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite a user's question to improve document retrieval. "
            "Keep the same intent and information need, but use more specific "
            "wording, synonyms, or terminology that might match how the answer "
            "is phrased in internal company documents (e.g. policy language "
            "rather than casual phrasing). Return only the rewritten question, "
            "nothing else — no preamble, no quotes.",
        ),
        ("human", "Original question: {question}"),
    ]
)


@dataclass
class AgentResult:
    answer: str
    source_chunks: list[Document]
    trace: Trace


def _search_with_top_score(question: str) -> tuple[list[Document], float]:
    results = get_vectorstore().similarity_search_with_relevance_scores(question, k=WIDE_RETRIEVAL_K)
    if not results:
        return [], 0.0
    docs = [doc for doc, _score in results]
    return docs, results[0][1]


def _rewrite_query(question: str, llm: ChatOllama) -> str:
    chain = REWRITE_PROMPT | llm
    return chain.invoke({"question": question}).content.strip().strip('"')


def _rerank(question: str, docs: list[Document], top_n: int) -> tuple[list[Document], bool]:
    """Returns (chunks, reranker_used). Falls back to wide-search order if no API key."""
    if not docs:
        return docs, False
    if not COHERE_API_KEY:
        return docs[:top_n], False
    reranker = CohereRerank(model=RERANK_MODEL, top_n=top_n, cohere_api_key=COHERE_API_KEY)
    return list(reranker.compress_documents(documents=docs, query=question)), True


def ask(question: str, score_threshold: float, max_rewrites: int, rerank_top_n: int) -> AgentResult:
    trace = Trace(question=question)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    current_question = question
    docs, score = _search_with_top_score(current_question)
    trace.log(
        "retrieve",
        act=f'Searched top-{WIDE_RETRIEVAL_K} for: "{current_question}"',
        observe=f"top relevance score = {score:.2f}",
    )

    rewrites = 0
    while score < score_threshold and rewrites < max_rewrites:
        rewrites += 1
        rewritten = _rewrite_query(current_question, llm)
        trace.log(
            "rewrite",
            reason=f"top score {score:.2f} < threshold {score_threshold} (attempt {rewrites}/{max_rewrites})",
            act=f'Rewrote query to: "{rewritten}"',
        )
        current_question = rewritten
        docs, score = _search_with_top_score(current_question)
        trace.log(
            "retrieve",
            act=f'Re-searched top-{WIDE_RETRIEVAL_K} for: "{current_question}"',
            observe=f"top relevance score = {score:.2f}",
        )

    if rewrites == 0:
        trace.log(
            "rewrite",
            reason=f"top score {score:.2f} >= threshold {score_threshold}",
            act="Skipped — initial retrieval was strong enough.",
        )
    elif score < score_threshold:
        trace.log(
            "rewrite",
            reason=f"still below threshold after {rewrites} rewrite(s)",
            act=f"Stopped — hit max_rewrites={max_rewrites} bound.",
        )

    reranked, reranker_used = _rerank(current_question, docs, top_n=rerank_top_n)
    trace.log(
        "rerank",
        act=(
            f"Cohere rerank ({RERANK_MODEL}): {len(docs)} candidates -> top {rerank_top_n}"
            if reranker_used
            else f"Reranker unavailable (no COHERE_API_KEY) — took top {rerank_top_n} from wide search instead"
        ),
        observe=f"{len(reranked)} chunks kept",
    )

    answer = generate_answer(question, reranked)
    trace.log(
        "generate",
        act=f"Answered with {LLM_MODEL} using {len(reranked)} chunks and the original question",
    )

    return AgentResult(answer=answer, source_chunks=reranked, trace=trace)
