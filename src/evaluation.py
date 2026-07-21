"""Stage 3 — measure the Stage 2 agent against a golden set, using Ragas.

For each question in the golden set: run it through the Stage 2 agent, capture
(question, retrieved chunks, answer, ground truth), then score all four Ragas
metrics. Output a per-question breakdown plus aggregate averages.
"""

import json
from dataclasses import dataclass

import src._ragas_compat  # noqa: F401 - must run before importing ragas (see module docstring)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src import agent
from src.config import EMBEDDING_MODEL, GOLDEN_SET_PATH, GOOGLE_API_KEY, JUDGE_MODEL

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalRow:
    question: str
    answer: str
    ground_truth: str
    scores: dict[str, float]


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_golden_set(
    score_threshold: float, max_rewrites: int, rerank_top_n: int
) -> tuple[list[EvalRow], dict[str, float]]:
    """Run every golden-set question through the Stage 2 agent, score with Ragas.

    Returns (per-question rows, aggregate metric averages).
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env — required for the Ragas judge model.")

    golden_set = load_golden_set()

    questions, answers, contexts, references = [], [], [], []
    for item in golden_set:
        result = agent.ask(
            item["question"],
            score_threshold=score_threshold,
            max_rewrites=max_rewrites,
            rerank_top_n=rerank_top_n,
        )
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([c.page_content for c in result.source_chunks])
        references.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": references,
        }
    )

    judge = ChatGoogleGenerativeAI(model=JUDGE_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)  # local judge-side embeddings, no extra API cost

    result = evaluate(dataset, metrics=METRICS, llm=judge, embeddings=embeddings, raise_exceptions=False)
    df = result.to_pandas()

    rows = [
        EvalRow(
            question=row["user_input"],
            answer=row["response"],
            ground_truth=row["reference"],
            scores={name: row[name] for name in METRIC_NAMES},
        )
        for _, row in df.iterrows()
    ]
    aggregate = {name: df[name].mean() for name in METRIC_NAMES}

    return rows, aggregate
