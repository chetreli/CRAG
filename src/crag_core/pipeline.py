from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.crag_core.generator import (
    generate_from_chunks,
    generate_from_web,
    generate_no_context,
)
from src.crag_core.grader import grade_chunks
from src.crag_core.query_rewriter import rewrite_query
from src.crag_core.web_search import web_search
from src.retrieval.hybrid import RetrievedChunk, hybrid_search


@dataclass
class CRAGResult:
    query: str
    rewritten_query: str | None
    answer: str
    source: str
    relevant_chunks: list[RetrievedChunk]
    used_fallback: bool


def is_context_sufficient(
    relevant: list[RetrievedChunk],
    min_relevant: int = 2,
    min_avg_score: float = 0.6,
) -> bool:
    """
    Контекст достаточен если:
    - релевантных чанков >= min_relevant
    - И средний score >= min_avg_score
    """
    if len(relevant) < min_relevant:
        return False
    avg_score = sum(c.score for c in relevant) / len(relevant)
    return avg_score >= min_avg_score


def run_crag_pipeline(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    top_k: int = 10,
    threshold: float = 0.6,
    min_relevant: int = 2,
) -> CRAGResult:
    print(f"\n{'=' * 50}")
    print(f"Запрос: {query}")

    # 1. Retrieval
    chunks = hybrid_search(query, model, client, settings.qdrant_collection, top_k=top_k)
    print(f"Найдено чанков: {len(chunks)}")

    # 2. Grading
    print("Оценка релевантности...")
    relevant, irrelevant = grade_chunks(query, chunks, threshold=threshold)
    print(f"Релевантных: {len(relevant)} / {len(chunks)}")

    # 3. Если мало релевантных — rewrite + повторный retrieval
    rewritten_query = None
    if not is_context_sufficient(relevant, min_relevant):
        print("Мало релевантных чанков — переформулируем запрос...")
        rewritten_query = rewrite_query(query)
        print(f"Переформулировка: {rewritten_query}")

        new_chunks = hybrid_search(
            rewritten_query, model, client, settings.qdrant_collection, top_k=top_k
        )
        new_relevant, _ = grade_chunks(rewritten_query, new_chunks, threshold=threshold)
        relevant = relevant + new_relevant
        print(f"После переформулировки релевантных: {len(relevant)}")

    # 4. Fallback на web search если всё ещё мало
    if not is_context_sufficient(relevant, min_relevant):
        print("Запускаем web search fallback...")
        web_results = web_search(rewritten_query or query, max_results=5)

        if web_results:
            answer = generate_from_web(query, web_results)
            return CRAGResult(
                query=query,
                rewritten_query=rewritten_query,
                answer=answer,
                source="web",
                relevant_chunks=[],
                used_fallback=True,
            )
        else:
            answer = generate_no_context(query)
            return CRAGResult(
                query=query,
                rewritten_query=rewritten_query,
                answer=answer,
                source="no_context",
                relevant_chunks=[],
                used_fallback=True,
            )

    print("Генерируем ответ из локальной базы...")
    answer = generate_from_chunks(query, relevant[:5])

    return CRAGResult(
        query=query,
        rewritten_query=rewritten_query,
        answer=answer,
        source="local",
        relevant_chunks=relevant,
        used_fallback=False,
    )
