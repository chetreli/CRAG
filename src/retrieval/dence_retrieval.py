from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

from src.crag_core.hyde import embed_with_hyde


def dense_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
    use_hyde: bool = False,
) -> list[ScoredPoint]:
    if use_hyde:
        query_vector = embed_with_hyde(query, model)
    else:
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points


def dense_search_multivector(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
    vector_name: str = "body",  # "body" | "summary" | "context"
) -> list[ScoredPoint]:
    """Поиск по конкретному вектору в multi-vector коллекции."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=top_k,
        with_payload=True,
    )
    return response.points


def dense_search_multivector_fusion(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
) -> list[ScoredPoint]:
    """
    Ищет по всем трём векторам и объединяет результаты через RRF.
    Это даёт устойчивость — даже если запрос плохо матчится с body,
    он может хорошо совпасть с summary или context.
    """
    all_results = {}
    for vector_name in ["body", "summary", "context"]:
        results = dense_search_multivector(
            query,
            model,
            client,
            collection_name,
            top_k=top_k * 2,
            vector_name=vector_name,
        )
        for rank, point in enumerate(results):
            doc_id = str(point.id)
            score_boost = 1 / (60 + rank + 1)
            if doc_id not in all_results:
                all_results[doc_id] = {"point": point, "score": 0.0}
            all_results[doc_id]["score"] += score_boost

    ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    return [r["point"] for r in ranked[:top_k]]
