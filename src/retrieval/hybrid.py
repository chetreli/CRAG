from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Record, ScoredPoint

from src.retrieval.dence_retrieval import dense_search
from src.retrieval.sparse_retrieval import build_bm25_index, sparse_search
from src.retrieval.reranker import rerank_chunks


@dataclass
class RetrievedChunk:
    text: str
    source: str
    file_name: str
    chunk_id: int
    score: float


def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    sparse_results: list[tuple[Record, float]],
    k: int = 60,
) -> tuple[list[tuple[str, float]], dict[str, dict]]:
    """RRF объединяет ранги из двух списков без нормализации скоров."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, point in enumerate(dense_results):
        doc_id = str(point.id)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = point.payload

    for rank, (record, _) in enumerate(sparse_results):
        doc_id = str(record.id)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = record.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked], doc_map


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
    use_hyde: bool = False,
    use_reranker: bool = False,
) -> list[RetrievedChunk]:
    # Если будем делать re-ranking — берём больше кандидатов на входе
    fetch_k = top_k * 3 if use_reranker else top_k * 2

    dense_results = dense_search(
        query, model, client, collection_name,
        top_k=fetch_k, use_hyde=use_hyde,
    )

    bm25, all_points = build_bm25_index(client, collection_name)
    sparse_results = sparse_search(query, bm25, all_points, top_k=fetch_k)

    ranked, doc_map = reciprocal_rank_fusion(dense_results, sparse_results)

    chunks = []
    for doc_id, score in ranked[:fetch_k]:
        payload = doc_map[doc_id]
        chunks.append(RetrievedChunk(
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            file_name=payload.get("file_name", "unknown"),
            chunk_id=payload.get("chunk_id", -1),
            score=score,
        ))

    if use_reranker:
        chunks = rerank_chunks(query, chunks, top_k=top_k)
    else:
        chunks = chunks[:top_k]

    return chunks