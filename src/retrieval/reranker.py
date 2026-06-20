from FlagEmbedding import FlagReranker
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    file_name: str
    chunk_id: int
    score: float


_reranker_instance: FlagReranker | None = None


def get_reranker(use_fp16: bool = True) -> FlagReranker:
    """Singleton — модель загружается один раз и кешируется."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=use_fp16,
        )
    return _reranker_instance


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Пересортировывает чанки используя cross-encoder.
    В отличие от dense retrieval (который сравнивает эмбеддинги отдельно),
    re-ranker смотрит на пару (запрос, документ) ВМЕСТЕ — это даёт точнее результат.
    """
    if not chunks:
        return chunks

    reranker = get_reranker()

    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.compute_score(pairs, normalize=True)

    # Если один чанк, compute_score может вернуть float а не список
    if isinstance(scores, float):
        scores = [scores]

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for chunk, score in scored_chunks[:top_k]:
        chunk.score = float(score)
        reranked.append(chunk)

    return reranked