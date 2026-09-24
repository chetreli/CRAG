import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid import RetrievedChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя векторами."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def grade_chunks_embedding(
    query: str,
    chunks: list[RetrievedChunk],
    model: SentenceTransformer,
    threshold: float = 0.5,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    Оценивает релевантность чанков через косинусное сходство эмбеддингов.
    Работает мгновенно — без LLM вызовов.
    threshold здесь имеет другой смысл чем в LLM grader:
    - 0.3-0.4: мягкий фильтр, пропускает больше
    - 0.5-0.6: сбалансированный
    - 0.7+: строгий фильтр
    """
    if not chunks:
        return [], []

    # Эмбеддинг запроса
    query_vector = model.encode(query, normalize_embeddings=True)

    # Эмбеддинги всех чанков одним батчем — очень быстро
    texts = [c.text for c in chunks]
    chunk_vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=len(chunks),
        show_progress_bar=False,
    )

    relevant = []
    irrelevant = []

    for i, chunk in enumerate(chunks):
        score = cosine_similarity(query_vector, chunk_vectors[i])
        chunk.score = float(score)

        print(f"  score={score:.3f} | {chunk.file_name}")

        if score >= threshold:
            relevant.append(chunk)
        else:
            irrelevant.append(chunk)

    return relevant, irrelevant
