import logging
import time

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.retrieval.dence_retrieval import dense_search
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.sparse_retrieval import build_bm25_index, sparse_search

logging.getLogger("transformers").setLevel(logging.ERROR)


def profile_query(query: str, model, client, top_k: int = 10):
    print(f"\n{'=' * 60}")
    print(f"Запрос: {query}")
    print(f"{'=' * 60}")

    total_start = time.time()

    # 1. Эмбеддинг запроса
    t0 = time.time()
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    t_embed = time.time() - t0
    print(f"  Эмбеддинг запроса:{query_vector} {t_embed:.4f} сек")

    # 2. Dense search в Qdrant
    t0 = time.time()
    dense_results = dense_search(query, model, client, settings.qdrant_collection, top_k=top_k * 2)
    t_dense = time.time() - t0
    print(f"  Dense search (Qdrant):    {t_dense:.4f} сек  ({len(dense_results)} результатов)")

    # 3. Построение BM25 индекса (читает ВСЮ коллекцию из Qdrant)
    t0 = time.time()
    bm25, all_points = build_bm25_index(client, settings.qdrant_collection)
    t_bm25_build = time.time() - t0
    print(f"  Построение BM25 индекса:  {t_bm25_build:.4f} сек  ({len(all_points)} документов)")

    # 4. Сам sparse поиск (после того как индекс построен)
    t0 = time.time()
    sparse_results = sparse_search(query, bm25, all_points, top_k=top_k * 2)
    t_sparse_search = time.time() - t0
    print(f"  Sparse search (BM25):     {t_sparse_search:.4f} сек")

    # 5. RRF слияние
    t0 = time.time()
    ranked, doc_map = reciprocal_rank_fusion(dense_results, sparse_results)
    t_fusion = time.time() - t0
    print(f"  RRF слияние:              {t_fusion:.4f} сек")

    total = time.time() - total_start
    print(f"\n  ИТОГО:                    {total:.4f} сек")

    print("\n  Доля времени по этапам:")
    print(f"    Эмбеддинг запроса:      {t_embed / total * 100:5.1f}%")
    print(f"    Dense search:           {t_dense / total * 100:5.1f}%")
    print(f"    Построение BM25:        {t_bm25_build / total * 100:5.1f}%")
    print(f"    Sparse search:          {t_sparse_search / total * 100:5.1f}%")
    print(f"    RRF слияние:            {t_fusion / total * 100:5.1f}%")


if __name__ == "__main__":
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    queries = [
        "что такое безопасность жизнедеятельности",
        "что вызывает кислотные дожди",
    ]

    for q in queries:
        profile_query(q, model, client)
