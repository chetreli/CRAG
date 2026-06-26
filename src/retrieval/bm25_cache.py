import time

from qdrant_client import QdrantClient
from qdrant_client.models import Record
from rank_bm25 import BM25Okapi

from src.retrieval.sparse_retrieval import build_bm25_index

_cache: dict[str, tuple[BM25Okapi, list[Record], float]] = {}
_cache_ttl_seconds = 300  # пересобираем индекс каждые 5 минут максимум


def get_cached_bm25_index(
    client: QdrantClient,
    collection_name: str,
    force_rebuild: bool = False,
) -> tuple[BM25Okapi, list[Record]]:
    """
    Возвращает закешированный BM25 индекс.
    Перестраивает только если кеша нет, истёк TTL, или force_rebuild=True.
    """
    now = time.time()

    if not force_rebuild and collection_name in _cache:
        bm25, all_points, cached_at = _cache[collection_name]
        if now - cached_at < _cache_ttl_seconds:
            return bm25, all_points

    bm25, all_points = build_bm25_index(client, collection_name)
    _cache[collection_name] = (bm25, all_points, now)
    return bm25, all_points


def invalidate_cache(collection_name: str | None = None) -> None:
    """Сбрасывает кеш — вызывать после ingestion новых документов."""
    if collection_name:
        _cache.pop(collection_name, None)
    else:
        _cache.clear()
