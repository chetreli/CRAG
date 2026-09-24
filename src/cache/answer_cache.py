import hashlib
import time
from dataclasses import dataclass


@dataclass
class CachedAnswer:
    answer: str
    source: str
    used_fallback: bool
    created_at: float
    hits: int = 0


class AnswerCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        self._cache: dict[str, CachedAnswer] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> CachedAnswer | None:
        key = self._make_key(query)
        entry = self._cache.get(key)

        if entry is None:
            return None

        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            return None

        entry.hits += 1
        return entry

    def set(self, query: str, answer: str, source: str, used_fallback: bool) -> None:
        if len(self._cache) >= self._max_size:
            # Удаляем самую старую запись
            oldest = min(self._cache.items(), key=lambda x: x[1].created_at)
            del self._cache[oldest[0]]

        key = self._make_key(query)
        self._cache[key] = CachedAnswer(
            answer=answer,
            source=source,
            used_fallback=used_fallback,
            created_at=time.time(),
        )

    def stats(self) -> dict:
        total = len(self._cache)
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "cached_queries": total,
            "total_hits": total_hits,
            "ttl_seconds": self._ttl,
        }

    def clear(self) -> None:
        self._cache.clear()


# Глобальный синглтон — живёт всё время работы FastAPI процесса
_cache_instance: AnswerCache | None = None


def get_answer_cache() -> AnswerCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AnswerCache(ttl_seconds=3600, max_size=100)
    return _cache_instance
