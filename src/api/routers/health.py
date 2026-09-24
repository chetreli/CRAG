from fastapi import APIRouter, Depends

from src.api.dependencies import get_qdrant_client
from src.cache.answer_cache import get_answer_cache
from src.config.setting import settings

router = APIRouter()


@router.get("/health")
def health(client=Depends(get_qdrant_client)):
    try:
        collections = client.get_collections()
        return {"status": "ok", "qdrant_collections": len(collections.collections)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@router.get("/cache/stats")
def cache_stats():
    return get_answer_cache().stats()

@router.delete("/cache")
def clear_cache():
    get_answer_cache().clear()
    return {"status": "cache cleared"}

@router.get("/grader/mode")
def get_grader_mode():
    return {
        "mode": settings.grader_mode,
        "description": {
            "llm": "Точный (LLM) — ~16-20 сек на запрос",
            "embedding": "Быстрый (Embedding) — ~1-2 сек на запрос",
        }[settings.grader_mode],
    }

@router.post("/grader/mode/{mode}")
def set_grader_mode(mode: str):
    if mode not in ("llm", "embedding"):
        from fastapi import HTTPException
        raise HTTPException(400, "Режим должен быть 'llm' или 'embedding'")
    settings.grader_mode = mode
    return {"status": "ok", "mode": mode}
