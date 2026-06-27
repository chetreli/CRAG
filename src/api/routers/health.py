from fastapi import APIRouter, Depends

from src.api.dependencies import get_qdrant_client

router = APIRouter()


@router.get("/health")
def health(client=Depends(get_qdrant_client)):
    try:
        collections = client.get_collections()
        return {"status": "ok", "qdrant_collections": len(collections.collections)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
