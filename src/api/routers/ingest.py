import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from src.api.dependencies import get_embedding_model, get_qdrant_client
from src.api.schemas import IngestResponse
from src.config.setting import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_index
from src.ingestion.loaders import load_document
from src.retrieval.bm25_cache import invalidate_cache

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    model=Depends(get_embedding_model),
    client=Depends(get_qdrant_client),
):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        documents = load_document(tmp_path)
        for doc in documents:
            doc.metadata["source"] = file.source
            doc.metadata["file_name"] = file.filename

        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        embed_and_index(chunks, model, client, settings.qdrant_collection)
        invalidate_cache(settings.qdrant_collection)

        return IngestResponse(
            status="success",
            chunks_indexed=len(chunks),
            message=f"Документ '{file.filename}' успешно проиндексирован",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
