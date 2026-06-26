from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_index
from src.ingestion.loaders import load_directory, load_document
from src.retrieval.bm25_cache import invalidate_cache


def run_ingestion_pipeline(
    source: Path,
    collection_name: str | None = None,
) -> None:
    collection_name = collection_name or settings.qdrant_collection

    print("=== Ingestion pipeline запущен ===")

    # 1. Загрузка
    if source.is_dir():
        documents = load_directory(source)
    else:
        documents = load_document(source)

    if not documents:
        print("Документы не найдены, завершение.")
        return

    # 2. Чанкинг
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 3. Эмбеддинг + индексация
    model = SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
    )
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    embed_and_index(chunks, model, client, collection_name)
    invalidate_cache(collection_name)
    print("=== Pipeline завершён ✓ ===")
