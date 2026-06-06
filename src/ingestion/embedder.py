from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer
import uuid


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1024,
) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Коллекция '{collection_name}' создана")
    else:
        print(f"Коллекция '{collection_name}' уже существует")


def embed_and_index(
    chunks: list[Document],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 64,
) -> None:
    ensure_collection(client, collection_name)

    texts = [chunk.page_content for chunk in chunks]

    print(f"Эмбеддинг {len(texts)} чанков...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i].page_content,
                **chunks[i].metadata,
            },
        )
        for i in range(len(chunks))
    ]

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"Проиндексировано точек: {len(points)} ✓")