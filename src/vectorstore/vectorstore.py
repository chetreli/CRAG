from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def ensure_multivector_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1024,
) -> None:
    """
    Создаёт коллекцию с тремя именованными векторами:
    - body: эмбеддинг текста чанка
    - summary: эмбеддинг краткого содержания чанка
    - context: эмбеддинг с учётом контекста документа
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        print(f"Коллекция '{collection_name}' уже существует")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "body": VectorParams(size=vector_size, distance=Distance.COSINE),
            "summary": VectorParams(size=vector_size, distance=Distance.COSINE),
            "context": VectorParams(size=vector_size, distance=Distance.COSINE),
        },
    )
    print(f"Multi-vector коллекция '{collection_name}' создана")
