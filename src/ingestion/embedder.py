from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer
from src.vectorstore.vectorstore import ensure_multivector_collection
from src.crag_core.summarizer import summarize_chunk
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

    # Загружаем батчами чтобы не перегружать Qdrant
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"Проиндексировано точек: {len(points)} ✓")


def embed_and_index_multivector(
    chunks: list[Document],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 32,
) -> None:
    """
    Индексирует чанки с тремя векторами: body, summary, context.
    """
    ensure_multivector_collection(client, collection_name)

    points = []
    print(f"Генерация summary и эмбеддингов для {len(chunks)} чанков...")

    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        file_name = chunk.metadata.get("file_name", "unknown")

        # Контекст = имя файла + текст (для тематической привязки)
        context_text = f"Документ: {file_name}\n{text}"

        # Summary через LLM
        summary_text = summarize_chunk(text)

        # Эмбеддинги для всех трёх представлений
        body_vec = model.encode(text, normalize_embeddings=True).tolist()
        summary_vec = model.encode(summary_text, normalize_embeddings=True).tolist()
        context_vec = model.encode(context_text, normalize_embeddings=True).tolist()

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "body": body_vec,
                "summary": summary_vec,
                "context": context_vec,
            },
            payload={
                "text": text,
                "summary": summary_text,
                **chunk.metadata,
            },
        ))

        if (i + 1) % 50 == 0:
            print(f"  обработано {i + 1}/{len(chunks)}")

    print(f"Загрузка {len(points)} точек в Qdrant...")
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print("Multi-vector индексация завершена ✓")