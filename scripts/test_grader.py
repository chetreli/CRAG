import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.crag_core.grader import grade_chunks, should_fallback
from src.retrieval.hybrid import hybrid_search

logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

query = "безопасность жизнедеятельности опасные факторы"
print(f"Запрос: {query}\n")

chunks = hybrid_search(query, model, client, settings.qdrant_collection, top_k=5)
print(f"Найдено чанков: {len(chunks)}\n")

print("Оценка релевантности:")
relevant, irrelevant = grade_chunks(query, chunks, threshold=0.5)

print(f"\nРелевантных: {len(relevant)}")
print(f"Нерелевантных: {len(irrelevant)}")
print(f"Нужен web fallback: {should_fallback(relevant)}")
