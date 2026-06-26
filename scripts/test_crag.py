import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.crag_core.pipeline import run_crag_pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

queries = [
    # "что такое безопасность жизнедеятельности",
    # "какие бывают экосистемы",
    "последние новости о климате 2025",
]

for query in queries:
    result = run_crag_pipeline(query, model, client)
    print(f"\nИсточник: {result.source}")
    print(f"Fallback: {result.used_fallback}")
    print(f"Ответ:\n{result.answer}")
    print("=" * 50)
