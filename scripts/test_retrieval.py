from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.retrieval.hybrid import hybrid_search

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

query = "Изменится лик земли"
results = hybrid_search(query, model, client, settings.qdrant_collection, top_k=10)

for i, chunk in enumerate(results):
    print(f"\n[{i + 1}] score={chunk.score:.4f} | {chunk.file_name}")
    print(chunk.text[:200])
