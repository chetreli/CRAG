from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.retrieval.hybrid import hybrid_search
from src.config.setting import settings

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

query = "что вызывает кислотные дожди"

print("--- Без re-ranking ---")
results = hybrid_search(query, model, client, settings.qdrant_collection, top_k=5)
for r in results:
    print(f"score={r.score:.4f} | {r.text[:800]}")

print("\n--- С re-ranking ---")
results_rerank = hybrid_search(query, model, client, settings.qdrant_collection, top_k=5, use_reranker=True)
for r in results_rerank:
    print(f"score={r.score:.4f} | {r.text[:800]}")