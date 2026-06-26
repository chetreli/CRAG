import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.crag_core.hyde import generate_hypothetical_document
from src.retrieval.hybrid import hybrid_search

logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

query = "что вызывает кислотные дожди"

print(f"Запрос: {query}")
print("\nГипотетический документ:")
print(generate_hypothetical_document(query))

print("\n--- Без HyDE ---")
results_normal = hybrid_search(
    query, model, client, settings.qdrant_collection, top_k=5, use_hyde=False
)
for r in results_normal:
    print(f"score={r.score:.3f} | {r.file_name}")

print("\n--- С HyDE ---")
results_hyde = hybrid_search(
    query, model, client, settings.qdrant_collection, top_k=5, use_hyde=True
)
for r in results_hyde:
    print(f"score={r.score:.3f} | {r.file_name}")
