import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.retrieval.dence_retrieval import dense_search_multivector, dense_search_multivector_fusion

logging.getLogger("transformers").setLevel(logging.ERROR)
model = SentenceTransformer("ai-forever/ru-en-RoSBERTa", device="cuda")

client = QdrantClient(host="localhost", port=6333)

query = "что вызывает кислотные дожди"
collection = "crag_documents_mv"

for vec_name in ["body", "summary", "context"]:
    print(f"\n--- Поиск по '{vec_name}' ---")
    results = dense_search_multivector(
        query, model, client, collection, top_k=3, vector_name=vec_name
    )
    for r in results:
        print(f"score={r.score:.4f} | {r.payload['text'][:80]}")

print("\n--- Fusion (все три вектора) ---")
results_fusion = dense_search_multivector_fusion(query, model, client, collection, top_k=5)
for r in results_fusion:
    print(f"{r.payload['text'][:80]}")
