import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.setting import settings
from src.eval.ragas_eval import run_ragas_evaluation

logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

result, df = run_ragas_evaluation(model, client)

# Сохраняем результаты в CSV для дальнейшего анализа
df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")
print("\nРезультаты сохранены в ragas_results.csv")
