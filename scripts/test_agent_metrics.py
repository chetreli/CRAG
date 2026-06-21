import logging
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.observability.prometheus import start_metrics_server
from src.config.setting import settings
logging.getLogger("transformers").setLevel(logging.ERROR)

start_metrics_server(port=8001)
print("Метрики на http://localhost:8001/metrics")

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

graph = build_graph(model, client)

queries = [
    "что такое безопасность жизнедеятельности",
    "что вызывает кислотные дожди",
    "последние новости о климате 2025",
]

for query in queries:
    print(f"\n{'='*50}\nЗапрос: {query}")
    result = graph.invoke({
        "query": query,
        "chunks": [], "relevant_chunks": [], "irrelevant_chunks": [],
        "rewritten_query": None, "rewrite_attempts": 0,
        "web_results": [], "used_fallback": False,
        "answer": "", "source": "",
        "messages": [HumanMessage(content=query)],
        "_trace_id": None,
    })
    print(f"Ответ: {result['answer'][:200]}")

print("\nГотово! Метрики наполнены. Процесс остаётся активным для Prometheus scrape.")
print("Ctrl+C для выхода.")


while True:
    time.sleep(1)