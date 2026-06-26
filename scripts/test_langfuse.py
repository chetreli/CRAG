import logging

from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.agent.graph import build_graph
from src.config.setting import settings
from src.observability.tracing import get_langfuse

logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

graph = build_graph(model, client)
langfuse = get_langfuse()

query = "что такое безопасность жизнедеятельности"
print(f"Запрос: {query}")

# Создаём один trace на весь pipeline
trace = langfuse.trace(name="crag-pipeline", input={"query": query})

result = graph.invoke(
    {
        "query": query,
        "chunks": [],
        "relevant_chunks": [],
        "irrelevant_chunks": [],
        "rewritten_query": None,
        "rewrite_attempts": 0,
        "web_results": [],
        "used_fallback": False,
        "answer": "",
        "source": "",
        "messages": [HumanMessage(content=query)],
        "_trace_id": trace.id,  # передаём id трейса в state
    }
)

trace.update(output={"answer": result["answer"], "source": result["source"]})

langfuse.flush()  # принудительная отправка перед завершением скрипта

print(f"\nОтвет: {result['answer']}")
print(f"\nПроверь трейсы тут: {settings.langfuse_host}")
print(f"Trace ID: {trace.id}")
