from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph
from src.agent.memory import get_memory
from src.config.setting import settings

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

graph = build_graph(model, client)
memory = get_memory()

config = {"configurable": {"thread_id": "test-session-1"}}

queries = [
    "что такое безопасность жизнедеятельности",
    "последние новости о климате 2025",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Вопрос: {query}")

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
        },
        config=config,
    )

    print(f"\nИсточник: {result['source']}")
    print(f"Fallback: {result['used_fallback']}")
    print(f"Ответ:\n{result['answer']}")