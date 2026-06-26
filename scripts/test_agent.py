import logging
import time

from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.agent.graph import build_graph
from src.agent.memory import get_memory
from src.config.setting import settings

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

initial_state_template = {
    "chunks": [],
    "relevant_chunks": [],
    "irrelevant_chunks": [],
    "rewritten_query": None,
    "rewrite_attempts": 0,
    "web_results": [],
    "used_fallback": False,
    "answer": "",
    "source": "",
}

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Вопрос: {query}")
    print(f"{'=' * 60}")

    state = {
        **initial_state_template,
        "query": query,
        "messages": [HumanMessage(content=query)],
    }

    stage_times = {}
    total_start = time.time()
    last_time = total_start

    # stream вместо invoke — позволяет замерить время КАЖДОЙ ноды отдельно
    for step_output in graph.stream(state, config=config):
        now = time.time()
        # step_output это dict вида {"имя_ноды": {новое состояние...}}
        node_name = list(step_output.keys())[0]
        duration = now - last_time
        stage_times[node_name] = stage_times.get(node_name, 0) + duration
        print(f"  [{node_name}] заняло {duration:.2f} сек")
        last_time = now

        # Обновляем state последним результатом ноды чтобы вытащить финальный ответ
        state.update(step_output[node_name])

    total_duration = time.time() - total_start

    print("\n--- Тайминги по стадиям ---")
    for stage, duration in sorted(stage_times.items(), key=lambda x: -x[1]):
        pct = duration / total_duration * 100
        print(f"  {stage:20s} {duration:7.2f} сек  ({pct:5.1f}%)")

    print(f"\n  ОБЩЕЕ ВРЕМЯ:          {total_duration:7.2f} сек")

    print(f"\nИсточник: {state.get('source')}")
    print(f"Fallback: {state.get('used_fallback')}")
    print(f"Ответ:\n{state.get('answer')}")
