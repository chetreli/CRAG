from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.agent.nodes import make_nodes
from src.agent.state import AgentState
from src.config.setting import settings


def _route_after_grade(state: AgentState) -> str:
    """
    Решает что делать после grading:
    - достаточно релевантных → generate
    - мало но можно переписать → rewrite
    - исчерпали попытки → fallback
    """
    relevant = state.get("relevant_chunks", [])
    attempts = state.get("rewrite_attempts", 0)

    # Проверяем достаточность контекста
    if len(relevant) >= settings.min_relevant_chunks:
        avg_score = sum(c.score for c in relevant) / len(relevant)
        if avg_score >= settings.min_avg_score:
            return "generate"

    # Ещё можно попробовать переписать
    if attempts < settings.max_rewrite_attempts:
        return "rewrite"

    # Исчерпали попытки — web fallback
    return "fallback"


def build_graph(
    model: SentenceTransformer,
    client: QdrantClient,
) -> StateGraph:
    nodes = make_nodes(model, client)

    graph = StateGraph(AgentState)

    # Добавляем узлы
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Точка входа
    graph.set_entry_point("retrieve")

    # Рёбра
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite", "retrieve")  # rewrite → снова retrieve
    graph.add_edge("fallback", "generate")
    graph.add_edge("generate", END)

    # Условный роутинг после grade
    graph.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    return graph.compile()
