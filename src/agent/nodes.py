from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.agent.state import AgentState
from src.retrieval.hybrid import hybrid_search
from src.crag_core.grader import grade_chunks
from src.crag_core.query_rewriter import rewrite_query
from src.crag_core.web_search import web_search
from src.crag_core.generator import (
    generate_from_chunks,
    generate_from_web,
    generate_no_context,
)
from src.config.setting import settings


def make_nodes(model: SentenceTransformer, client: QdrantClient):
    """Фабрика нод — замыкание над model и client."""

    def retrieve(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        chunks = hybrid_search(
            query, model, client,
            settings.qdrant_collection,
            top_k=settings.retrieval_top_k,
        )
        print(f"[retrieve] найдено чанков: {len(chunks)}")
        return {**state, "chunks": chunks}

    def grade(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        relevant, irrelevant = grade_chunks(
            query,
            state["chunks"],
            threshold=settings.grader_threshold,
        )
        print(f"[grade] релевантных: {len(relevant)} / {len(state['chunks'])}")
        return {
            **state,
            "relevant_chunks": relevant,
            "irrelevant_chunks": irrelevant,
        }

    def rewrite(state: AgentState) -> AgentState:
        attempts = state.get("rewrite_attempts", 0)
        rewritten = rewrite_query(state["query"])
        print(f"[rewrite] попытка {attempts + 1}: {rewritten}")
        return {
            **state,
            "rewritten_query": rewritten,
            "rewrite_attempts": attempts + 1,
            "chunks": [],  # сбрасываем чанки для повторного retrieval
        }

    def fallback(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        print(f"[fallback] web search: {query}")
        results = web_search(query, max_results=5)
        print(f"[fallback] найдено результатов: {len(results)}")
        return {**state, "web_results": results, "used_fallback": True}

    def generate(state: AgentState) -> AgentState:
        query = state["query"]

        if state.get("relevant_chunks"):
            print("[generate] из локальной базы")
            answer = generate_from_chunks(query, state["relevant_chunks"][:5])
            source = "local"
        elif state.get("web_results"):
            print("[generate] из веб результатов")
            answer = generate_from_web(query, state["web_results"])
            source = "web"
        else:
            print("[generate] без контекста")
            answer = generate_no_context(query)
            source = "no_context"

        return {**state, "answer": answer, "source": source}

    return {
        "retrieve": retrieve,
        "grade": grade,
        "rewrite": rewrite,
        "fallback": fallback,
        "generate": generate,
    }