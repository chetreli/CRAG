import time

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.agent.state import AgentState
from src.config.setting import settings
from src.crag_core.generator import (
    generate_from_chunks,
    generate_from_web,
    generate_no_context,
)
from src.crag_core.grader import grade_chunks_batch
from src.crag_core.query_rewriter import rewrite_query
from src.crag_core.web_search import web_search
from src.observability.prometheus import (
    crag_fallback_total,
    crag_generate_duration,
    crag_grade_duration,
    crag_relevant_chunks_ratio,
    crag_requests_total,
    crag_retrieve_duration,
)
from src.observability.tracing import get_langfuse
from src.retrieval.hybrid import hybrid_search


def make_nodes(model: SentenceTransformer, client: QdrantClient):
    langfuse = get_langfuse()

    def retrieve(state: AgentState) -> AgentState:
        start = time.time()
        query = state.get("rewritten_query") or state["query"]

        chunks = hybrid_search(
            query,
            model,
            client,
            settings.qdrant_collection,
            top_k=settings.retrieval_top_k,
        )
        duration = time.time() - start
        crag_retrieve_duration.observe(duration)

        trace_id = state.get("_trace_id")
        if trace_id:
            langfuse.span(
                trace_id=trace_id,
                name="retrieve",
                input={"query": query},
                output={"chunks_count": len(chunks)},
                metadata={"duration_sec": duration},
            )

        print(f"[retrieve] найдено чанков: {len(chunks)}")
        return {**state, "chunks": chunks}

    def grade(state: AgentState) -> AgentState:
        start = time.time()
        query = state.get("rewritten_query") or state["query"]

        relevant, irrelevant = grade_chunks_batch(
            query,
            state["chunks"],
            threshold=settings.grader_threshold,
        )
        duration = time.time() - start
        crag_grade_duration.observe(duration)

        if state["chunks"]:
            crag_relevant_chunks_ratio.set(len(relevant) / len(state["chunks"]))

        avg_score = sum(c.score for c in relevant) / len(relevant) if relevant else 0.0

        trace_id = state.get("_trace_id")
        if trace_id:
            langfuse.span(
                trace_id=trace_id,
                name="grade",
                input={"query": query, "chunks_count": len(state["chunks"])},
                output={
                    "relevant_count": len(relevant),
                    "irrelevant_count": len(irrelevant),
                    "avg_score": avg_score,
                },
                metadata={"duration_sec": duration},
            )

        print(f"[grade] релевантных: {len(relevant)} / {len(state['chunks'])}")
        return {**state, "relevant_chunks": relevant, "irrelevant_chunks": irrelevant}

    def rewrite(state: AgentState) -> AgentState:
        attempts = state.get("rewrite_attempts", 0)
        rewritten = rewrite_query(state["query"])

        trace_id = state.get("_trace_id")
        if trace_id:
            langfuse.span(
                trace_id=trace_id,
                name="rewrite",
                input={"original_query": state["query"], "attempt": attempts + 1},
                output={"rewritten_query": rewritten},
            )

        print(f"[rewrite] попытка {attempts + 1}: {rewritten}")
        return {
            **state,
            "rewritten_query": rewritten,
            "rewrite_attempts": attempts + 1,
            "chunks": [],
        }

    def fallback(state: AgentState) -> AgentState:
        query = state.get("rewritten_query") or state["query"]
        crag_fallback_total.inc()

        results = web_search(query, max_results=5)

        trace_id = state.get("_trace_id")
        if trace_id:
            langfuse.span(
                trace_id=trace_id,
                name="fallback_web_search",
                input={"query": query},
                output={"results_count": len(results)},
            )

        print(f"[fallback] найдено результатов: {len(results)}")
        return {**state, "web_results": results, "used_fallback": True}

    def generate(state: AgentState) -> AgentState:
        start = time.time()
        query = state["query"]

        if state.get("relevant_chunks"):
            answer = generate_from_chunks(query, state["relevant_chunks"][:5])
            source = "local"
        elif state.get("web_results"):
            answer = generate_from_web(query, state["web_results"])
            source = "web"
        else:
            answer = generate_no_context(query)
            source = "no_context"

        duration = time.time() - start
        crag_generate_duration.observe(duration)
        crag_requests_total.labels(source=source).inc()

        trace_id = state.get("_trace_id")
        if trace_id:
            langfuse.span(
                trace_id=trace_id,
                name="generate",
                input={"query": query, "source": source},
                output={"answer": answer},
                metadata={"duration_sec": duration},
            )

        return {**state, "answer": answer, "source": source}

    return {
        "retrieve": retrieve,
        "grade": grade,
        "rewrite": rewrite,
        "fallback": fallback,
        "generate": generate,
    }
