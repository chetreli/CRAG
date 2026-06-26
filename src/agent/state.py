from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.crag_core.web_search import WebResult
from src.retrieval.hybrid import RetrievedChunk


class AgentState(TypedDict):
    # Входные данные
    query: str
    # Retrieval
    chunks: list[RetrievedChunk]
    # Grading
    relevant_chunks: list[RetrievedChunk]
    irrelevant_chunks: list[RetrievedChunk]
    # Query rewriting
    rewritten_query: str | None
    rewrite_attempts: int
    # Web search
    web_results: list[WebResult]
    used_fallback: bool
    # Финальный ответ
    answer: str
    source: str  # "local" | "web" | "no_context"
    # История сообщений для memory
    messages: Annotated[list, add_messages]
    # идентификатор для tracing
    _trace_id: str | None
