from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from src.retrieval.hybrid import RetrievedChunk
from src.crag_core.web_search import WebResult


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