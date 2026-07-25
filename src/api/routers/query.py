import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.api.dependencies import get_crag_graph
from src.api.schemas import ChatRequest
from src.observability.tracing import get_langfuse

router = APIRouter()


def stream_crag_response(message: str, session_id: str, graph):
    """Генератор SSE событий с прогрессом выполнения."""
    langfuse = get_langfuse()
    trace = langfuse.trace(name="crag-pipeline", input={"query": message})

    def send_event(stage: str, message: str, progress: int) -> str:
        data = json.dumps({
            "stage": stage,
            "message": message,
            "progress": progress,
        })
        return f"data: {data}\n\n"

    yield send_event("retrieve", "🔍 Ищу релевантные фрагменты...", 10)

    config = {"configurable": {"thread_id": session_id}}
    state = {
        "query": message,
        "chunks": [], "relevant_chunks": [], "irrelevant_chunks": [],
        "rewritten_query": None, "rewrite_attempts": 0,
        "web_results": [], "used_fallback": False,
        "answer": "", "source": "",
        "messages": [HumanMessage(content=message)],
        "_trace_id": trace.id,
    }

    progress_map = {
        "retrieve": ("🔍 Найдены фрагменты, оцениваю релевантность...", 25),
        "grade":    ("✅ Оценка завершена, генерирую ответ...", 55),
        "rewrite":  ("✏️ Переформулирую запрос для лучшего поиска...", 40),
        "fallback": ("🌐 Ищу в интернете...", 70),
        "generate": ("💬 Генерирую ответ...", 85),
    }

    for step_output in graph.stream(state, config=config):
        node_name = list(step_output.keys())[0]
        state.update(step_output[node_name])

        if node_name in progress_map:
            msg, progress = progress_map[node_name]
            yield send_event(node_name, msg, progress)

    answer = state.get("answer", "")
    source = state.get("source", "")
    used_fallback = state.get("used_fallback", False)

    trace.update(output={"answer": answer, "source": source})
    langfuse.flush()

    final_data = json.dumps({
        "stage": "done",
        "message": "✓ Готово",
        "progress": 100,
        "answer": answer,
        "source": source,
        "used_fallback": used_fallback,
    })
    yield f"data: {final_data}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, graph=Depends(get_crag_graph)):
    return StreamingResponse(
        stream_crag_response(request.message, request.session_id, graph),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat", response_model=None)
def chat(request: ChatRequest, graph=Depends(get_crag_graph)):
    """Оставляем для обратной совместимости."""
    for _ in stream_crag_response(request.message, request.session_id, graph):
        pass
    # Используем stream endpoint напрямую через requests в Streamlit
    return {"detail": "Use /chat/stream instead"}
