from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage

from src.api.dependencies import get_crag_graph
from src.api.schemas import ChatRequest, ChatResponse
from src.observability.tracing import get_langfuse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, graph=Depends(get_crag_graph)):
    langfuse = get_langfuse()
    trace = langfuse.trace(name="crag-pipeline", input={"query": request.message})

    config = {"configurable": {"thread_id": request.session_id}}

    result = graph.invoke(
        {
            "query": request.message,
            "chunks": [], "relevant_chunks": [], "irrelevant_chunks": [],
            "rewritten_query": None, "rewrite_attempts": 0,
            "web_results": [], "used_fallback": False,
            "answer": "", "source": "",
            "messages": [HumanMessage(content=request.message)],
            "_trace_id": trace.id,
        },
        config=config,
    )

    trace.update(output={"answer": result["answer"], "source": result["source"]})
    langfuse.flush()

    return ChatResponse(
        answer=result["answer"],
        source=result["source"],
        used_fallback=result["used_fallback"],
    )
