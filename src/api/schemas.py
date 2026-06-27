from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    source: str
    used_fallback: bool


class IngestRequest(BaseModel):
    collection_name: str = "crag_documents"


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str
