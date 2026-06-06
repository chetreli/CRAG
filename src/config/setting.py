from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    
    # Embeddings
    embedding_model: str = "ai-forever/ru-en-RoSBERTa"
    embedding_dimension: int = 1024  # размерность RoSBERTa
    embedding_device: str = "cuda"   # у тебя RTX 5060 Ti
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "crag_documents"
    
    class Config:
        env_file = ".env"

settings = Settings()