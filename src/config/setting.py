from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"

    # Embeddings
    embedding_model: str = "ai-forever/ru-en-RoSBERTa"
    embedding_dimension: int = 1024
    embedding_device: str = "cuda"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "crag_documents"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_top_k: int = 10

    # Grader
    grader_threshold: float = 0.7
    min_relevant_chunks: int = 2
    min_avg_score: float = 0.6

    # Langfusion
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_host: str

    # Agent
    max_rewrite_attempts: int = 1

    class Config:
        env_file = ".env"


settings = Settings()
