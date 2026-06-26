from langfuse import Langfuse

from src.config.setting import settings

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse_client


def trace_crag_run(query: str):
    """Контекстный менеджер для трейсинга одного прогона CRAG."""
    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="crag-pipeline",
        input={"query": query},
    )
    return trace


def log_span(trace, name: str, input_data: dict, output_data: dict, metadata: dict | None = None):
    """Логирует один шаг pipeline (retrieve, grade, rewrite, generate)."""
    trace.span(
        name=name,
        input=input_data,
        output=output_data,
        metadata=metadata or {},
    )
