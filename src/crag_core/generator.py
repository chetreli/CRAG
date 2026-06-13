from src.models.llm import generate
from src.retrieval.hybrid import RetrievedChunk
from src.crag_core.web_search import WebResult, format_web_results


GENERATOR_SYSTEM = """/no_think
Ты — helpful ассистент отвечающий на вопросы на основе предоставленного контекста.
Правила:
- Отвечай ТОЛЬКО на основе контекста
- Если контекст не содержит ответа — честно скажи об этом
- Отвечай на русском языке
- Будь точным и конкретным
- Не выдумывай факты которых нет в контексте"""

GENERATOR_PROMPT_LOCAL = """/no_think
Контекст из базы знаний:
{context}

Вопрос: {query}

Ответ:"""

GENERATOR_PROMPT_WEB = """/no_think
Контекст из веб-поиска:
{context}

Вопрос: {query}

Ответ:"""


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Источник: {chunk.file_name}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def generate_from_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    max_tokens: int = 1024,
) -> str:
    """Генерирует ответ на основе релевантных чанков из локальной базы."""
    context = _format_chunks(chunks)
    prompt = GENERATOR_PROMPT_LOCAL.format(
        context=context[:4000],
        query=query,
    )
    return generate(
        prompt=prompt,
        system=GENERATOR_SYSTEM,
        temperature=0.3,
        max_tokens=max_tokens,
    )


def generate_from_web(
    query: str,
    web_results: list[WebResult],
    max_tokens: int = 1024,
) -> str:
    """Генерирует ответ на основе результатов веб-поиска."""
    context = format_web_results(web_results)
    prompt = GENERATOR_PROMPT_WEB.format(
        context=context[:4000],
        query=query,
    )
    return generate(
        prompt=prompt,
        system=GENERATOR_SYSTEM,
        temperature=0.3,
        max_tokens=max_tokens,
    )


def generate_no_context(query: str) -> str:
    """Fallback когда нет ни локального контекста ни веб-результатов."""
    prompt = f"/no_think\nВопрос: {query}\n\nОтвет (без внешнего контекста):"
    return generate(
        prompt=prompt,
        system=GENERATOR_SYSTEM,
        temperature=0.3,
        max_tokens=1024,
    )