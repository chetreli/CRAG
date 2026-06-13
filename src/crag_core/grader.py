import json
from src.models.llm import generate
from src.retrieval.hybrid import RetrievedChunk


GRADER_SYSTEM = """/no_think
Ты оцениваешь релевантность фрагмента текста к поисковому запросу.
Отвечай ТОЛЬКО валидным JSON без лишнего текста.
Формат: {"score": 0.0, "reason": "краткое обоснование"}
Где score — число от 0.0 до 1.0:
- 0.0-0.3: фрагмент не релевантен запросу
- 0.4-0.6: фрагмент частично релевантен
- 0.7-1.0: фрагмент хорошо отвечает на запрос"""

GRADER_PROMPT = """/no_think
{{"query": "{query}", "fragment": "{fragment}"}}
Оцени релевантность фрагмента к запросу:"""


def grade_chunk(query: str, chunk: RetrievedChunk) -> tuple[float, str]:
    """
    Оценивает релевантность одного чанка к запросу.
    Возвращает (score, reason).
    """
    # Обрезаем чанк до 500 символов чтобы не раздувать промпт
    fragment = chunk.text[:500].replace('"', "'").replace("\n", " ")
    query_clean = query.replace('"', "'")

    prompt = GRADER_PROMPT.format(query=query_clean, fragment=fragment)
    response = generate(
        prompt=prompt,
        system=GRADER_SYSTEM,
        temperature=0.0,  # детерминированно — всегда один и тот же ответ
        max_tokens=1024,
    )

    # Парсим JSON ответ
    start = response.find("{")
    end = response.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            data = json.loads(response[start:end])
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")
            return max(0.0, min(1.0, score)), reason
        except (json.JSONDecodeError, ValueError):
            pass

    return 0.0, "не удалось распарсить ответ"


def grade_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    threshold: float = 0.5,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    Оценивает все чанки и разделяет на релевантные и нерелевантные.
    Возвращает (relevant, irrelevant).
    """
    relevant = []
    irrelevant = []

    for chunk in chunks:
        score, reason = grade_chunk(query, chunk)
        chunk.score = score  # перезаписываем retrieval score на grader score
        print(f"  score={score:.2f} | {chunk.file_name} | {reason[:256]}")

        if score >= threshold:
            relevant.append(chunk)
        else:
            irrelevant.append(chunk)

    return relevant, irrelevant


def should_fallback(
    relevant: list[RetrievedChunk],
    min_relevant: int = 2,
) -> bool:
    """
    Решает нужен ли web search fallback.
    True если релевантных чанков меньше min_relevant.
    """
    return len(relevant) < min_relevant