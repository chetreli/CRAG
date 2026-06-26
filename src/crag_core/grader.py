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


GRADER_BATCH_SYSTEM = """/no_think
Ты оцениваешь релевантность нескольких фрагментов текста к одному поисковому запросу.
Отвечай ТОЛЬКО валидным JSON-массивом без лишнего текста, без пояснений.
Формат: [{"id": 0, "score": 0.0}, {"id": 1, "score": 0.0}, ...]
Где score — число от 0.0 до 1.0:
- 0.0-0.3: фрагмент не релевантен запросу
- 0.4-0.6: фрагмент частично релевантен
- 0.7-1.0: фрагмент хорошо отвечает на запрос
Верни оценку для КАЖДОГО фрагмента из списка, в том же порядке id."""

GRADER_BATCH_PROMPT = """/no_think
Запрос: "{query}"

Фрагменты:
{fragments}

Верни JSON-массив с оценками для всех фрагментов:"""


def grade_chunk(query: str, chunk: RetrievedChunk) -> tuple[float, str]:
    """
    Оценивает релевантность ОДНОГО чанка к запросу (старая версия, 1 LLM-вызов на чанк).
    Оставлена для обратной совместимости / точечного дебага.
    """
    fragment = chunk.text[:500].replace('"', "'").replace("\n", " ")
    query_clean = query.replace('"', "'")

    prompt = GRADER_PROMPT.format(query=query_clean, fragment=fragment)
    response = generate(
        prompt=prompt,
        system=GRADER_SYSTEM,
        temperature=0.0,
        max_tokens=1024,
    )

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
    СТАРАЯ версия — оценивает чанки ПО ОДНОМУ (N LLM-вызовов).
    Медленно: ~9 сек на чанк. Оставлена для сравнения/дебага.
    """
    relevant = []
    irrelevant = []

    for chunk in chunks:
        score, reason = grade_chunk(query, chunk)
        chunk.score = score
        print(f"  score={score:.2f} | {chunk.file_name} | {reason[:256]}")

        if score >= threshold:
            relevant.append(chunk)
        else:
            irrelevant.append(chunk)

    return relevant, irrelevant


def grade_chunks_batch(
    query: str,
    chunks: list[RetrievedChunk],
    threshold: float = 0.5,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    НОВАЯ версия — оценивает ВСЕ чанки ОДНИМ LLM-вызовом.
    Быстро: вместо N вызовов по ~9 сек, один вызов на все чанки сразу.
    """
    if not chunks:
        return [], []

    fragments_text = "\n\n".join(
        f"[{i}] {c.text[:400].replace(chr(34), chr(39)).replace(chr(10), ' ')}"
        for i, c in enumerate(chunks)
    )
    query_clean = query.replace('"', "'")

    prompt = GRADER_BATCH_PROMPT.format(query=query_clean, fragments=fragments_text)
    response = generate(
        prompt=prompt,
        system=GRADER_BATCH_SYSTEM,
        temperature=0.0,
        max_tokens=2048,  # больше токенов, но всего ОДИН вызов вместо N
    )

    start = response.find("[")
    end = response.rfind("]") + 1

    scores_map: dict[int, float] = {}
    if start != -1 and end > 0:
        try:
            data = json.loads(response[start:end])
            for item in data:
                scores_map[int(item["id"])] = float(item["score"])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            print(f"  [grader_batch] не удалось распарсить ответ: {response[:200]}")

    relevant = []
    irrelevant = []

    for i, chunk in enumerate(chunks):
        score = max(0.0, min(1.0, scores_map.get(i, 0.0)))
        chunk.score = score
        print(f"  score={score:.2f} | {chunk.file_name}")

        if score >= threshold:
            relevant.append(chunk)
        else:
            irrelevant.append(chunk)

    return relevant, irrelevant


def should_fallback(
    relevant: list[RetrievedChunk],
    min_relevant: int = 2,
) -> bool:
    return len(relevant) < min_relevant
