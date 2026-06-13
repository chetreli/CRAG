from src.models.llm import generate
import json

REWRITE_SYSTEM = """/no_think
Переформулируй поисковый запрос для векторной базы знаний.
Отвечай ТОЛЬКО валидным JSON без лишнего текста.
Формат: {"query": "переформулированный запрос на русском"}"""

REWRITE_PROMPT = """/no_think
{{"input": "{query}"}}"""


REWRITE_MULTIPLE_SYSTEM = """/no_think
Ты переформулируешь поисковые запросы для векторной базы знаний.
Отвечай СТРОГО в формате JSON: {{"queries": ["вариант1", "вариант2", "вариант3"]}}
Никакого текста кроме JSON."""


def _parse_json_response(text: str, key: str) -> str | list:
    """Извлекает JSON из ответа модели, даже если вокруг есть мусор."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return text.strip() if key == "query" else []
    
    try:
        data = json.loads(text[start:end])
        return data.get(key, text.strip() if key == "query" else [])
    except json.JSONDecodeError:
        return text.strip() if key == "query" else []
    
def rewrite_query(query: str) -> str:
    prompt = REWRITE_PROMPT.format(query=query)
    response = generate(
        prompt=prompt,
        system=REWRITE_SYSTEM,
        temperature=0.3,
        max_tokens=1024,
    )
    # Если content пришёл — парсим JSON
    start = response.find("{")
    end = response.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            data = json.loads(response[start:end])
            result = data.get("query", "").strip()
            if result:
                return result
        except json.JSONDecodeError:
            pass
    return query  # fallback — возвращаем оригинал


def rewrite_query_multiple(query: str, n: int = 3) -> list[str]:
    system = REWRITE_MULTIPLE_SYSTEM.format()
    prompt = f"Запрос: {query}\n\nСгенерируй {n} варианта переформулировки. Верни JSON:"
    
    response = generate(
        prompt=prompt,
        system=system,
        temperature=0.7,
        max_tokens=512,
    )
    result = _parse_json_response(response, "queries")
    return result[:n] if isinstance(result, list) and result else [query]