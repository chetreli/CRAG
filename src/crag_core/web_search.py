from dataclasses import dataclass
from ddgs import DDGS


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def web_search(
    query: str,
    max_results: int = 5,
    region: str = "ru-ru",
) -> list[WebResult]:
    results = []
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(
                query,
                region=region,
                max_results=max_results,
                safesearch="moderate",
            )
            for r in raw:
                results.append(WebResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
    except Exception as e:
        print(f"Web search ошибка: {e}")

    return results


def format_web_results(results: list[WebResult]) -> str:
    """Форматирует результаты поиска в текст для контекста LLM."""
    if not results:
        return "Результаты веб-поиска недоступны."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.title}\n"
            f"URL: {r.url}\n"
            f"{r.snippet}"
        )
    return "\n\n".join(parts)