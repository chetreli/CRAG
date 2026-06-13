import re
import ollama
from src.config.setting import settings


def get_llm_client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_base_url)


def extract_response(content: str, thinking: str | None = None) -> str:
    """
    Qwen3 в thinking mode пишет ответ в поле thinking, а content оставляет пустым.
    Извлекаем финальный ответ из thinking — он идёт после последнего абзаца рассуждений.
    """
    if content and content.strip():
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if cleaned.strip():
            return cleaned.strip()

    if thinking and thinking.strip():
        # Ищем финальный ответ после ключевых фраз
        markers = [
            "Переформулированный запрос:",
            "Итого:",
            "Финальный вариант:",
            "Ответ:",
            "Получается:",
            "Стicking with",
        ]
        for marker in markers:
            idx = thinking.rfind(marker)
            if idx != -1:
                answer = thinking[idx + len(marker):].strip()
                # Берём первую строку после маркера
                first_line = answer.split("\n")[0].strip()
                if len(first_line) > 5:
                    return first_line

        # Fallback — последняя завершённая строка (не обрезанная)
        lines = [l.strip() for l in thinking.split("\n") if l.strip()]
        # Ищем последнюю строку которая заканчивается точкой или кавычкой
        for line in reversed(lines[:-1]):  # пропускаем последнюю — она обрезана
            if line.endswith((".", "»", '"', "знаний", "языке")):
                return line
        # Крайний fallback — предпоследняя строка
        if len(lines) >= 2:
            return lines[-2]

    return ""


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    client = get_llm_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(
            model=settings.llm_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "think": False,
            },
        )
        return extract_response(
            content=response.message.content,
            thinking=getattr(response.message, "thinking", None),
        )
    finally:
        client._client.close()