from src.models.llm import generate


SUMMARY_SYSTEM = """/no_think
Ты создаёшь очень короткое summary текстового фрагмента — одно предложение, отражающее главную тему.
Пиши по-русски, без вводных фраз."""

SUMMARY_PROMPT = """/no_think
Фрагмент: {text}

Главная тема одним предложением:"""


def summarize_chunk(text: str, max_chars: int = 800) -> str:
    """Генерирует краткое summary чанка для multi-vector индексации."""
    truncated = text[:max_chars]
    prompt = SUMMARY_PROMPT.format(text=truncated)
    summary = generate(
        prompt=prompt,
        system=SUMMARY_SYSTEM,
        temperature=0.2,
        max_tokens=128,
    )
    return summary or text[:100]  # fallback на начало текста