from sentence_transformers import SentenceTransformer

from src.models.llm import generate

HYDE_SYSTEM = """/no_think
Ты пишешь короткий гипотетический ответ на вопрос, как если бы он был фрагментом из учебника или энциклопедии.
Пиши по-русски, фактологично, в академическом стиле. Не более 3-4 предложений."""

HYDE_PROMPT = """/no_think
Вопрос: {query}

Гипотетический ответ:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Генерирует гипотетический документ-ответ для улучшения семантического поиска.
    Используется ТОЛЬКО для эмбеддинга — не показывается пользователю.
    """
    prompt = HYDE_PROMPT.format(query=query)
    hypothetical = generate(
        prompt=prompt,
        system=HYDE_SYSTEM,
        temperature=0.3,
        max_tokens=512,
    )
    return hypothetical or query  # fallback на оригинальный запрос


def embed_with_hyde(
    query: str,
    model: SentenceTransformer,
) -> list[float]:
    """
    Возвращает эмбеддинг гипотетического документа вместо эмбеддинга запроса.
    """
    hypothetical_doc = generate_hypothetical_document(query)
    embedding = model.encode(hypothetical_doc, normalize_embeddings=True)
    return embedding.tolist()
