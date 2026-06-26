import re

from qdrant_client import QdrantClient
from qdrant_client.models import Record
from rank_bm25 import BM25Okapi

STOPWORDS_RU = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "то",
    "её",
    "мой",
    "тем",
    "чтобы",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
}


def tokenize_ru(text: str) -> list[str]:
    """
    Улучшенная токенизация для русского текста:
    - приводит к нижнему регистру
    - извлекает кириллические и латинские слова
    - сохраняет числа и аббревиатуры
    - удаляет стоп-слова
    - фильтрует короткие токены (< 2 символов)
    """
    text = text.lower()

    # Извлекаем слова (кириллица, латиница) и числа отдельно
    words = re.findall(r"[а-яёa-z]+", text)
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)

    tokens = words + numbers

    # Фильтрация
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPWORDS_RU]

    return tokens


def build_bm25_index(
    client: QdrantClient,
    collection_name: str,
) -> tuple[BM25Okapi, list[Record]]:
    """Загружает все точки из Qdrant и строит BM25 индекс в памяти."""
    all_points: list[Record] = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    corpus = [tokenize_ru(p.payload.get("text", "")) for p in all_points]
    bm25 = BM25Okapi(corpus)

    print(f"BM25 индекс построен: {len(all_points)} документов")
    return bm25, all_points


def sparse_search(
    query: str,
    bm25: BM25Okapi,
    all_points: list[Record],
    top_k: int = 10,
) -> list[tuple[Record, float]]:
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [(all_points[i], float(score)) for i, score in ranked]
