import torch
from sentence_transformers import SentenceTransformer


def get_embedding_model(
    model_name: str = "ai-forever/ru-en-RoSBERTa",
    device: str | None = None,
) -> SentenceTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        model_name,
        device=device,
    )
    return model


if __name__ == "__main__":
    model = get_embedding_model()

    test_texts = [
        "Корректирующая RAG система для поиска информации",
        "Векторный поиск с использованием Qdrant",
    ]

    embeddings = model.encode(test_texts, normalize_embeddings=True)
    print(f"Устройство: {model.device}")
    print(f"Размер эмбеддинга: {embeddings.shape[1]}")
