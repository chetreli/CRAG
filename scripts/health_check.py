import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU не найден, используется CPU")


def check_embeddings():
    model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    emb = model.encode(["тест"], normalize_embeddings=True)
    print(f"Эмбеддинги: размер {emb.shape[1]} ")


def check_qdrant():
    client = QdrantClient(host="localhost", port=6333)
    info = client.get_collections()
    print(f"Qdrant: подключен, коллекций {len(info.collections)} ")


if __name__ == "__main__":
    check_gpu()
    check_embeddings()
    check_qdrant()
