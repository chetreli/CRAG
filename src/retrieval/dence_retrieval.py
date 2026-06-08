from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



def dense_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 10,
) -> list:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    responce = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return responce.points