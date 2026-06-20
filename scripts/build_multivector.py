from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.ingestion.loaders import load_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_index_multivector
from src.config.setting import settings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

documents = load_directory(Path("data/raw"))
chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

embed_and_index_multivector(
    chunks, model, client,
    collection_name="crag_documents_mv",
)