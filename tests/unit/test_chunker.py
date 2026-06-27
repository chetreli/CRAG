from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


def test_chunk_documents_creates_chunks():
    docs = [Document(page_content="А " * 1000, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all("file_name" in c.metadata for c in chunks)


def test_chunk_documents_preserves_file_name():
    docs = [Document(page_content="Текст " * 50, metadata={"source": "/path/to/document.pdf"})]
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)

    assert all(c.metadata["file_name"] == "document.pdf" for c in chunks)


def test_chunk_documents_empty_input():
    chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
    assert chunks == []
