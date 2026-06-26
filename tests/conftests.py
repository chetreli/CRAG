from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_retrieved_chunk():
    from src.retrieval.hybrid import RetrievedChunk

    return RetrievedChunk(
        text="Тестовый текст чанка про экологию",
        source="test.pdf",
        file_name="test.pdf",
        chunk_id=0,
        score=0.0,
    )


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    return client
