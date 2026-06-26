from unittest.mock import patch

from src.crag_core.grader import grade_chunks_batch


@patch("src.crag_core.grader.generate")
def test_grade_chunks_batch_parses_response(mock_generate, mock_retrieved_chunk):
    mock_generate.return_value = '[{"id": 0, "score": 0.9}]'

    relevant, irrelevant = grade_chunks_batch(
        query="тест",
        chunks=[mock_retrieved_chunk],
        threshold=0.5,
    )

    assert len(relevant) == 1
    assert relevant[0].score == 0.9


@patch("src.crag_core.grader.generate")
def test_grade_chunks_batch_handles_malformed_json(mock_generate, mock_retrieved_chunk):
    mock_generate.return_value = "это не json вообще"

    relevant, irrelevant = grade_chunks_batch(
        query="тест",
        chunks=[mock_retrieved_chunk],
        threshold=0.5,
    )

    assert len(irrelevant) == 1
    assert relevant == []


def test_grade_chunks_batch_empty_list():
    relevant, irrelevant = grade_chunks_batch("тест", [], threshold=0.5)
    assert relevant == []
    assert irrelevant == []
