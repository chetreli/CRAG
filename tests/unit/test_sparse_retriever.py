from src.retrieval.sparse_retrieval import tokenize_ru


def test_tokenize_basic():
    tokens = tokenize_ru("Безопасность жизнедеятельности важна")
    assert "безопасность" in tokens
    assert "жизнедеятельности" in tokens


def test_tokenize_removes_stopwords():
    tokens = tokenize_ru("это не имеет значения для меня")
    assert "это" not in tokens
    assert "не" not in tokens


def test_tokenize_filters_short_tokens():
    tokens = tokenize_ru("я и ты в дом")
    assert "я" not in tokens
    assert "и" not in tokens


def test_tokenize_preserves_numbers():
    tokens = tokenize_ru("ГОСТ 2024 год")
    assert "2024" in tokens
