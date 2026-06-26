from src.config.setting import settings


def test_settings_has_required_fields():
    assert settings.qdrant_host
    assert settings.qdrant_port
    assert settings.embedding_model
    assert settings.llm_model
