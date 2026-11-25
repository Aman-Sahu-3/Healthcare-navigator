import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.environment == "test"
    assert s.retrieval_top_k == 5
    assert s.medium_conf_threshold < s.high_conf_threshold
    assert s.openai_api_key is None


def test_cors_accepts_comma_separated(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com")
    assert Settings().cors_origins == ["http://a.com", "http://b.com"]


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    with pytest.raises(ValidationError):
        Settings()


def test_confidence_thresholds_ordered(monkeypatch):
    monkeypatch.setenv("MEDIUM_CONF_THRESHOLD", "0.9")
    with pytest.raises(ValidationError):
        Settings()


def test_api_key_is_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    s = Settings()
    assert "sk-test-123" not in repr(s)
    assert s.openai_api_key.get_secret_value() == "sk-test-123"


def test_empty_env_values_mean_unset(monkeypatch):
    for key in (
        "EMBEDDING_THREADS",
        "DOCS_ENABLED",
        "INDEX_URL",
        "CORS_ORIGIN_REGEX",
        "LLM_BASE_URL",
        "ADMIN_API_KEY",
    ):
        monkeypatch.setenv(key, "")
    s = Settings()
    assert s.embedding_threads is None and s.docs_enabled is None
    assert s.index_url is None and s.cors_origin_regex is None
    assert s.llm_base_url is None and s.admin_api_key is None
