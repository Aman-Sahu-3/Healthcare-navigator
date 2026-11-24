import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Tests never read the developer's real .env or API keys."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("ENVIRONMENT", "test")
    for var in ("OPENAI_API_KEY", "LLM_BASE_URL", "ADMIN_API_KEY", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(tmp_path):
    """App with an empty fake-embedding store and no LLM (never loads real models)."""
    from app.api.deps import Services
    from app.ingestion.vectorstore import get_vectorstore
    from app.main import create_app

    from .ingestion_helpers import HashingEmbeddings

    store = get_vectorstore(HashingEmbeddings(), tmp_path / "chroma", "test")
    services = Services(store=store, rag=None, llm_error="OPENAI_API_KEY is not set")
    with TestClient(create_app(services)) as c:
        yield c
