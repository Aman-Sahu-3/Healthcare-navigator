import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Tests never read the developer's real .env or API keys."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
