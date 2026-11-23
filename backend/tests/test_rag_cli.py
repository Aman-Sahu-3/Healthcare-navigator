import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import get_settings
from app.rag.__main__ import main

from .ingestion_helpers import HashingEmbeddings, make_pdf
from .test_ingestion import HTN


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    from app.ingestion.pipeline import ingest_path

    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "htn.pdf", [HTN])
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("MIN_SIMILARITY", "0.1")
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )
    get_settings.cache_clear()
    ingest_path(raw, get_settings())


def _fake_llm(monkeypatch, text):
    monkeypatch.setattr("app.rag.llm.get_llm", lambda s: FakeListChatModel(responses=[text]))


def test_cli_pretty_output(cli_env, monkeypatch, capsys):
    _fake_llm(monkeypatch, "Thiazide diuretics are first-line [1].")
    assert main(["first-line treatment hypertension thiazide"]) == 0
    out = capsys.readouterr().out
    assert "Confidence:" in out and "Sources:" in out and "htn.pdf" in out


def test_cli_json_output(cli_env, monkeypatch, capsys):
    _fake_llm(monkeypatch, "Thiazide diuretics are first-line [1].")
    assert main(["first-line hypertension thiazide", "--json", "-k", "2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) >= {"answer", "citations", "confidence", "latency_ms"}
    assert data["citations"][0]["source"] == "htn.pdf"


def test_cli_without_api_key(cli_env, capsys):
    assert main(["anything"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
