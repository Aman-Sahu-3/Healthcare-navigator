import pytest

from app.core.config import get_settings
from app.ingestion.__main__ import main

from .ingestion_helpers import HashingEmbeddings, make_pdf


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(
        raw / "htn.pdf",
        [
            "Thiazide diuretics are first-line for hypertension in adults. "
            "Lifestyle modification and sodium reduction are recommended for all patients."
        ],
    )
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAW_DOCS_DIR", str(raw))
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )
    get_settings.cache_clear()
    return raw


def test_cli_ingest_list_search_reset(cli_env, capsys):
    assert main([]) == 0
    assert "ingested 1" in capsys.readouterr().out

    assert main([]) == 0
    assert "skipped 1" in capsys.readouterr().out

    assert main(["--list"]) == 0
    assert "htn.pdf" in capsys.readouterr().out

    assert main(["--search", "first-line hypertension thiazide", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "htn.pdf" in out and "p.1" in out

    assert main(["--reset"]) == 0
    assert "ingested 1" in capsys.readouterr().out  # re-ingested after wipe, not skipped


def test_cli_missing_path(cli_env, tmp_path):
    assert main([str(tmp_path / "nope")]) == 2


def test_cli_failed_file_exit_code(cli_env):
    (cli_env / "bad.pdf").write_text("garbage")
    assert main([]) == 1
