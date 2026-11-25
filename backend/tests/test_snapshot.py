import io
import json
import tarfile

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.ingestion.__main__ import main as ingest_cli
from app.ingestion.pipeline import build_store, ingest_path
from app.ingestion.snapshot import SnapshotError, export_index, has_index, restore_index
from app.ingestion.vectorstore import count
from app.main import create_app

from .ingestion_helpers import HashingEmbeddings, make_pdf
from .test_ingestion import DM, HTN

MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture
def built_index(tmp_path):
    settings = Settings(chroma_dir=tmp_path / "src_chroma", chunk_size=300, chunk_overlap=50)
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "htn.pdf", [HTN])
    make_pdf(raw / "dm.pdf", [DM])
    emb = HashingEmbeddings()
    ingest_path(raw, settings, embeddings=emb)
    n = count(build_store(settings, emb))
    return settings, n


def _export(settings, tmp_path, n):
    return export_index(
        settings.chroma_dir,
        tmp_path / "out" / "index.tar.gz",
        {"embedding_model": MODEL, "chunks": n, "documents": 2},
    )


def test_export_then_restore_roundtrip(built_index, tmp_path):
    settings, n = built_index
    result = _export(settings, tmp_path, n)
    assert result.size_bytes > 0 and len(result.sha256) == 64
    assert result.manifest["chunks"] == n and "created_at" in result.manifest

    dest = tmp_path / "restored"
    manifest = restore_index(
        result.path.as_uri(), dest, expected_sha256=result.sha256, expected_model=MODEL
    )
    assert manifest["documents"] == 2 and has_index(dest)
    restored = build_store(settings.model_copy(update={"chroma_dir": dest}), HashingEmbeddings())
    assert count(restored) == n


def test_checksum_mismatch_rejected(built_index, tmp_path):
    settings, n = built_index
    result = _export(settings, tmp_path, n)
    with pytest.raises(SnapshotError, match="checksum"):
        restore_index(result.path.as_uri(), tmp_path / "r", expected_sha256="0" * 64)
    assert not has_index(tmp_path / "r")


def test_model_mismatch_rejected(built_index, tmp_path):
    settings, n = built_index
    result = _export(settings, tmp_path, n)
    with pytest.raises(SnapshotError, match="vectors would not match"):
        restore_index(result.path.as_uri(), tmp_path / "r", expected_model="other/model")


def test_export_without_index_fails(tmp_path):
    with pytest.raises(SnapshotError, match="No index"):
        export_index(tmp_path / "empty", tmp_path / "x.tar.gz", {})


@pytest.mark.parametrize("bad_name", ["../evil.txt", "/etc/evil", "chroma/../../evil"])
def test_path_traversal_rejected(tmp_path, bad_name):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in ((("manifest.json"), b"{}"), (bad_name, b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(SnapshotError, match="Unsafe path"):
        restore_index(archive.as_uri(), tmp_path / "r")
    assert not (tmp_path / "evil.txt").exists()


def test_symlink_rejected(tmp_path):
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("chroma/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(SnapshotError, match="Unsupported entry"):
        restore_index(archive.as_uri(), tmp_path / "r")


def test_missing_manifest_rejected(tmp_path):
    archive = tmp_path / "nomanifest.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("chroma/chroma.sqlite3")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(SnapshotError, match="manifest"):
        restore_index(archive.as_uri(), tmp_path / "r")


# ---------- CLI + startup ----------


@pytest.fixture
def patched_embeddings(monkeypatch):
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )


def test_cli_export(built_index, tmp_path, monkeypatch, capsys, patched_embeddings):
    settings, n = built_index
    monkeypatch.setenv("CHROMA_DIR", str(settings.chroma_dir))
    get_settings.cache_clear()
    out = tmp_path / "cli" / "index.tar.gz"
    assert ingest_cli(["--export", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"{n} chunks" in text and "sha256:" in text
    with tarfile.open(out) as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
    assert manifest["embedding_model"] == MODEL and manifest["chunks"] == n


def test_startup_restores_from_index_url(built_index, tmp_path, monkeypatch, patched_embeddings):
    settings, n = built_index
    result = _export(settings, tmp_path, n)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "server_chroma"))
    monkeypatch.setenv("INDEX_URL", result.path.as_uri())
    monkeypatch.setenv("INDEX_SHA256", result.sha256)
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        body = c.get("/health").json()
    assert body["index"] == {"documents": 2, "chunks": n}


def test_startup_survives_bad_index_url(tmp_path, monkeypatch, patched_embeddings):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "server_chroma"))
    monkeypatch.setenv("INDEX_URL", (tmp_path / "missing.tar.gz").as_uri())
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/health/live").status_code == 200
        assert any("empty" in i for i in c.get("/health").json()["issues"])


def test_docs_toggle(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    assert TestClient(create_app(services=object())).get("/docs").status_code == 404
    monkeypatch.setenv("DOCS_ENABLED", "true")
    get_settings.cache_clear()
    assert TestClient(create_app(services=object())).get("/docs").status_code == 200


def test_cors_origin_regex(monkeypatch):
    monkeypatch.setenv("CORS_ORIGIN_REGEX", r"https://healthnav-.*\.vercel\.app")
    get_settings.cache_clear()
    c = TestClient(create_app(services=object()))
    ok = c.options(
        "/query",
        headers={
            "Origin": "https://healthnav-git-main-abc.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert (
        ok.headers.get("access-control-allow-origin") == "https://healthnav-git-main-abc.vercel.app"
    )
    bad = c.options(
        "/query",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
