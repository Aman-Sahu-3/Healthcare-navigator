import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.api.deps import Services
from app.api.ratelimit import RateLimiter, client_ip, parse_rate
from app.core.config import Settings, get_settings
from app.ingestion.pipeline import build_store, ingest_path
from app.main import create_app
from app.rag.chain import RAGService

from .ingestion_helpers import HashingEmbeddings, make_pdf
from .test_ingestion import DM, HTN

ADMIN_KEY = "test-admin-key"


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        chroma_dir=tmp_path / "chroma",
        raw_docs_dir=tmp_path / "raw",
        chunk_size=300,
        chunk_overlap=50,
        min_similarity=0.1,
        high_conf_threshold=0.5,
        medium_conf_threshold=0.3,
        **overrides,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Real Chroma + fake embeddings + configurable fake LLM, wired the way the app is."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAW_DOCS_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("RATE_LIMIT", "5/minute")
    get_settings.cache_clear()

    settings = _settings(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "htn.pdf", [HTN], title="Hypertension Guideline")
    make_pdf(raw / "dm.pdf", [DM])
    emb = HashingEmbeddings()
    ingest_path(raw, settings, embeddings=emb)
    store = build_store(settings, emb)

    def make_client(llm=None, raise_server_exceptions=True):
        rag = RAGService(store, llm, settings) if llm else None
        services = Services(store=store, rag=rag, llm_error=None if llm else "no key")
        return TestClient(create_app(services), raise_server_exceptions=raise_server_exceptions)

    return make_client


def _llm(text="Thiazide diuretics are first-line [1]."):
    return FakeListChatModel(responses=[text])


# ---------- /query ----------


def test_query_returns_answer_citations_confidence(setup):
    with setup(_llm()) as c:
        r = c.post("/query", json={"question": "first-line treatment hypertension thiazide"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Thiazide")
    cit = body["citations"][0]
    assert cit["source"] == "htn.pdf" and cit["page"] == 1 and cit["title"]
    assert body["confidence"]["level"] in {"HIGH", "MEDIUM", "LOW"}
    assert set(body) >= {"latency_ms", "model", "retrieved_chunks", "insufficient_evidence"}
    assert r.headers["X-Request-ID"]


def test_query_top_k_respected(setup):
    with setup(_llm("ok [1]")) as c:
        r = c.post("/query", json={"question": "hypertension thiazide", "top_k": 1})
    assert r.json()["retrieved_chunks"] == 1


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"question": ""}, "question"),
        ({"question": "   a  "}, "question"),
        ({"question": "x" * 1001}, "question"),
        ({"question": "valid question", "top_k": 0}, "top_k"),
        ({"question": "valid question", "top_k": 50}, "top_k"),
        ({}, "question"),
    ],
)
def test_query_validation(setup, payload, field):
    with setup(_llm()) as c:
        r = c.post("/query", json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert err["details"][0]["field"] == field


def test_query_without_llm_is_503(setup):
    with setup(llm=None) as c:
        r = c.post("/query", json={"question": "hypertension thiazide"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "llm_not_configured"


def _openai_error(cls, status):
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return cls("upstream says no", response=httpx.Response(status, request=req), body=None)


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (_openai_error(openai.RateLimitError, 429), 503, "llm_rate_limited"),
        (_openai_error(openai.AuthenticationError, 401), 502, "llm_auth_error"),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
            504,
            "llm_timeout",
        ),
    ],
)
def test_upstream_llm_errors_are_mapped(setup, exc, status, code):
    class Failing(FakeListChatModel):
        def _call(self, *a, **k):
            raise exc

    with setup(Failing(responses=["x"])) as c:
        r = c.post("/query", json={"question": "hypertension thiazide"})
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    if code == "llm_rate_limited":
        assert r.headers["Retry-After"]


def test_unexpected_error_is_500_json(setup):
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("secret internal detail")

    with setup(Boom(responses=["x"]), raise_server_exceptions=False) as c:
        r = c.post("/query", json={"question": "hypertension thiazide"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text


def test_rate_limit(setup):
    with setup(_llm("ok [1]")) as c:
        codes = [c.post("/query", json={"question": "hypertension"}).status_code for _ in range(6)]
        assert codes[:5] == [200] * 5
        assert codes[5] == 429
        last = c.post("/query", json={"question": "hypertension"})
        assert last.json()["error"]["code"] == "rate_limited"
        assert int(last.headers["Retry-After"]) >= 1
        assert c.get("/health").status_code == 200  # other endpoints unaffected


# ---------- /sources & /health ----------


def test_sources(setup):
    with setup() as c:
        body = c.get("/sources").json()
    assert body["count"] == 2
    names = {d["source"] for d in body["documents"]}
    assert names == {"htn.pdf", "dm.pdf"}
    assert all(d["chunks"] > 0 for d in body["documents"])


def test_health_ok_when_ready(setup):
    with setup(_llm()) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["issues"] == []
    assert body["index"]["documents"] == 2


# ---------- /ingest ----------


def test_ingest_requires_key(setup):
    with setup() as c:
        assert c.post("/ingest").status_code == 401
        r = c.post("/ingest", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_ingest_disabled_without_admin_key(setup, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY")
    get_settings.cache_clear()
    with setup() as c:
        r = c.post("/ingest", headers={"X-API-Key": "anything"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "ingestion_disabled"


def test_ingest_rescan_skips_unchanged(setup):
    with setup() as c:
        r = c.post("/ingest", headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 200
        assert {f["status"] for f in r.json()["files"]} == {"skipped"}
        forced = c.post("/ingest", headers={"X-API-Key": ADMIN_KEY}, json={"force": True})
        assert {f["status"] for f in forced.json()["files"]} == {"ingested"}


def test_upload_pdf_then_query_it(setup, tmp_path):
    pdf = make_pdf(
        tmp_path / "asthma care.pdf",
        [
            "Inhaled corticosteroids are the cornerstone of persistent asthma maintenance "
            "therapy. Short-acting beta agonists are used for symptom relief."
        ],
    )
    with setup(_llm("ICS are first-line [1].")) as c:
        r = c.post(
            "/ingest/upload",
            headers={"X-API-Key": ADMIN_KEY},
            files={"file": ("asthma care.pdf", pdf.read_bytes(), "application/pdf")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["files"][0]["status"] == "ingested"
        assert c.get("/sources").json()["count"] == 3
        q = c.post("/query", json={"question": "inhaled corticosteroids asthma maintenance"})
    assert q.json()["citations"][0]["source"] == "asthma care.pdf"


@pytest.mark.parametrize(
    ("name", "content", "status"),
    [
        ("notes.txt", b"hello", 400),
        ("fake.pdf", b"not really a pdf", 400),
        ("../../evil.pdf", b"%PDF-1.4 tiny", 200),
    ],
)
def test_upload_validation(setup, tmp_path, name, content, status):
    with setup() as c:
        r = c.post(
            "/ingest/upload",
            headers={"X-API-Key": ADMIN_KEY},
            files={"file": (name, content, "application/pdf")},
        )
    assert r.status_code == status
    if status == 200:  # path traversal stripped: saved inside raw dir as evil.pdf
        assert r.json()["files"][0]["source"] == "evil.pdf"
        assert (tmp_path / "raw" / "evil.pdf").exists()
        assert not (tmp_path.parent / "evil.pdf").exists()


def test_upload_too_large(setup, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    with setup() as c:
        r = c.post(
            "/ingest/upload",
            headers={"X-API-Key": ADMIN_KEY},
            files={"file": ("big.pdf", b"%PDF" + b"0" * (1024 * 1024 + 10), "application/pdf")},
        )
    assert r.status_code == 413


def test_concurrent_ingest_rejected(setup):
    with setup() as c:
        services = c.app.state.services
        services.ingest_lock.acquire()
        try:
            r = c.post("/ingest", headers={"X-API-Key": ADMIN_KEY})
        finally:
            services.ingest_lock.release()
    assert r.status_code == 409 and r.json()["error"]["code"] == "ingest_in_progress"


# ---------- rate limiter unit tests ----------


def test_parse_rate():
    assert parse_rate("20/minute") == (20, 60)
    assert parse_rate("1/second") == (1, 1)


def test_sliding_window():
    now = [0.0]
    rl = RateLimiter(2, 10, clock=lambda: now[0])
    assert rl.hit("a")[0] and rl.hit("a")[0]
    allowed, retry = rl.hit("a")
    assert not allowed and retry >= 1
    assert rl.hit("b")[0]  # separate key
    now[0] = 10.5
    assert rl.hit("a")[0]  # window moved on


def test_client_ip_proxy_header():
    class Req:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}

        class client:  # noqa: N801
            host = "10.0.0.1"

    assert client_ip(Req, trust_proxy=True) == "203.0.113.7"
    assert client_ip(Req, trust_proxy=False) == "10.0.0.1"


def test_invalid_rate_limit_setting_rejected():
    with pytest.raises(ValueError):
        Settings(rate_limit="lots")


# ---------- startup ----------


def test_startup_builds_services(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(
        "app.ingestion.pipeline.get_embeddings", lambda *a, **k: HashingEmbeddings()
    )
    get_settings.cache_clear()
    with TestClient(create_app()) as c:  # no services injected -> lifespan loads them
        services = c.app.state.services
        assert services.store is not None
        assert services.rag is None and "OPENAI_API_KEY" in services.llm_error
        assert c.post("/query", json={"question": "anything here"}).status_code == 503

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.app.state.services.rag is not None
        assert c.get("/health").json()["llm_configured"] is True
