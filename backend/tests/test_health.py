def test_health_degraded_without_llm_or_documents(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["environment"] == "test"
    assert body["llm_configured"] is False
    assert body["index"] == {"documents": 0, "chunks": 0}
    assert any("LLM" in i for i in body["issues"])
    assert any("empty" in i for i in body["issues"])


def test_docs_available_outside_production(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["paths"]["/query"]


def test_request_id_header(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert generated.headers["X-Request-ID"] != "bad id with spaces!"


def test_unknown_route_uses_error_format(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    assert r.json()["request_id"]
