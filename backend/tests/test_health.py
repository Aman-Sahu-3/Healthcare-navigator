def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["llm_configured"] is False


def test_docs_available_outside_production(client):
    assert client.get("/docs").status_code == 200
