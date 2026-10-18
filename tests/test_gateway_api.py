"""End-to-end API tests for VeriRoute FastAPI Gateway."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_chat_completion_flow():
    client.post("/api/v1/reset?warm_start=true")

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 1. First request (Cache Miss -> Routed via LinUCB Bandit)
    r1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "How do I enable CORS in FastAPI?"}],
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["veriroute"]["cache"]["cache_hit"] is False
    assert d1["veriroute"]["router"]["routing_policy"] == "linucb_bandit"

    # 2. Paraphrase request (Verified Stage-2 NLI Cache Hit -> $0 cost)
    r2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is the way to turn on cross-origin in FastAPI?"}],
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["veriroute"]["cache"]["cache_hit"] is True
    assert d2["veriroute"]["cost_usd"] == 0.0

    # 3. Adversarial antonym request ("disable" vs "enable" -> Blocked by Stage-2 NLI!)
    r3 = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "How do I disable CORS in FastAPI?"}],
        },
    )
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["veriroute"]["cache"]["cache_hit"] is False
    assert d3["veriroute"]["cache"]["false_positive_prevented"] is True


def test_streaming_and_prometheus_metrics():
    r_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "What does HTTP stand for?"}],
        },
    )
    assert r_stream.status_code == 200
    assert "data: [DONE]" in r_stream.text

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "veriroute_requests_total" in r_metrics.text
