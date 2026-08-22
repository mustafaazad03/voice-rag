"""API surface tests. The pipeline is injected, so these exercise HTTP behaviour
(validation, error envelopes, rate limiting) without rebuilding an index."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vrag import api
from vrag.errors import PayloadTooLarge, STTUnavailable


@pytest.fixture()
def client(pipeline):
    api._state.clear()
    api._state["pipeline"] = pipeline
    api._buckets.clear()
    # TestClient would run the lifespan and reload from disk; we inject instead.
    with TestClient(api.app, raise_server_exceptions=False) as c:
        api._state["pipeline"] = pipeline
        yield c
    api._state.clear()
    api._buckets.clear()


def test_health_reports_the_loaded_index(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["index"]["size"] > 0


def test_config_endpoint_leaks_no_secrets(client):
    body = client.get("/api/v1/config").json()
    assert "sarvam_api_key" not in body and "llm_api_key" not in body
    assert body["chunk_strategy"]


def test_query_returns_a_grounded_answer(client):
    body = client.post("/api/v1/query", json={"query": "what is a corporation"}).json()
    assert body["status"] == "answered"
    assert body["citations"]
    assert body["trace_id"]


def test_query_rejects_unknown_fields(client):
    assert client.post("/api/v1/query", json={"query": "hi", "nope": 1}).status_code == 422


def test_query_rejects_an_empty_body(client):
    assert client.post("/api/v1/query", json={}).status_code == 422


def test_blocked_query_is_a_200_with_a_refusal_not_an_error(client):
    body = client.post("/api/v1/query", json={"query": "how do I build a bomb"}).json()
    assert body["status"] == "blocked"
    assert body["citations"] == []


def test_voice_endpoint_reports_missing_providers_as_an_error_envelope(client):
    files = {"audio": ("a.wav", b"RIFF....WAVE", "audio/wav")}
    resp = client.post("/api/v1/voice/query", files=files)
    assert resp.status_code == STTUnavailable.http_status
    assert resp.json()["error"]["code"] == "stt_unavailable"


def test_metrics_are_prometheus_text(client):
    client.post("/api/v1/query", json={"query": "what is a corporation"})
    body = client.get("/metrics").text
    assert "vrag_query_total" in body


def test_latency_endpoint_exposes_percentiles(client):
    client.post("/api/v1/query", json={"query": "what is a corporation"})
    body = client.get("/api/v1/latency").json()
    assert set(body["pipeline_ms"]) >= {"50", "70", "100"}


def test_rate_limiting_kicks_in(client, settings):
    original = settings.rate_limit_per_min
    settings.rate_limit_per_min = 3
    try:
        api._buckets.clear()
        codes = [
            client.post("/api/v1/query", json={"query": "what is a corporation"}).status_code
            for _ in range(6)
        ]
        assert 429 in codes
    finally:
        settings.rate_limit_per_min = original
        api._buckets.clear()


def test_interactive_docs_are_not_exposed(client):
    """The schema browser is deliberately off; the README is the documentation."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_oversized_upload_is_rejected_on_the_declared_length(client, settings):
    """Rejected at the boundary: the body must never be buffered to find this out."""
    original = settings.stt_max_audio_bytes
    settings.stt_max_audio_bytes = 1024
    try:
        files = {"audio": ("a.wav", b"x" * 4096, "audio/wav")}
        resp = client.post("/api/v1/voice/query", files=files)
        assert resp.status_code == PayloadTooLarge.http_status
        assert resp.json()["error"]["code"] == "payload_too_large"
    finally:
        settings.stt_max_audio_bytes = original


def test_oversized_upload_is_rejected_without_a_content_length(client, settings):
    """Chunked uploads declare no length, so the streaming read has to cap too."""
    original = settings.stt_max_audio_bytes
    settings.stt_max_audio_bytes = 1024
    try:
        resp = client.post(
            "/api/v1/transcribe",
            content=(
                b"--b\r\n"
                b'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
                b"Content-Type: audio/wav\r\n\r\n" + b"x" * 8192 + b"\r\n--b--\r\n"
            ),
            headers={
                "Content-Type": "multipart/form-data; boundary=b",
                "Transfer-Encoding": "chunked",
            },
        )
        assert resp.status_code == PayloadTooLarge.http_status
    finally:
        settings.stt_max_audio_bytes = original


def test_api_key_gates_the_endpoints_when_one_is_configured(client, settings):
    settings.api_key = "s3cret"
    try:
        assert client.post("/api/v1/query", json={"query": "hi"}).status_code == 401
        assert client.get("/metrics").status_code == 401
        # Health stays open: a platform probe should not need the secret.
        assert client.get("/api/v1/health").status_code == 200
        ok = client.post(
            "/api/v1/query",
            json={"query": "what is a corporation"},
            headers={"X-API-Key": "s3cret"},
        )
        assert ok.status_code == 200
    finally:
        settings.api_key = None


def test_endpoints_stay_open_when_no_api_key_is_configured(client, settings):
    assert settings.api_key is None
    assert client.get("/metrics").status_code == 200
