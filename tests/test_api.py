"""API, auth, websocket and live-client tests"""

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import main
from app.config import settings


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def receive_until(ws, msg_type, limit=40):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == msg_type:
            return msg
    raise AssertionError(f"no {msg_type} message received")


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["mode"] == "simulated"


def test_config(client):
    body = client.get("/api/config").json()
    assert body["mode"] == "simulated"
    assert [v["key"] for v in body["variants"]] == ["BF16", "FP8", "INT4", "SPEC_DECODE"]
    assert body["variants"][0]["label"] == "BF16"
    assert body["variants"][0]["gpus"] == 2
    # weights per GPU come through from the startup-log figures, not null
    weights = {v["key"]: v["weights_gib_per_gpu"] for v in body["variants"]}
    # INT4 is Red Hat's build
    assert weights == {"BF16": 65.74, "FP8": 67.7, "INT4": 37.11, "SPEC_DECODE": 73.24}
    assert body["gpu_hourly_usd"] == 0
    bench = body["benchmark"]
    assert bench["variants"]["BF16"]["label"] == "BF16"
    raw = json.loads((main.settings.resolve(main.settings.benchmark_file)).read_text())["variants"]
    per_gpu = bench["variants"]["BF16"]["tokens_per_second_per_gpu"]
    assert per_gpu == pytest.approx(raw["BF16"]["throughput_tps"] / 2)
    assert bench["variants"]["INT4"]["gpus"] == 1
    fp8 = bench["variants"]["FP8"]
    assert fp8["gpus"] == 1 and fp8["kernel"] == "CutlassFP8ScaledMMLinearKernel"


def test_root_redirect(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/presenter"
    for gone in ("/api/me", "/qr.svg", "/arena/votes", "/arena/state"):
        assert client.get(gone).status_code == 404, gone
    assert client.post("/request", json={"model_type": "INT4"}).status_code in (404, 405)


def test_quality(client):
    assert client.get("/quality/nope").status_code == 404
    body = client.get("/quality/complex_reasoning").json()
    assert body["source"] in {"not_captured", "captured"}
    assert set(body["responses"]) <= {"BF16", "FP8", "INT4", "SPEC_DECODE"}


def test_metrics_shape(client):
    infer("INT4")
    snap = client.get("/metrics").json()["INT4"]
    assert snap["label"] == "INT4 (GPTQ via AutoGPTQ)"
    assert snap["source"] == "simulated"
    assert snap["basis"].startswith("benchmark")  # with a sweep on disk, the basis is "benchmark c≈1"
    assert snap["gpus"] == 1
    assert snap["total_requests"] == 1
    assert snap["cost_per_request"] is None
    assert snap["p95_latency_ms"] > 1000  # real per-request latency, seconds not milliseconds


def infer(model_type, prompt_id="chat"):
    """Run one inference the way background traffic does, and return the response or the HTTP error."""
    try:
        return asyncio.run(main.run_inference(model_type, prompt_id))
    except HTTPException as e:
        return e


def test_simulated_request(client):
    body = infer("BF16", "reasoning")
    assert body.source == "simulated"
    assert body.completion_tokens > 0
    assert body.cost_per_request is None


def test_fp8_variant(client):
    assert infer("FP8").model_type == "FP8"
    snap = client.get("/metrics").json()["FP8"]
    assert snap["gpus"] == 1 and snap["label"] == "FP8"


def test_cost_when_price_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "gpu_hourly_usd", 4.0)
    body = infer("BF16")
    expected = 2 * 4.0 * body.latency_ms / 1000 / 3600
    assert body.cost_per_request == pytest.approx(expected)


def test_busy_variant_fails_fast(client, monkeypatch):
    monkeypatch.setitem(main.sim_slots, "BF16", asyncio.Semaphore(0))
    error = infer("BF16")
    assert error.status_code == 503 and error.detail == "busy"


def test_live_mode_without_endpoint_is_clear(client, monkeypatch):
    main.simulator.disable()
    monkeypatch.setattr(settings, "model_bf16_endpoint", "")
    error = infer("BF16")
    assert error.status_code == 503 and "MODEL_BF16_ENDPOINT" in error.detail


def test_live_mode_uses_completion_tokens(client, monkeypatch):
    seen = {}

    def upstream(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "9"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000},
        })

    main.simulator.disable()
    monkeypatch.setattr(settings, "model_int4_endpoint", "http://upstream/v1/chat/completions")
    monkeypatch.setattr(settings, "model_int4_name", "llama-70b-int4")
    mock = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    monkeypatch.setattr(main.openshift_client, "_client", mock)
    body = infer("INT4")
    assert body.source == "live"
    assert body.completion_tokens == 100
    assert body.tokens_per_second == pytest.approx(100 / (body.latency_ms / 1000))
    assert seen["temperature"] == 0 and seen["model"] == "llama-70b-int4"


def test_upstream_error_detail(client, monkeypatch):
    main.simulator.disable()
    monkeypatch.setattr(settings, "model_int4_endpoint", "http://secret-internal-host/v1")
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    monkeypatch.setattr(main.openshift_client, "_client", httpx.AsyncClient(transport=transport))
    error = infer("INT4")
    assert error.status_code == 502
    assert "secret-internal-host" not in str(error.detail)
    assert main.metrics_collector.error_counts["INT4"] == 1


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(settings, "presenter_key", "s3cret")


def test_controls_require_key(client, key):
    for path in ("/demo/start", "/demo/stop", "/demo/reset", "/simulation/toggle"):
        assert client.post(path).status_code == 401
        assert client.post(path, headers={"X-Presenter-Key": "wrong"}).status_code == 403
    assert client.post("/demo/start", headers={"X-Presenter-Key": "s3cret"}).status_code == 200


def test_presenter_key_cookie(client, key):
    assert client.get("/presenter").status_code == 401
    assert client.get("/presenter?key=wrong", follow_redirects=False).status_code == 403
    response = client.get("/presenter?key=s3cret", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/presenter"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.post("/demo/stop").status_code == 200  # cookie now carries the key


def test_presenter_websocket_requires_key(client, key):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/presenter"):
            pass
    assert exc.value.code == 1008
    client.cookies.set("presenter_key", "s3cret")
    with client.websocket_connect("/ws/presenter") as ws:
        assert ws.receive_json()["type"] in {"metrics_update", "state_update"}


def test_stop_does_not_drop_clients(client):
    """Regression: stop/reset broadcast a datetime, which threw and silently dropped every client."""
    with client.websocket_connect("/ws/presenter") as ws:
        assert client.post("/demo/start").status_code == 200
        assert client.post("/demo/stop").status_code == 200
        for _ in range(10):
            msg = receive_until(ws, "state_update")
            if msg["data"]["is_running"] is False:
                break
        else:
            raise AssertionError("no state update after stop")
        assert client.post("/demo/reset").status_code == 200
        assert receive_until(ws, "state_update")["data"]["start_time"] is None


def test_screen_messages_are_ignored(client):
    with client.websocket_connect("/ws/presenter") as ws:
        ws.send_text("x" * 5000)
        ws.send_text("not json")
        ws.send_json({"type": "arena_state", "data": {"hacked": True}})
        ws.send_bytes(b"\x00\x01")
        receive_until(ws, "metrics_update")


def test_auto_traffic_generates_requests(client, monkeypatch):
    monkeypatch.setattr(settings, "auto_traffic_rps", 50.0)
    client.post("/demo/start")
    with client.websocket_connect("/ws/presenter") as ws:
        for _ in range(40):
            msg = receive_until(ws, "state_update")
            if msg["data"]["total_requests"] > 0:
                break
        else:
            raise AssertionError("auto traffic produced no requests")
    client.post("/demo/stop")
