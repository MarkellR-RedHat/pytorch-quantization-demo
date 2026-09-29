"""The Ask scene: one question streamed to each variant, presenter-only, honest in replay mode."""

import json

import pytest
from fastapi.testclient import TestClient

from app import ask as ask_module
from app import main
from app.config import settings


@pytest.fixture
def client(monkeypatch):
    # replay without real waiting
    async def no_sleep(_):
        return None

    monkeypatch.setattr(ask_module, "_sleep", no_sleep)
    with TestClient(main.app) as c:
        yield c


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_preset_streams_and_finishes_with_timing(client):
    r = client.post("/ask/INT4", json={"preset": "reasoning"})
    assert r.status_code == 200
    ev = events(r)
    assert ev[0] == {"t": "start", "source": "replay"}
    assert ev[-1]["t"] == "done"
    assert ev[-1]["completion_tokens"] > 0
    assert ev[-1]["ttft_ms"] is None  # replay never invents a time to first token
    text = "".join(e["text"] for e in ev if e["t"] == "delta")
    assert "9 sheep" in text


def test_replay_never_invents_a_quality_difference(client):
    texts = {}
    for key in ("FP16", "INT4", "SPEC_DECODE"):
        ev = events(client.post(f"/ask/{key}", json={"preset": "reasoning"}))
        texts[key] = "".join(e["text"] for e in ev if e["t"] == "delta")
    assert texts["FP16"] == texts["INT4"] == texts["SPEC_DECODE"]


def test_free_text_in_replay_says_no_model_was_called(client):
    ev = events(client.post("/ask/FP16", json={"prompt": "What is the capital of France?"}))
    text = "".join(e["text"] for e in ev if e["t"] == "delta")
    assert "without calling a model" in text


def test_empty_question_and_unknown_variant_are_rejected(client):
    assert client.post("/ask/FP16", json={"prompt": "   "}).status_code == 422
    assert client.post("/ask/NOPE", json={"prompt": "hi"}).status_code == 404


def test_ask_requires_presenter_key_when_set(client, monkeypatch):
    monkeypatch.setattr(settings, "presenter_key", "s3cret")
    assert client.post("/ask/FP16", json={"prompt": "hi"}).status_code in (401, 403)
    client.cookies.set("presenter_key", "s3cret")
    assert client.post("/ask/FP16", json={"prompt": "hi"}).status_code == 200


def test_live_failure_becomes_an_error_event(client):
    main.simulator.disable()  # live mode with no endpoints configured
    ev = events(client.post("/ask/FP16", json={"prompt": "hi"}))
    assert ev[-1]["t"] == "error"


def test_arena_booth_page_is_served(client):
    assert client.get("/arena").status_code == 200
