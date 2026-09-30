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


@pytest.fixture
def no_captures(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "quality_dir", str(tmp_path))


def test_preset_streams_and_finishes_with_timing(client, no_captures):
    r = client.post("/ask/INT4", json={"preset": "reasoning"})
    assert r.status_code == 200
    ev = events(r)
    # with no captured output, replay shows a note and never an answer no model gave
    assert ev[0] == {"t": "start", "source": "replay", "text_source": "note"}
    assert ev[-1]["t"] == "done"
    # replay reports only what was measured: the benchmark's first-token time and speed, no token count
    assert ev[-1]["ttft_ms"] == main.benchmark_data.variant("INT4")["ttft_ms_avg"]
    assert ev[-1]["total_ms"] is None
    assert ev[-1]["completion_tokens"] is None
    assert ev[-1]["tokens_per_second"] == main.benchmark_data.variant("INT4")["throughput_tps"]
    text = "".join(e["text"] for e in ev if e["t"] == "delta")
    assert "no answer to this question was captured" in text
    assert "sheep" not in text


def test_replay_has_no_first_token_time_where_none_was_measured(client):
    ev = events(client.post("/ask/SPEC_DECODE", json={"preset": "reasoning"}))
    assert ev[-1]["ttft_ms"] is None


def test_replay_never_invents_a_quality_difference(client, no_captures):
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


def test_load_points_report_median_and_request_sizes(tmp_path):
    from app.benchmark import BenchmarkData

    (tmp_path / "INT4").mkdir()
    (tmp_path / "INT4" / "c8.json").write_text(json.dumps({
        "output_throughput": 400.0, "mean_e2el_ms": 6000.0, "median_e2el_ms": 5500.0,
        "completed": 64, "total_input_tokens": 64 * 512, "total_output_tokens": 64 * 256,
    }))
    bench = BenchmarkData(main.settings.resolve(main.settings.benchmark_file), tmp_path)
    [point] = bench.load_points("INT4")
    assert point["latency_ms"] == 5500.0 and point["latency_kind"] == "median"
    assert point["avg_input_tokens"] == 512 and point["avg_output_tokens"] == 256
    assert point["output_tokens_per_second_per_gpu"] == 400.0  # INT4 runs on one GPU


def test_replay_uses_each_setups_captured_answer(client):
    """With the Sep 29 captures in quality/, replay shows each setup's own real answer."""
    for key in ("FP16", "INT4", "SPEC_DECODE"):
        ev = events(client.post(f"/ask/{key}", json={"preset": "reasoning"}))
        assert ev[0]["text_source"] == "captured"
        text = "".join(e["text"] for e in ev if e["t"] == "delta")
        path = settings.resolve(settings.quality_dir) / key / "complex_reasoning.json"
        captured = json.loads(path.read_text())
        assert text == captured["response_text"]
