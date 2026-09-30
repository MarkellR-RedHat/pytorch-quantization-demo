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


# ---------------------------------------------------------------- live failure falls back to a recording


def fake_vllm(monkeypatch, handler):
    """Point every setup at a mock vLLM endpoint served by handler, and switch to live mode."""
    import httpx

    from app.openshift import openshift_client

    for key in ("fp16", "int4", "spec_decode"):
        monkeypatch.setattr(settings, f"model_{key}_endpoint", "http://vllm.test/v1/chat/completions")
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(openshift_client, "_client", mock)
    monkeypatch.setattr(ask_module, "FIRST_TOKEN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(ask_module, "STALL_TIMEOUT_S", 0.3)
    main.simulator.disable()


def sse(*words, stall_after=None, delay_first=0.0):
    """A streamed chat completion that can wait before its first token or go quiet partway."""
    import asyncio

    async def body():
        if delay_first:
            await asyncio.sleep(delay_first)
        for i, w in enumerate(words):
            if stall_after is not None and i == stall_after:
                await asyncio.sleep(5)
            yield f'data: {json.dumps({"choices": [{"delta": {"content": w}}]})}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    return body()


def captured(key, scenario="complex_reasoning"):
    return json.loads((settings.resolve(settings.quality_dir) / key / f"{scenario}.json").read_text())


def test_live_error_plays_the_recorded_answer_labeled_as_recorded(client, monkeypatch):
    import httpx

    fake_vllm(monkeypatch, lambda request: httpx.Response(503))
    ev = events(client.post("/ask/FP16", json={"preset": "reasoning"}))
    assert ev[0] == {"t": "start", "source": "live"}
    fb = next(e for e in ev if e["t"] == "fallback")
    assert fb["label"] == "Recorded Sep 29 · live request failed"
    text = "".join(e["text"] for e in ev if e["t"] == "delta")
    assert text == captured("FP16")["response_text"]
    done = ev[-1]
    assert done["t"] == "done" and done["source"] == "recorded"
    assert done["completion_tokens"] == captured("FP16")["usage"]["completion_tokens"]
    # the Sep 29 recordings carry no timings, so nothing is shown as if it had been measured
    assert done["ttft_ms"] is None and done["total_ms"] is None and done["tps_basis"] == "benchmark"


def test_a_stalled_stream_keeps_its_live_tokens_and_says_how_many(client, monkeypatch):
    import httpx

    stalls = sse("The ", "farmer ", "has", stall_after=2)
    fake_vllm(monkeypatch, lambda request: httpx.Response(200, content=stalls))
    ev = events(client.post("/ask/INT4", json={"preset": "reasoning"}))
    kinds = [e["t"] for e in ev]
    assert kinds[:3] == ["start", "delta", "delta"] and kinds[3] == "fallback"
    assert ev[3]["label"] == "Recorded Sep 29 · live request failed after 2 tokens"
    assert ev[3]["live_tokens"] == 2
    recorded = "".join(e["text"] for e in ev[4:] if e["t"] == "delta")
    assert recorded == captured("INT4")["response_text"]


def test_a_slow_first_token_falls_back(client, monkeypatch):
    import httpx

    fake_vllm(monkeypatch, lambda request: httpx.Response(200, content=sse("Hi", delay_first=2)))
    ev = events(client.post("/ask/SPEC_DECODE", json={"preset": "reasoning"}))
    assert any(e["t"] == "fallback" for e in ev) and ev[-1]["source"] == "recorded"


def test_a_healthy_live_stream_is_not_labeled_recorded(client, monkeypatch):
    import httpx

    fake_vllm(monkeypatch, lambda request: httpx.Response(200, content=sse("Nine ", "sheep.")))
    ev = events(client.post("/ask/FP16", json={"preset": "reasoning"}))
    assert not any(e["t"] == "fallback" for e in ev)
    assert ev[-1]["source"] == "live" and ev[-1]["completion_tokens"] == 2


def test_a_typed_question_has_no_recording_to_fall_back_on(client, monkeypatch):
    import httpx

    fake_vllm(monkeypatch, lambda request: httpx.Response(503))
    ev = events(client.post("/ask/FP16", json={"prompt": "What is the capital of France?"}))
    assert ev[-1] == {"t": "error", "detail": ask_module.NO_RECORDING_TYPED}


def test_an_unrecorded_preset_says_so_instead_of_inventing_one(client, monkeypatch, no_captures):
    import httpx

    fake_vllm(monkeypatch, lambda request: httpx.Response(503))
    ev = events(client.post("/ask/FP16", json={"preset": "puzzle"}))
    assert ev[-1] == {"t": "error", "detail": ask_module.NO_RECORDING_PRESET}


def test_recorded_label_formats():
    label = ask_module.recorded_label("2026-09-30T14:02:11-04:00", 0)
    assert label == "Recorded Sep 30 · live request failed"
    assert ask_module.recorded_label(None, 12) == "Recorded earlier · live request failed after 12 tokens"


def test_every_preset_is_offered_with_its_exact_prompt(client):
    from app.quality import PROMPTS

    presets = client.get("/api/config").json()["presets"]
    assert len(presets) == 8
    for p in presets:
        assert p["prompt"] == PROMPTS[ask_module.PRESETS[p["key"]][0]]
        assert client.post("/ask/FP16", json={"preset": p["key"]}).status_code == 200


# ---------------------------------------------------------------- a setup recorded by plan for the day


def never_called(request):
    raise AssertionError("a recorded setup must never call its endpoint")


def test_a_recorded_setup_plays_its_recording_without_calling_the_endpoint(client, monkeypatch):
    fake_vllm(monkeypatch, never_called)
    monkeypatch.setattr(settings, "model_spec_decode_mode", "recorded")
    ev = events(client.post("/ask/SPEC_DECODE", json={"preset": "reasoning"}))
    assert ev[0] == {"t": "start", "source": "recorded"}
    assert ev[1] == {"t": "recorded", "label": "Recorded Sep 29"}
    assert "".join(e["text"] for e in ev if e["t"] == "delta") == captured("SPEC_DECODE")["response_text"]
    assert ev[-1]["source"] == "recorded" and not any(e["t"] in ("error", "fallback") for e in ev)


def test_a_typed_question_on_a_recorded_setup_explains_and_never_errors(client, monkeypatch):
    fake_vllm(monkeypatch, never_called)
    monkeypatch.setattr(settings, "model_fp16_mode", "recorded")
    ev = events(client.post("/ask/FP16", json={"prompt": "What is the capital of France?"}))
    assert [e["t"] for e in ev] == ["start", "delta", "done"]
    assert ev[1]["text"] == ask_module.NOT_LIVE_TODAY
    assert ev[-1]["note"] is True


def test_config_reports_each_setups_mode(client, monkeypatch):
    fake_vllm(monkeypatch, never_called)
    monkeypatch.setattr(settings, "model_spec_decode_mode", "recorded")
    modes = {v["key"]: v["mode"] for v in client.get("/api/config").json()["variants"]}
    assert modes == {"FP16": "live", "INT4": "live", "SPEC_DECODE": "recorded"}


def test_the_int4_column_can_run_red_hats_build(client, monkeypatch, tmp_path):
    from app.quality import PROMPTS

    (tmp_path / "INT4_RH").mkdir()
    (tmp_path / "INT4_RH" / "complex_reasoning.json").write_text(json.dumps({
        "prompt": PROMPTS["complex_reasoning"], "response_text": "Red Hat build: 9 sheep.",
        "captured_at": "2026-09-30T10:00:00-04:00",
    }))
    monkeypatch.setattr(settings, "quality_dir", str(tmp_path))
    monkeypatch.setattr(settings, "model_int4_captures", "INT4_RH")
    body = client.get("/api/config").json()
    int4 = next(v for v in body["variants"] if v["key"] == "INT4")
    assert int4["label"] == "INT4 (LLM Compressor)" and int4["build"] == "Red Hat LLM Compressor build"
    # the Sep 29 numbers are still the AWQ build's, and stay labeled that way
    assert body["benchmark"]["variants"]["INT4"]["label"] == "INT4 AWQ"
    ev = events(client.post("/ask/INT4", json={"preset": "reasoning"}))
    assert "".join(e["text"] for e in ev if e["t"] == "delta") == "Red Hat build: 9 sheep."
