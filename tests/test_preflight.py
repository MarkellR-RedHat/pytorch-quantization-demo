"""The preflight script: endpoint checks, recordings, the INT4 build, and the badge line"""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from app.ask import PRESETS
from app.config import settings
from app.quality import PROMPTS

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts" / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

NAMES = {"BF16": "benchmark-bf16", "INT4": "benchmark-int4-rh", "SPEC_DECODE": "benchmark-spec"}


@pytest.fixture
def setups(monkeypatch, tmp_path):
    """Three live setups on a mock vLLM, with every preset recorded (INT4 from Red Hat's build)."""
    for key, name in NAMES.items():
        port = {"BF16": 1, "INT4": 2, "SPEC_DECODE": 3}[key]
        monkeypatch.setattr(settings, f"model_{key.lower()}_endpoint", f"http://vllm{port}.test/v1/chat/completions")
        monkeypatch.setattr(settings, f"model_{key.lower()}_name", name)
        folder = tmp_path / ("INT4_RH" if key == "INT4" else key)
        folder.mkdir()
        for scenario, _label in PRESETS.values():
            record = {"prompt": PROMPTS[scenario], "response_text": "ok"}
            (folder / f"{scenario}.json").write_text(json.dumps(record))
    monkeypatch.setattr(settings, "model_fp8_endpoint", "")
    monkeypatch.setattr(settings, "model_int4_captures", "INT4_RH")
    monkeypatch.setattr(settings, "quality_dir", str(tmp_path))
    monkeypatch.setattr(settings, "simulation_mode", False)
    return tmp_path


def vllm(down=()):
    def handler(request):
        host = request.url.host
        key = {"vllm1.test": "BF16", "vllm2.test": "INT4", "vllm3.test": "SPEC_DECODE"}[host]
        if key in down:
            return httpx.Response(503)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": NAMES[key]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def results(rows):
    return {(setup, check): result for setup, check, result, _detail in rows}


def test_preflight_pass(setups):
    rows = preflight.run(vllm())
    assert all(r[2] == "PASS" for r in rows), rows
    assert rows[-1][3] == '"Llama 70B · Live models"'
    assert results(rows)[("INT4 (GPTQ via AutoGPTQ)", "warm-up")] == "PASS"


def test_preflight_down_endpoint(setups):
    rows = results(preflight.run(vllm(down={"SPEC_DECODE"})))
    assert rows[("Spec Decode", "models list")] == "FAIL"


def test_preflight_missing_recording(setups):
    (setups / "BF16" / "logic_puzzle.json").unlink()
    rows = preflight.run(vllm())
    row = next(r for r in rows if r[:2] == ("BF16", "recordings"))
    assert row[2] == "FAIL" and "Logic puzzle" in row[3]


def test_preflight_recorded_setup(setups, monkeypatch):
    monkeypatch.setattr(settings, "model_spec_decode_mode", "recorded")
    rows = preflight.run(vllm(down={"SPEC_DECODE"}))
    got = results(rows)
    assert got[("Spec Decode", "mode")] == "PASS" and ("Spec Decode", "models list") not in got
    assert rows[-1][3] == '"Llama 70B · Live: BF16, INT4 (GPTQ via AutoGPTQ) · Recorded: Spec Decode"'


def test_preflight_build_mismatch(setups, monkeypatch):
    monkeypatch.setattr(settings, "model_int4_captures", "INT4")
    assert results(preflight.run(vllm()))[("INT4 AWQ", "INT4 build")] == "WARN"


def test_preflight_replay_badge(setups, monkeypatch):
    monkeypatch.setattr(settings, "simulation_mode", True)
    assert preflight.run(vllm())[-1][2] == "FAIL"


def test_preflight_day_of_env(setups, monkeypatch):
    """Oct 20: BF16, spec decode and FP8 live on five H200s, INT4 recorded by plan from quality/INT4_RH/."""
    monkeypatch.setattr(settings, "model_int4_endpoint", "")
    monkeypatch.setattr(settings, "model_int4_name", "")
    monkeypatch.setattr(settings, "model_int4_mode", "recorded")
    monkeypatch.setattr(settings, "model_fp8_endpoint", "http://vllm4.test/v1/chat/completions")
    monkeypatch.setattr(settings, "model_fp8_name", "benchmark-fp8")
    (setups / "FP8").mkdir()
    for scenario, _label in PRESETS.values():
        record = {"prompt": PROMPTS[scenario], "response_text": "ok"}
        (setups / "FP8" / f"{scenario}.json").write_text(json.dumps(record))

    def handler(request):
        if request.url.host == "vllm4.test":
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "benchmark-fp8"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        if request.url.host == "vllm2.test":
            raise AssertionError("the recorded INT4 setup must not be called")
        if request.url.path.endswith("/models"):
            key = {"vllm1.test": "BF16", "vllm3.test": "SPEC_DECODE"}[request.url.host]
            return httpx.Response(200, json={"data": [{"id": NAMES[key]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    rows = preflight.run(httpx.Client(transport=httpx.MockTransport(handler)))
    assert all(r[2] == "PASS" for r in rows), [r for r in rows if r[2] != "PASS"]
    got = results(rows)
    int4 = "INT4 (GPTQ via AutoGPTQ)"
    assert got[(int4, "mode")] == "PASS" and (int4, "models list") not in got
    assert got[("FP8", "warm-up")] == "PASS"
    assert rows[-1][3] == '"Llama 70B · Live: BF16, FP8, Spec Decode · Recorded: INT4 (GPTQ via AutoGPTQ)"'


def test_preflight_fully_recorded_qwen_track(monkeypatch):
    """The Qwen track on the day: every column recorded by plan (the default), so it passes on its
    recordings alone and no endpoint is called; run_all checks it after the Llama track."""
    from app import tracks

    monkeypatch.setattr(settings, "simulation_mode", False)
    tracks.select("qwen")
    try:
        def never(request):
            raise AssertionError(f"a recorded column must not call {request.url}")

        rows = preflight.run(httpx.Client(transport=httpx.MockTransport(never)))
    finally:
        tracks.select("llama")
    assert all(r[2] == "PASS" for r in rows), [r for r in rows if r[2] != "PASS"]
    got = results(rows)
    for label in ("BF16", "FP8", "INT4 (GPTQ via LLM Compressor)", "Spec Decode"):
        assert got[(label, "mode")] == "PASS" and got[(label, "recordings")] == "PASS"
    assert rows[-1][3] == '"Qwen 27B · Recorded: BF16, FP8, INT4 (GPTQ via LLM Compressor), Spec Decode"'
    assert "quality/qwen/INT4/" in [r for r in rows if r[0].startswith("INT4")][1][3]


def test_preflight_run_all_covers_both_tracks(setups, monkeypatch):
    from app import tracks

    rows = preflight.run_all(vllm())
    assert tracks.active().key == "llama"
    setups_seen = {r[0] for r in rows}
    assert "Llama 70B: BF16" in setups_seen and "Qwen 27B: BF16" in setups_seen
    assert all(r[2] == "PASS" for r in rows), [r for r in rows if r[2] != "PASS"]
