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

NAMES = {"FP16": "benchmark-bf16", "INT4": "benchmark-int4-rh", "SPEC_DECODE": "benchmark-spec"}


@pytest.fixture
def setups(monkeypatch, tmp_path):
    """Three live setups on a mock vLLM, with every preset recorded (INT4 from Red Hat's build)."""
    for key, name in NAMES.items():
        port = {"FP16": 1, "INT4": 2, "SPEC_DECODE": 3}[key]
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
        key = {"vllm1.test": "FP16", "vllm2.test": "INT4", "vllm3.test": "SPEC_DECODE"}[host]
        if key in down:
            return httpx.Response(503)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": NAMES[key]}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def results(rows):
    return {(setup, check): result for setup, check, result, _detail in rows}


def test_all_live_and_recorded_passes(setups):
    rows = preflight.run(vllm())
    assert all(r[2] == "PASS" for r in rows), rows
    assert rows[-1][3] == '"Live models"'
    assert results(rows)[("INT4 (LLM Compressor)", "warm-up")] == "PASS"


def test_a_down_endpoint_fails(setups):
    rows = results(preflight.run(vllm(down={"SPEC_DECODE"})))
    assert rows[("Spec Decode", "models list")] == "FAIL"


def test_a_missing_recording_fails(setups):
    (setups / "FP16" / "logic_puzzle.json").unlink()
    rows = preflight.run(vllm())
    row = next(r for r in rows if r[:2] == ("BF16", "recordings"))
    assert row[2] == "FAIL" and "Logic puzzle" in row[3]


def test_a_recorded_setup_skips_its_endpoint_and_shows_in_the_badge(setups, monkeypatch):
    monkeypatch.setattr(settings, "model_spec_decode_mode", "recorded")
    rows = preflight.run(vllm(down={"SPEC_DECODE"}))
    got = results(rows)
    assert got[("Spec Decode", "mode")] == "PASS" and ("Spec Decode", "models list") not in got
    assert rows[-1][3] == '"Live: BF16, INT4 (LLM Compressor) · Recorded: Spec Decode"'


def test_int4_build_mismatch_warns(setups, monkeypatch):
    monkeypatch.setattr(settings, "model_int4_captures", "INT4")
    assert results(preflight.run(vllm()))[("INT4 AWQ", "INT4 build")] == "WARN"


def test_replay_badge_fails(setups, monkeypatch):
    monkeypatch.setattr(settings, "simulation_mode", True)
    assert preflight.run(vllm())[-1][2] == "FAIL"
