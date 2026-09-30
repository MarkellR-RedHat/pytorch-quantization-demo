"""Two tracks, one demo: the selector swaps the benchmark, the recordings and the device labels, and
refuses a track whose data hasn't landed unless a flag says pending is fine."""

import json

import pytest
from fastapi.testclient import TestClient

from app import main, tracks
from app.config import settings
from app.quality import PROMPTS


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c
    tracks.select("llama")


@pytest.fixture
def qwen_data(tmp_path, monkeypatch):
    """A Qwen track with data on disk: a benchmark file for four setups and one recording."""
    bench = tmp_path / "benchmark_results.qwen.json"
    bench.write_text(json.dumps({
        "model": "Qwen/Qwen3.8-27B", "vllm_version": "0.24.0+rhaiv.13", "date": "2026-10-02",
        "variants": {
            "BF16": {"gpus": 1, "throughput_tps": 60.0, "avg_latency_ms": 3000.0, "p95_latency_ms": 3200.0},
            "FP8": {"gpus": 1, "throughput_tps": 62.0, "avg_latency_ms": 2900.0, "p95_latency_ms": 3100.0},
            "INT4": {"gpus": 1, "throughput_tps": 55.0, "avg_latency_ms": 3300.0, "p95_latency_ms": 3500.0},
            "SPEC_DECODE": {
                "gpus": 1, "throughput_tps": 80.0, "avg_latency_ms": 2300.0, "p95_latency_ms": 2500.0,
            },
        },
    }))
    quality = tmp_path / "quality-qwen"
    (quality / "BF16").mkdir(parents=True)
    (quality / "BF16" / "complex_reasoning.json").write_text(json.dumps({
        "prompt": PROMPTS["complex_reasoning"], "response_text": "Qwen says 9 sheep.",
        "captured_at": "2026-10-02T10:00:00-04:00",
    }))
    monkeypatch.setattr(settings, "qwen_benchmark_file", str(bench))
    monkeypatch.setattr(settings, "qwen_quality_dir", str(quality))
    monkeypatch.setattr(settings, "qwen_bench_dir", str(tmp_path / "sweeps"))
    return tmp_path


def test_registry_lists_both_tracks_and_their_status():
    status = {t["key"]: t for t in tracks.listing()}
    assert status["llama"]["status"] == "ready"
    assert status["qwen"]["status"] == "pending"
    assert "benchmark_results.qwen.json" in " ".join(status["qwen"]["missing"])


def test_default_track_is_llama(client):
    body = client.get("/api/config").json()
    assert body["track"]["key"] == "llama" and body["track"]["model"].startswith("Llama 3.1 70B")
    assert [t["key"] for t in body["track"]["tracks"]] == ["llama", "qwen"]
    devices = {v["key"]: v["device"] for v in body["variants"]}
    assert devices["BF16"] == {"name": "H200", "count": 2, "per_h200": 1}
    assert devices["FP8"] == {"name": "H200", "count": 1, "per_h200": 1}


def test_a_pending_track_is_refused(client):
    r = client.post("/track/qwen")
    assert r.status_code == 409 and "pending" in r.json()["detail"]
    assert client.get("/api/config").json()["track"]["key"] == "llama"
    assert client.post("/track/nope").status_code == 404


def test_allow_pending_shows_the_track_with_no_numbers(client, monkeypatch):
    monkeypatch.setattr(settings, "allow_pending_tracks", True)
    assert client.post("/track/qwen").status_code == 200
    body = client.get("/api/config").json()
    assert body["track"]["key"] == "qwen" and body["track"]["status"] == "pending"
    assert body["benchmark"]["variants"] == {}


def test_presenter_query_selects_a_ready_track_and_ignores_a_pending_one(client, qwen_data):
    assert client.get("/presenter?track=qwen").status_code == 200
    assert client.get("/api/config").json()["track"]["key"] == "qwen"
    tracks.select("llama")
    with TestClient(main.app) as fresh:  # pending again once the files are gone
        settings.qwen_benchmark_file = str(qwen_data / "missing.json")
        assert fresh.get("/presenter?track=qwen").status_code == 200
        assert fresh.get("/api/config").json()["track"]["key"] == "llama"


def test_switching_tracks_swaps_benchmark_devices_and_recordings(client, qwen_data):
    assert client.post("/track/qwen").status_code == 200
    body = client.get("/api/config").json()
    assert body["track"]["key"] == "qwen" and body["track"]["status"] == "ready"
    assert body["benchmark"]["variants"]["BF16"]["throughput_tps"] == 60.0
    devices = {v["key"]: v["device"] for v in body["variants"]}
    assert devices["BF16"] == {"name": "H200", "count": 1, "per_h200": 1}
    assert devices["FP8"] == {"name": "71 GB slice", "count": 1, "per_h200": 2}
    assert devices["INT4"] == {"name": "35 GB slice", "count": 1, "per_h200": 3}
    stream = client.post("/ask/BF16", json={"preset": "reasoning"}).text
    lines = [json.loads(line) for line in stream.splitlines()]
    assert "".join(e["text"] for e in lines if e["t"] == "delta") == "Qwen says 9 sheep."
    # and back
    assert client.post("/track/llama").status_code == 200
    body = client.get("/api/config").json()
    assert body["track"]["key"] == "llama" and body["benchmark"]["variants"]["BF16"]["throughput_tps"] == 49.7
    assert main.simulator.benchmark is tracks.active().benchmark


def test_the_badge_and_title_name_the_track(client, qwen_data):
    client.post("/track/qwen")
    body = client.get("/api/config").json()
    assert body["track"]["title"] == tracks.TITLE == "Not Every Question Needs the Whole GPU"
    assert body["track"]["model"] == "Qwen3.8-27B"


def test_both_tracks_share_the_title(client):
    body = client.get("/api/config").json()
    assert body["track"]["title"] == tracks.TITLE
    assert "FP8" in body["track"]["subtitle"]
    assert body["track"]["lanes"]["INT4"]  # the routing strip reads its lane text from the track


def test_track_env_setting_picks_the_start_track(qwen_data, monkeypatch):
    monkeypatch.setattr(settings, "track", "qwen")
    tracks.select_from_settings()
    assert tracks.active().key == "qwen"
    tracks.select("llama")


def test_a_pending_start_track_falls_back_to_llama(monkeypatch):
    monkeypatch.setattr(settings, "track", "qwen")
    tracks.select_from_settings()
    assert tracks.active().key == "llama"
