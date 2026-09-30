"""The Qwen builder against a synthetic results-qwen/ in the agreed layout, so the real import is a drop-in.
The INT4 card is the 35 GB slice: speed, load numbers and recordings from INT4_35, accuracy from the same
checkpoint on the 71 GB slice, and that run kept under the card as the like-for-like line against FP8."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main, tracks
from app.config import build_note, settings, variant_label
from tests import qwen_synthetic

ROOT = Path(__file__).resolve().parent.parent


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    raw = qwen_synthetic.write(tmp_path_factory.mktemp("results-qwen"))
    builder = load_module("build_qwen_file")
    return raw, builder.build(raw)


def test_four_setups_with_devices(built):
    _raw, out = built
    v = out["variants"]
    assert list(v) == ["BF16", "FP8", "INT4", "SPEC_DECODE"]
    assert v["BF16"]["device"] == {"name": "H200", "count": 1, "per_h200": 1}
    assert v["FP8"]["device"] == {"name": "71 GB slice", "count": 1, "per_h200": 2}
    assert v["INT4"]["device"] == {"name": "35 GB slice", "count": 1, "per_h200": 3}
    assert v["INT4"]["gpus"] == 1
    assert out["vllm_version"] == "0.24.0+rhaiv.13" and out["transformers_version"] == "5.16.1"
    assert out["thinking"] == "thinking: off"


def test_numbers_trace_to_the_files(built):
    raw, out = built
    for key, folder in (("BF16", "BF16"), ("FP8", "FP8"), ("INT4", "INT4_35")):
        single = json.loads((raw / "sweeps" / folder / "single-t0.json").read_text())
        assert out["variants"][key]["throughput_tps"] == round(single["output_throughput"], 1)
        assert out["variants"][key]["sweep_dir"] == out["variants"][key]["captures_dir"] == folder
    assert out["variants"]["SPEC_DECODE"]["speed_vs_baseline"] == 1.3
    assert out["variants"]["BF16"]["weights_gib_per_gpu"] == 51.89
    assert out["variants"]["BF16"]["kv_cache_tokens"] == 403613
    assert out["variants"]["FP8"]["kernels"] == ["CutlassFP8ScaledMMLinearKernel"]
    assert out["variants"]["INT4"]["kernels"] == ["MacheteLinearKernel"]
    digest = out["variants"]["BF16"]["image_digest"]
    assert digest.startswith("registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e6")
    assert "'max_model_len': 32768" in out["variants"]["BF16"]["non_default_args"]


def test_per_h200_is_separate_and_labeled(built):
    _raw, out = built
    v = out["variants"]
    assert v["BF16"]["per_h200"] == {
        "slices": 1, "throughput_tps": 60.0, "note": "one full H200, the measured number",
    }
    assert v["FP8"]["per_h200"]["throughput_tps"] == round(2 * v["FP8"]["throughput_tps"], 1)
    assert v["INT4"]["per_h200"]["throughput_tps"] == round(3 * v["INT4"]["throughput_tps"], 1)
    assert "not a measurement" in v["INT4"]["per_h200"]["note"]
    assert "not a measurement" in out["per_h200_note"]
    assert v["INT4"]["speed_vs_baseline"] == round(40.0 / 60.0, 3)  # per device, never the ×3 figure


def test_int4_card_accuracy_from_the_71gb_run_with_that_run_as_a_footnote(built):
    _raw, out = built
    v = out["variants"]
    int4, foot = v["INT4"], v["INT4"]["int4_71"]
    assert int4["accuracy"]["gsm8k"]["task"] == "gsm8k_cot" and int4["accuracy"]["gsm8k"]["questions"] == 1319
    assert int4["accuracy"]["mmlu_pro"]["questions"] == 280 and int4["accuracy"]["mmlu_pro"]["subjects"] == 14
    assert "property of the checkpoint" in int4["accuracy_note"]
    assert foot["device"] == {"name": "71 GB slice", "count": 1, "per_h200": 2}
    assert foot["throughput_tps"] == 52.0
    assert foot["speed_vs_baseline"] == round(52.0 / 60.0, 3)
    assert foot["accuracy"] == int4["accuracy"]  # the run the evals actually ran on
    assert "c64-failed.txt" in int4["sweep_note"] and "sharegpt-c64-failed.txt" in int4["sweep_note"]
    assert "sweep_note" not in foot and "accuracy" not in v["SPEC_DECODE"]
    assert int4["build"].startswith("Red Hat's LLM Compressor W4A16 build")


def test_mtp_acceptance_per_temperature_and_k(built):
    _raw, out = built
    spec = out["variants"]["SPEC_DECODE"]
    assert spec["spec_method"] == "mtp" and spec["spec_tokens_from_log"] == 4
    acc = spec["acceptance"]
    assert set(acc) == {"t0-k1", "t0-k2", "t0-k4", "t0.7-k4"}
    assert acc["t0-k4"]["rate"] == 0.72 and acc["t0-k4"]["mean_acceptance_length"] == round(1 + 0.72 * 4, 2)
    assert acc["t0-k1"]["rate"] == 0.9 and acc["t0.7-k4"]["rate"] == 0.64
    assert len(acc["t0-k4"]["accepted_per_position"]) == 4
    assert spec["draft_acceptance_rate"] == 0.72 and spec["throughput_tps_k1"] and spec["throughput_tps_k2"]


def test_counters_with_unknown_names_are_kept_raw(tmp_path):
    builder = load_module("build_qwen_file")
    (tmp_path / "logs").mkdir()
    for when, n in (("before", 0), ("after", 5)):
        line = f'vllm:spec_decode_something_new_total{{a="b"}} {n}\n'
        (tmp_path / "logs" / f"spec-metrics-{when}-t0-k4.txt").write_text(line)
    got = builder.acceptance(tmp_path, "0", 4)
    assert got == {"spec_tokens": 4, "counters": {"something_new_total": 5}}


def test_the_track_goes_ready_once_the_built_file_and_captures_exist(built, tmp_path, monkeypatch):
    raw, out = built
    bench = tmp_path / "benchmark_results.qwen.json"
    bench.write_text(json.dumps(out))
    monkeypatch.setattr(settings, "qwen_benchmark_file", str(bench))
    monkeypatch.setattr(settings, "qwen_quality_dir", str(raw / "captures"))
    monkeypatch.setattr(settings, "qwen_bench_dir", str(raw / "sweeps"))
    with TestClient(main.app) as client:
        assert client.post("/track/qwen").status_code == 200
        body = client.get("/api/config").json()
        assert body["track"]["status"] == "ready"
        int4 = next(v for v in body["variants"] if v["key"] == "INT4")
        assert int4["label"] == "INT4 (LLM Compressor W4A16)" and int4["device"]["name"] == "35 GB slice"
        assert int4["build"] == build_note("INT4")
        assert int4["build"] == "Red Hat's LLM Compressor W4A16 build (AWQ smoothing + GPTQ)"
        bm = body["benchmark"]
        assert bm["variants"]["INT4"]["label"] == "INT4 (LLM Compressor W4A16)"
        assert bm["variants"]["FP8"]["device"]["name"] == "71 GB slice"
        assert bm["variants"]["FP8"]["slices_per_h200"] == 2
        assert bm["variants"]["INT4"]["int4_71"]["device"]["name"] == "71 GB slice"
        # the INT4 card's load numbers are the 35 GB run's: c=32 is the last point under 50 ms, 480 tok/s
        # per slice, and c=64 failed on the slice
        best = bm["at_target"]["INT4"]
        assert best["output_tokens_per_second_per_gpu"] == 480.0
        assert best["output_tokens_per_second_per_h200"] == 1440.0 and best["slices_per_h200"] == 3
        assert bm["load"]["INT4"][-1]["concurrency"] == 32
        last = bm["load"]["BF16"][-1]
        assert last["output_tokens_per_second_per_h200"] == last["output_tokens_per_second_per_gpu"]
        # and its recordings come from captures/INT4_35
        assert settings.captures_for("INT4") == "INT4_35"
        stream = client.post("/ask/INT4", json={"preset": "puzzle"}).text
        assert "Monday" in stream
    tracks.select("llama")
    assert settings.captures_for("INT4") == "INT4_RH" and variant_label("INT4") == "INT4 (Red Hat W4A16)"
