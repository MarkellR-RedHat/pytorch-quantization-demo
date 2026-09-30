"""The Qwen builder against a synthetic results-qwen/ in the delivered layout, so its rules (pod choice,
eager mode, overlaps, exclusions, any K, per-H200 arithmetic) are pinned independently of the real files."""

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
    # the builder names the real run's pods; the synthetic run has one pod per setup
    for key, spec in builder.SETUPS.items():
        spec["pod"] = f"{qwen_synthetic.SETUPS[key][0]}-predictor-abc12-xyz34"
    builder.INT4_35["pod"] = f"{qwen_synthetic.SETUPS['INT4_35'][0]}-predictor-abc12-xyz34"
    return raw, builder.build(raw)


def test_four_setups_with_devices(built):
    _raw, out = built
    v = out["variants"]
    assert list(v) == ["BF16", "FP8", "INT4", "SPEC_DECODE"]
    assert v["BF16"]["device"] == {"name": "H200", "count": 1, "per_h200": 1}
    assert v["FP8"]["device"] == v["INT4"]["device"] == {"name": "71 GB slice", "count": 1, "per_h200": 2}
    assert v["INT4"]["int4_35"]["device"] == {"name": "35 GB slice", "count": 1, "per_h200": 3}
    assert out["vllm_version"] == "0.24.0+rhaiv.13" and out["transformers_version"] == "5.16.1"
    assert out["thinking"].startswith("off; off server-side in every pod")
    assert out["date"] == "2026-10-02"


def test_numbers_trace_to_the_files(built):
    raw, out = built
    for key in ("BF16", "FP8", "INT4"):
        single = json.loads((raw / "sweeps" / key / "single-t0.json").read_text())
        assert out["variants"][key]["throughput_tps"] == round(single["output_throughput"], 1)
        assert out["variants"][key]["sweep_dir"] == out["variants"][key]["captures_dir"] == key
    assert out["variants"]["SPEC_DECODE"]["speed_vs_baseline"] == 1.3
    assert out["variants"]["BF16"]["weights_gib_per_gpu"] == 51.89
    assert out["variants"]["BF16"]["kv_cache_tokens"] == 403613
    assert out["variants"]["FP8"]["kernels"] == ["CutlassFP8ScaledMMLinearKernel"]
    assert out["variants"]["INT4"]["kernels"] == ["MacheteLinearKernel"]
    digest = out["variants"]["BF16"]["image_digest"]
    assert digest.startswith("registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:c056e6")
    assert "'max_model_len': 32768" in out["variants"]["BF16"]["non_default_args"]
    assert out["variants"]["BF16"]["pod"] == "qwen-bf16-predictor-abc12-xyz34"


def test_eager_mode_is_read_from_each_pod(built):
    _raw, out = built
    v = out["variants"]
    assert not v["INT4"]["enforce_eager"] and v["INT4"]["cuda_graphs_captured"]
    assert v["INT4"]["int4_35"]["enforce_eager"] and not v["INT4"]["int4_35"]["cuda_graphs_captured"]
    assert out["eager_devices"] == ["35 GB slice"]


def test_per_h200_is_separate_and_labeled(built):
    _raw, out = built
    v = out["variants"]
    assert v["BF16"]["per_h200"] == {
        "slices": 1, "throughput_tps": 60.0, "note": "one full H200, the measured number",
    }
    assert v["FP8"]["per_h200"]["throughput_tps"] == round(2 * v["FP8"]["throughput_tps"], 1)
    assert v["INT4"]["int4_35"]["per_h200"]["throughput_tps"] == round(3 * 40.0, 1)
    assert "not a measurement" in v["FP8"]["per_h200"]["note"]
    assert "not a measurement" in out["per_h200_note"]
    assert v["INT4"]["speed_vs_baseline"] == round(52.0 / 60.0, 3)  # per device, never the x2 figure


def test_temperature_0_7_and_first_pass(built):
    _raw, out = built
    v = out["variants"]
    assert v["BF16"]["at_temperature_0_7"]["throughput_tps"] == 59.0
    assert v["FP8"]["at_temperature_0_7"]["throughput_tps"] == 60.5
    assert v["FP8"]["speed_vs_baseline_t0_7"] == round(60.5 / 59.0, 3)
    assert "excluded_runs" not in v["FP8"] and "first_pass" not in v["FP8"]  # no first-pass/ folder here


def test_a_first_pass_folder_is_named_and_not_read(built, tmp_path):
    raw, _out = built
    first = raw / "sweeps" / "INT4" / "first-pass"
    first.mkdir()
    (first / "c64.json").write_text(json.dumps(qwen_synthetic.bench_json("qwen-int4", 1.0, 999.0, 64, 64)))
    try:
        builder = load_module("build_qwen_file")
        for key, spec in builder.SETUPS.items():
            spec["pod"] = f"{qwen_synthetic.SETUPS[key][0]}-predictor-abc12-xyz34"
        builder.INT4_35["pod"] = f"{qwen_synthetic.SETUPS['INT4_35'][0]}-predictor-abc12-xyz34"
        int4 = builder.build(raw)["variants"]["INT4"]
        assert int4["first_pass"]["folder"] == "sweeps/INT4/first-pass"
        assert "GSM8K eval" in int4["first_pass"]["note"]
        assert "c64.json" in int4["sweep_runs"] and int4["sweep_runs"]["c64.json"]["overlapped_with"] == []
    finally:
        (first / "c64.json").unlink()
        first.rmdir()


def test_overlapping_runs_are_annotated(built):
    _raw, out = built
    runs = out["variants"]["FP8"]["sweep_runs"]
    assert runs["c64.json"]["overlapped_with"] == [{"file": "sharegpt-c64.json", "concurrency": 64}]
    assert runs["sharegpt-c64.json"]["overlapped_with"] == [{"file": "c64.json", "concurrency": 64}]
    clean = [n for n, r in runs.items() if not r["overlapped_with"]]
    assert len(clean) == len(runs) - 2
    assert out["variants"]["FP8"]["overlap_note"].startswith("2 of 12 runs shared the pod")
    assert "overlap_note" not in out["variants"]["BF16"]


def test_int4_35_is_the_footnote_with_a_failed_point_kept(built):
    _raw, out = built
    v = out["variants"]
    foot = v["INT4"]["int4_35"]
    acc = v["INT4"]["accuracy"]
    assert acc["gsm8k"]["task"] == "gsm8k_cot" and acc["gsm8k"]["questions"] == 1319
    assert acc["mmlu_pro"]["questions"] == 280 and acc["mmlu_pro"]["subjects"] == 14
    assert "accuracy" not in foot and "accuracy" not in v["SPEC_DECODE"]
    assert foot["throughput_tps"] == 40.0 and foot["speed_vs_baseline"] == round(40.0 / 60.0, 3)
    assert "c64-failed.txt" in foot["sweep_note"] and "sharegpt-c64-failed.txt" in foot["sweep_note"]
    assert "eager mode" in foot["note"] and foot["build"].startswith("Red Hat's LLM Compressor W4A16 build")


def test_mtp_acceptance_per_temperature_and_any_k(built):
    _raw, out = built
    spec = out["variants"]["SPEC_DECODE"]
    assert spec["spec_method"] == "mtp" and spec["spec_tokens_from_log"] == 4
    acc = spec["acceptance"]
    assert set(acc) == {"t0-k1", "t0-k2", "t0-k4", "t0.7-k4", "t0-k8"}
    assert spec["spec_tokens_measured"] == [1, 2, 4, 8]
    assert acc["t0-k4"]["rate"] == 0.72 and acc["t0-k4"]["mean_acceptance_length"] == round(1 + 0.72 * 4, 2)
    assert acc["t0-k1"]["rate"] == 0.9 and acc["t0.7-k4"]["rate"] == 0.64 and acc["t0-k8"]["rate"] == 0.55
    assert len(acc["t0-k4"]["accepted_per_position"]) == 4
    assert len(acc["t0-k8"]["accepted_per_position"]) == 8
    assert acc["t0-k4"]["accepted_per_position"][0] == 0.72
    assert spec["draft_acceptance_rate"] == 0.72
    assert spec["throughput_tps_k1"] and spec["throughput_tps_k2"] and spec["throughput_tps_k8"]


def test_counters_with_unknown_names_are_kept_raw(tmp_path):
    builder = load_module("build_qwen_file")
    (tmp_path / "logs").mkdir()
    for when, n in (("before", 0), ("after", 5)):
        line = f'vllm:spec_decode_something_new_total{{a="b"}} {n}\n'
        (tmp_path / "logs" / f"spec-metrics-{when}-t0-k4.txt").write_text(line)
    got = builder.acceptance(tmp_path, "0", 4)
    assert got == {"spec_tokens": 4, "counters": {"something_new_total": 5}}


def test_the_track_serves_the_built_file(built, tmp_path, monkeypatch):
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
        assert int4["label"] == "INT4 (LLM Compressor W4A16)" and int4["device"]["name"] == "71 GB slice"
        assert int4["build"] == build_note("INT4")
        assert int4["build"] == "Red Hat's LLM Compressor W4A16 build (AWQ smoothing + GPTQ)"
        bm = body["benchmark"]
        assert bm["variants"]["INT4"]["label"] == "INT4 (LLM Compressor W4A16)"
        assert bm["variants"]["INT4"]["int4_35"]["device"]["name"] == "35 GB slice"
        assert bm["variants"]["INT4"]["int4_35"]["enforce_eager"] is True
        # per slice on the cards, per H200 as labeled arithmetic beside it
        best = bm["at_target"]["INT4"]  # c=64, the synthetic INT4's last point under 50 ms
        assert best["output_tokens_per_second_per_gpu"] == 980.0
        assert best["output_tokens_per_second_per_h200"] == 1960.0
        assert best["slices_per_h200"] == 2 and best["lower_bound"] is False
        # FP8's c64 shared its pod with the ShareGPT run: hollow on the chart, and the target line
        # prefers the clean c32 point
        fp8 = {p["concurrency"]: p for p in bm["load"]["FP8"]}
        assert fp8[64]["overlapped_with"] == [{"file": "sharegpt-c64.json", "concurrency": 64}]
        assert fp8[32]["overlapped_with"] == [] and bm["at_target"]["FP8"]["concurrency"] == 32
        assert settings.captures_for("INT4") == "INT4"
        stream = client.post("/ask/INT4", json={"preset": "puzzle"}).text
        assert "Monday" in stream
        assert len(body["presets"]) == 8  # every preset is recorded on this track
    tracks.select("llama")
    assert settings.captures_for("INT4") == "INT4_RH" and variant_label("INT4") == "INT4 (Red Hat W4A16)"


def test_a_lower_bound_wins_only_when_nothing_clean_is_under_target():
    from app.benchmark import at_target

    shared = [{"file": "sharegpt-c8.json", "concurrency": 8}]

    def point(c, tps, tail, overlapped):
        return {"concurrency": c, "output_tokens_per_second_per_gpu": tps, "tpot_tail_ms": tail,
                "tpot_tail_kind": "p95", "overlapped_with": overlapped}

    pts = [point(8, 300.0, 20.0, shared), point(32, 800.0, 40.0, shared), point(64, 900.0, 70.0, [])]
    best = at_target(pts, 50)
    assert best["concurrency"] == 32 and best["lower_bound"] is True and best["overlapped_with"] == shared
    pts[0]["overlapped_with"] = []
    assert at_target(pts, 50)["concurrency"] == 8  # the clean point wins even though it's smaller
