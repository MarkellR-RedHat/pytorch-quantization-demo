"""The Qwen track's Sep 30 run: benchmark_results.qwen.json traces to bench/raw/2026-09-30-qwen-r1, and
every recording the track plays grades."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.ask import PRESETS
from app.quality import GRADERS, PROMPTS, puzzle_verdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "bench" / "raw" / "2026-09-30-qwen-r1"
BENCH = json.loads((ROOT / "benchmark_results.qwen.json").read_text())
SETUPS = ("BF16", "FP8", "INT4", "SPEC_DECODE")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_file_rebuilds_from_the_raw_files():
    assert load_module("build_qwen_file").build(RAW) == BENCH


@pytest.mark.parametrize("setup", SETUPS)
def test_recordings_complete_and_graded(setup):
    for scenario, _label in PRESETS.values():
        rec = json.loads((ROOT / "quality" / "qwen" / setup / f"{scenario}.json").read_text())
        assert rec["prompt"] == PROMPTS[scenario]
        assert rec["finish_reason"] == "stop" and rec["temperature"] == 0 and rec["max_tokens"] == 1024
        assert rec["ttft_ms"] > 0 and rec["total_ms"] > rec["ttft_ms"] and rec["tokens_per_second"] > 0
        assert "<think>" not in rec["response_text"]  # thinking was off server-side
        if grader := GRADERS.get(scenario):
            assert grader(rec["response_text"]), (setup, scenario)
        assert rec == json.loads((RAW / "captures" / setup / f"{scenario}.json").read_text())
    samples = json.loads((ROOT / "quality" / "qwen" / setup / "logic_puzzle_samples_t0.7.json").read_text())
    assert samples["temperature"] == 0.7 and len(samples["responses"]) == 5
    assert all(puzzle_verdict(r) for r in samples["responses"]), setup


def test_under_load_at_the_target_from_clean_points():
    from app.benchmark import BenchmarkData
    from app.config import settings

    b = BenchmarkData(ROOT / "benchmark_results.qwen.json", ROOT / settings.qwen_bench_dir)
    best = b.meta()["at_target"]
    got = {k: (t["concurrency"], t["output_tokens_per_second_per_gpu"], t["lower_bound"])
           for k, t in best.items()}
    assert got == {
        "BF16": (64, 1980.0, False), "FP8": (64, 1540.9, False), "INT4": (32, 1017.9, False),
        "SPEC_DECODE": (32, 1597.8, False),
    }
    assert best["FP8"]["output_tokens_per_second_per_h200"] == 3081.9  # arithmetic, two slices per card
    assert all(not p["overlapped_with"] for pts in b.meta()["load"].values() for p in pts)


def test_the_facts_the_talk_quotes():
    v = BENCH["variants"]
    assert BENCH["vllm_version"] == "0.24.0+rhaiv.13" and BENCH["date"] == "2026-09-30"
    assert [v[k]["throughput_tps"] for k in SETUPS] == [66.0, 63.5, 56.8, 151.2]
    assert v["SPEC_DECODE"]["speed_vs_baseline"] == 2.291 and v["INT4"]["speed_vs_baseline"] == 0.861
    # devices: full cards with CUDA graphs, 71 GB slices with CUDA graphs, the 35 GB footnote in eager mode
    assert v["FP8"]["device"] == v["INT4"]["device"] == {"name": "71 GB slice", "count": 1, "per_h200": 2}
    assert not any(v[k]["enforce_eager"] for k in SETUPS)
    assert all(v[k]["cuda_graphs_captured"] for k in SETUPS)
    s35 = v["INT4"]["int4_35"]
    assert s35["enforce_eager"] and s35["throughput_tps"] == 15.9 and s35["device"]["per_h200"] == 3
    assert BENCH["eager_devices"] == ["35 GB slice"]
    # the pods each setup's log came from
    assert v["FP8"]["pod"] == "qwen-fp8-predictor-5446d888bb-r9nmt"
    assert v["FP8"]["thinking_in_server_args"] is False
    assert "off server-side in every pod" in BENCH["thinking"]
    # accuracy: gsm8k_cot (not the Llama prompt format), the latest of FP8's three files
    for k in ("BF16", "FP8", "INT4"):
        assert v[k]["accuracy"]["gsm8k"]["task"] == "gsm8k_cot"
        assert v[k]["accuracy"]["gsm8k"]["questions"] == 1319
    assert v["FP8"]["accuracy"]["gsm8k"]["score"] == 88.63
    assert [r["score"] for r in v["FP8"]["accuracy"]["gsm8k"]["other_runs"]] == [88.63, 89.08]
    mmlu = [v[k]["accuracy"]["mmlu_pro"]["score"] for k in ("BF16", "FP8", "INT4")]
    assert mmlu == [77.14, 77.5, 77.86]
    assert "accuracy" not in v["SPEC_DECODE"] and "accuracy" not in s35
    # MTP: from the counter deltas, every K the run made
    sd = v["SPEC_DECODE"]
    assert sd["spec_tokens_measured"] == [1, 2, 4, 8] and sd["draft_acceptance_rate"] == 0.5542
    assert sd["mean_acceptance_length"] == 3.22
    assert sd["acceptance"]["t0-k4"]["accepted_per_position"] == [0.7871, 0.6057, 0.4636, 0.3603]
    assert [sd[f"throughput_tps_k{k}"] for k in (1, 2, 8)] == [105.2, 128.6, 142.7]
    # the evening pass: clean temperature-0.7 runs, one client per pod, the afternoon pass set aside
    assert [v[k]["at_temperature_0_7"]["throughput_tps"] for k in SETUPS] == [65.3, 62.9, 56.3, 135.4]
    for k in SETUPS:
        assert all(r["overlapped_with"] == [] for r in v[k]["sweep_runs"].values()), k
        assert v[k]["first_pass"]["folder"] == f"sweeps/{k}/first-pass" and "excluded_runs" not in v[k]
    assert "GSM8K eval" in v["BF16"]["first_pass"]["note"]
    assert all(r["overlapped_with"] == [] for r in s35["sweep_runs"].values())
    # per-H200 stays arithmetic: two slices were not loaded at once
    assert v["FP8"]["per_h200"] == {"slices": 2, "throughput_tps": 127.0, "note": BENCH["per_h200_note"]}
    assert "measured" not in v["FP8"]["per_h200"]
