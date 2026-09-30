"""Round 2 (Sep 29 evening): every recording grades, and benchmark_results.json traces to the raw files"""

import json
from pathlib import Path

import pytest

from app.ask import PRESETS
from app.quality import GRADERS, PROMPTS, json_verdict, puzzle_verdict, sheep_verdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "bench" / "raw" / "2026-09-29-r2"
SETUPS = ("BF16", "INT4", "INT4_RH", "SPEC_DECODE", "FP8")
# reworded after round 2; its round-2 recordings answer the old prompt
RE_RECORD = {"long_explanation"}
BENCH = json.loads((ROOT / "benchmark_results.json").read_text())


@pytest.mark.parametrize("setup", SETUPS)
def test_captures_complete(setup):
    for scenario, _label in PRESETS.values():
        if scenario in RE_RECORD:
            continue
        rec = json.loads((ROOT / "quality" / setup / f"{scenario}.json").read_text())
        assert rec["prompt"] == PROMPTS[scenario]
        if (setup, scenario) == ("SPEC_DECODE", "complex_reasoning"):
            # the r1 capture of the same answer stands in for the cold r2 recording
            # (bench/raw/README.md); it has no timings, and the answer is the r2 one token for token
            r1 = RAW.parent / "2026-09-29-r1" / "captures" / setup / f"{scenario}.json"
            assert rec == json.loads(r1.read_text())
            r2 = json.loads((RAW / "captures" / setup / f"{scenario}.json").read_text())
            assert rec["response_text"] == r2["response_text"] and rec["temperature"] == 0
            assert rec["raw"]["choices"][0]["finish_reason"] == "stop" and "ttft_ms" not in rec
        else:
            assert rec["finish_reason"] == "stop" and rec["temperature"] == 0 and rec["max_tokens"] == 1024
            assert rec["ttft_ms"] > 0 and rec["total_ms"] > rec["ttft_ms"] and rec["tokens_per_second"] > 0
        if grader := GRADERS.get(scenario):
            assert grader(rec["response_text"]), (setup, scenario)


@pytest.mark.parametrize("setup", SETUPS)
def test_puzzle_samples(setup):
    samples = json.loads((ROOT / "quality" / setup / "logic_puzzle_samples_t0.7.json").read_text())
    assert samples["n"] == 5 == len(samples["responses"])
    assert all(puzzle_verdict(t) for t in samples["responses"]), setup


def test_graders_on_captures():
    """The recordings graded: sheep 9, Carol on Monday, and all three JSON values, on all four."""
    for setup in SETUPS:
        folder = ROOT / "quality" / setup
        assert sheep_verdict(json.loads((folder / "complex_reasoning.json").read_text())["response_text"])
        assert json_verdict(json.loads((folder / "json_extraction.json").read_text())["response_text"])


def test_single_stream_provenance():
    raw_dir = {"BF16": "FP16"}  # the raw folders keep the names they were delivered with
    raw = {k: json.loads((RAW / "sweeps" / raw_dir.get(k, k) / "single-t0.json").read_text()) for k in SETUPS}
    v = BENCH["variants"]
    assert v["BF16"]["throughput_tps"] == round(raw["BF16"]["output_throughput"], 1) == 49.7
    assert v["INT4"]["throughput_tps"] == round(raw["INT4_RH"]["output_throughput"], 1) == 44.0
    assert v["INT4"]["reference"]["throughput_tps"] == round(raw["INT4"]["output_throughput"], 1) == 47.7
    assert v["SPEC_DECODE"]["throughput_tps"] == round(raw["SPEC_DECODE"]["output_throughput"], 1) == 62.2
    spec = v["SPEC_DECODE"]
    assert spec["speed_vs_baseline"] == 1.252 and spec["speed_vs_baseline_t0_7"] == 1.184
    assert v["INT4"]["speed_vs_baseline"] == 0.886 and v["INT4"]["reference"]["speed_vs_baseline"] == 0.959
    assert all(raw[k]["failed"] == 0 and raw[k]["completed"] == 30 for k in SETUPS)


def test_acceptance_is_isolated_per_temperature():
    acc = BENCH["variants"]["SPEC_DECODE"]["acceptance"]
    assert acc["0"] == {"drafts": 1330, "draft_tokens": 6650, "accepted": 4688, "rate": 0.705,
                        "mean_acceptance_length": 4.52}
    assert acc["0.7"]["rate"] == 0.6386 and acc["0.7"]["mean_acceptance_length"] == 4.19


def test_accuracy_carries_its_sample_sizes():
    for key in ("BF16", "INT4"):
        acc = BENCH["variants"][key]["accuracy"]
        assert acc["gsm8k"]["questions"] == 1319 and acc["gsm8k"]["fewshot"] == 8
        assert acc["mmlu_pro"]["questions"] == 280 and acc["mmlu_pro"]["subjects"] == 14
        assert 2.7 < acc["mmlu_pro"]["stderr"] < 2.9
    assert BENCH["variants"]["BF16"]["accuracy"]["mmlu_pro"]["score"] == 66.79
    assert BENCH["variants"]["INT4"]["accuracy"]["mmlu_pro"]["score"] == 63.57
    assert BENCH["variants"]["INT4"]["reference"]["accuracy"]["mmlu_pro"]["score"] == 62.86
    assert "accuracy" not in BENCH["variants"]["SPEC_DECODE"]  # its output is the 70B's by design


def test_int4_build_and_reference():
    int4 = BENCH["variants"]["INT4"]
    assert int4["checkpoint"].startswith("RedHatAI/") and int4["sweep_dir"] == "INT4_RH"
    assert int4["kernel"] == "MacheteLinearKernel" == int4["reference"]["kernel"]
    assert int4["reference"]["checkpoint"].startswith("hugging-quants/")


def test_benchmark_file_reproducible():
    import importlib.util

    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts" / "build_benchmark_file.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.build_round2() == BENCH
