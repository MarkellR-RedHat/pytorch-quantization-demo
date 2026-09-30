"""Under load: output tokens per GPU at a latency target, from synthetic vllm bench serve files"""

import json

import pytest

from app import main
from app.benchmark import BenchmarkData, at_target
from app.config import settings


def sweep(folder, points, tail="p99"):
    """Write c<N>.json files shaped like `vllm bench serve --save-result` output."""
    folder.mkdir(parents=True)
    for c, (tput, med, tail_ms) in points.items():
        (folder / f"c{c}.json").write_text(json.dumps({
            "output_throughput": tput, "median_e2el_ms": 256 * med, "median_tpot_ms": med,
            f"{tail}_tpot_ms": tail_ms, "completed": 64, "total_input_tokens": 64 * 512,
            "total_output_tokens": 64 * 256,
        }))


@pytest.fixture
def bench(tmp_path, monkeypatch):
    bench = BenchmarkData(settings.resolve(settings.benchmark_file), tmp_path)
    # each column reads the folder its benchmark entry points at
    folder = lambda key: tmp_path / (bench.variant(key).get("sweep_dir") or key)  # noqa: E731
    sweep(folder("BF16"), {1: (45, 21, 24), 16: (600, 26, 31), 64: (1700, 37, 48)})
    sweep(folder("INT4"), {1: (44, 22, 25), 16: (470, 33, 40), 64: (950, 65, 88)})
    sweep(folder("SPEC_DECODE"), {1: (65, 15, 19), 16: (620, 25, 34), 64: (1200, 52, 71)}, tail="p95")
    monkeypatch.setattr(settings, "tpot_target_ms", 50.0)
    return bench


def test_points_carry_median_and_tail_time_per_token(bench):
    [c1, c16, c64] = bench.load_points("INT4")
    assert (c64["tpot_median_ms"], c64["tpot_tail_ms"], c64["tpot_tail_kind"]) == (65.0, 88.0, "p99")
    assert bench.load_points("SPEC_DECODE")[0]["tpot_tail_kind"] == "p95"
    assert c16["output_tokens_per_second_per_gpu"] == 470.0  # INT4 runs on one GPU


def test_throughput_counts_only_under_the_target(bench):
    best = bench.meta()["at_target"]
    assert best["BF16"]["concurrency"] == 64 and best["BF16"]["output_tokens_per_second_per_gpu"] == 850.0
    # INT4 at 64 in flight serves more per GPU, but its tail time per token breaks 50 ms
    assert best["INT4"]["concurrency"] == 16 and best["INT4"]["output_tokens_per_second_per_gpu"] == 470.0
    assert best["SPEC_DECODE"]["concurrency"] == 16 and best["SPEC_DECODE"]["tpot_tail_kind"] == "p95"


def test_the_target_is_a_setting(bench, monkeypatch):
    monkeypatch.setattr(settings, "tpot_target_ms", 100.0)
    assert bench.meta()["at_target"]["INT4"]["concurrency"] == 64
    assert bench.meta()["tpot_target_ms"] == 100.0


def test_no_point_under_the_target_means_no_number():
    assert at_target([{"tpot_tail_ms": 80.0, "output_tokens_per_second_per_gpu": 1.0}], 50.0) is None
    assert at_target([{"tpot_tail_ms": None, "output_tokens_per_second_per_gpu": 1.0}], 50.0) is None


def test_config_exposes_the_target_without_a_sweep():
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        bm = client.get("/api/config").json()["benchmark"]
    assert bm["tpot_target_ms"] == settings.tpot_target_ms
    assert bm["at_target"] == {} or set(bm["at_target"]) <= {"BF16", "INT4", "SPEC_DECODE", "FP8"}
