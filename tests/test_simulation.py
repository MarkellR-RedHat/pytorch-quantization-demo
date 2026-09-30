"""Simulation mode must be grounded in benchmark_results.json, not invented numbers"""

import json
import random
import statistics

import pytest

from app.benchmark import BenchmarkData, lognormal_sigma
from app.config import REPO_ROOT
from app.simulation import Simulator

BENCH = json.loads((REPO_ROOT / "benchmark_results.json").read_text())


@pytest.fixture
def sim(tmp_path):
    return Simulator(BenchmarkData(REPO_ROOT / "benchmark_results.json", tmp_path / "no-sweeps"))


class TestSimulator:

    def test_starts_disabled_and_toggles(self, sim):
        assert sim.is_enabled() is False
        sim.enable()
        assert sim.is_enabled() is True
        sim.disable()
        assert sim.is_enabled() is False

    @pytest.mark.parametrize("variant", ["FP16", "INT4", "SPEC_DECODE"])
    def test_samples_match_benchmark_mean_and_p95(self, sim, variant):
        rng = random.Random(7)
        latencies = sorted(sim.sample(variant, 1, rng)[0] for _ in range(20000))
        expected = BENCH["variants"][variant]
        assert statistics.mean(latencies) == pytest.approx(expected["avg_latency_ms"], rel=0.02)
        assert latencies[int(0.95 * len(latencies))] == pytest.approx(expected["p95_latency_ms"], rel=0.03)

    @pytest.mark.parametrize("variant", ["FP16", "INT4", "SPEC_DECODE"])
    def test_tokens_track_benchmark(self, sim, variant):
        rng = random.Random(3)
        avg = BENCH["variants"][variant]["avg_tokens_per_request"]
        tokens = [sim.sample(variant, 1, rng)[1] for _ in range(2000)]
        assert min(tokens) >= int(avg * 0.9) and max(tokens) <= round(avg * 1.1)

    async def test_simulate_request_returns_consistent_tps(self, sim):
        text, latency_ms, tps, tokens = await sim.simulate_request("INT4")
        assert text.startswith("[simulated]")
        assert tps == pytest.approx(tokens / (latency_ms / 1000))

    def test_basis_is_single_stream_without_sweeps(self, sim):
        assert sim.sample("FP16", 5)[2] == "benchmark · 1 stream"

    def test_unknown_variant_raises(self, sim):
        with pytest.raises(ValueError):
            sim.sample("NOPE")

    def test_sweep_scales_latency_with_concurrency(self, tmp_path):
        folder = tmp_path / "bench" / "FP16"
        folder.mkdir(parents=True)
        (folder / "c1.json").write_text(json.dumps({"mean_e2el_ms": 5000.0}))
        (folder / "c32.json").write_text(json.dumps({"median_e2el_ms": 9000.0}))
        (folder / "notes.json").write_text("{}")
        data = BenchmarkData(REPO_ROOT / "benchmark_results.json", tmp_path / "bench")
        assert data.latency_model("FP16", 1)[0] == 5000.0
        mean, _, basis = data.latency_model("FP16", 16)
        assert 5000.0 < mean < 9000.0 and basis == "benchmark c≈16"
        assert data.latency_model("FP16", 64)[0] == 9000.0

    def test_lognormal_sigma_edge_cases(self):
        assert lognormal_sigma(100, 90) == 0.05
        assert lognormal_sigma(0, 10) == 0.05
        assert 0 < lognormal_sigma(5000, 6000) < 1

    def test_missing_benchmark_file_is_not_fatal(self, tmp_path):
        data = BenchmarkData(tmp_path / "missing.json", tmp_path)
        assert data.has("FP16") is False
        assert data.gpus("FP16") == 2
