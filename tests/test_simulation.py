"""Tests for simulation mode"""

import pytest
from app.simulation import Simulator


class TestSimulator:

    def setup_method(self):
        self.sim = Simulator()

    def test_simulator_starts_disabled(self):
        assert self.sim.is_enabled() is False

    def test_enable_simulation(self):
        self.sim.enable()
        assert self.sim.is_enabled() is True

    def test_disable_simulation(self):
        self.sim.enable()
        self.sim.disable()
        assert self.sim.is_enabled() is False

    @pytest.mark.asyncio
    async def test_simulate_request_fp16(self):
        self.sim.enable()
        response_text, latency_ms, tps, cost = await self.sim.simulate_request("FP16")
        assert isinstance(response_text, str)
        assert latency_ms > 0
        assert tps > 0
        assert cost > 0

    @pytest.mark.asyncio
    async def test_simulate_request_int4(self):
        self.sim.enable()
        response_text, latency_ms, tps, cost = await self.sim.simulate_request("INT4")
        assert latency_ms > 0
        assert cost > 0

    @pytest.mark.asyncio
    async def test_simulate_request_spec_decode(self):
        self.sim.enable()
        response_text, latency_ms, tps, cost = await self.sim.simulate_request("SPEC_DECODE")
        assert latency_ms > 0
        assert cost > 0

    def test_get_synthetic_metrics_all_variants(self):
        for variant in ["FP16", "INT4", "SPEC_DECODE"]:
            metrics = self.sim.get_synthetic_metrics(variant, 50)
            assert "requests_per_second" in metrics
            assert "avg_latency_ms" in metrics
            assert "gpu_memory_gb" in metrics
            assert metrics["requests_per_second"] > 0

    def test_spec_decode_latency_between_fp16_and_int4(self):
        baseline = self.sim.baseline_metrics
        assert baseline["INT4"]["latency"] < baseline["SPEC_DECODE"]["latency"]
        assert baseline["SPEC_DECODE"]["latency"] < baseline["FP16"]["latency"]

    def test_spec_decode_memory_between_fp16_and_int4(self):
        baseline = self.sim.baseline_metrics
        assert baseline["INT4"]["memory"] < baseline["SPEC_DECODE"]["memory"]
        assert baseline["SPEC_DECODE"]["memory"] < baseline["FP16"]["memory"]

    def test_quality_comparison_returns_all_variants(self):
        result = self.sim.get_quality_comparison("complex_reasoning")
        assert "prompt" in result
        assert "FP16" in result["responses"]
        assert "INT4" in result["responses"]
        assert "SPEC_DECODE" in result["responses"]

    def test_quality_comparison_all_scenarios(self):
        for scenario in ["complex_reasoning", "code_generation", "summarization"]:
            result = self.sim.get_quality_comparison(scenario)
            assert len(result["responses"]) == 3
