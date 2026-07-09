"""Tests for simulation mode"""

import pytest
from app.simulation import simulator


class TestSimulator:
    """Test simulator functionality"""

    def test_simulator_starts_disabled(self):
        """Simulator should start in disabled state"""
        sim = simulator
        assert sim.is_enabled() == False

    def test_enable_simulation(self):
        """Should enable simulation mode"""
        simulator.enable()
        assert simulator.is_enabled() == True
        simulator.disable()

    def test_disable_simulation(self):
        """Should disable simulation mode"""
        simulator.enable()
        simulator.disable()
        assert simulator.is_enabled() == False

    @pytest.mark.asyncio
    async def test_simulate_request(self):
        """Should generate synthetic request data"""
        simulator.enable()
        response_text, latency_ms, tps, cost = await simulator.simulate_request("FP16")

        assert isinstance(response_text, str)
        assert latency_ms > 0
        assert tps > 0
        assert cost > 0

        simulator.disable()

    def test_get_synthetic_metrics(self):
        """Should generate synthetic metrics"""
        metrics = simulator.get_synthetic_metrics("FP16", 50)

        assert "requests_per_second" in metrics
        assert "avg_latency_ms" in metrics
        assert "gpu_memory_gb" in metrics
        assert metrics["requests_per_second"] > 0
