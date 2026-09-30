"""Simulation mode: every number is sampled from the recorded H200 benchmark"""

import asyncio
import math
import random

from app.benchmark import BenchmarkData, benchmark_data
from app.config import settings

SIM_RESPONSES = [
    "[simulated] Latency and token count for this reply are sampled from the recorded H200 benchmark.",
    "[simulated] No model was called. Switch to live mode to hit the vLLM endpoints.",
]


class Simulator:

    def __init__(self, benchmark: BenchmarkData | None = None):
        self.benchmark = benchmark or benchmark_data
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def sample(self, model_type: str, concurrency: int = 1, rng=random) -> tuple[float, int, str]:
        """(latency ms, completion tokens, basis) drawn from the benchmark for this variant."""
        mean, sigma, basis = self.benchmark.latency_model(model_type, concurrency)
        if mean <= 0:
            raise ValueError(f"No benchmark data for {model_type}")
        mu = math.log(mean) - sigma * sigma / 2
        latency_ms = rng.lognormvariate(mu, sigma)
        avg_tokens = float(self.benchmark.variant(model_type).get("avg_tokens_per_request") or 256)
        tokens = max(1, round(avg_tokens * rng.uniform(0.9, 1.1)))
        return latency_ms, tokens, basis

    async def simulate_request(
        self,
        model_type: str,
        prompt: str = "",
        concurrency: int = 1,
    ) -> tuple[str, float, float, int]:
        latency_ms, tokens, _ = self.sample(model_type, concurrency)
        if settings.sim_time_scale > 0:
            await asyncio.sleep(latency_ms / 1000 * settings.sim_time_scale)
        tokens_per_second = tokens / (latency_ms / 1000)
        return random.choice(SIM_RESPONSES), latency_ms, tokens_per_second, tokens


simulator = Simulator()


def _follow_track(track) -> None:
    simulator.benchmark = track.benchmark


from app import tracks  # noqa: E402 - registered after the simulator exists

tracks.on_change(_follow_track)
tracks.select_from_settings()
