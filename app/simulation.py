"""Simulation mode for failsafe demo execution"""

import random
import time
from typing import Tuple
from app.models import SimulationConfig


class Simulator:
    """Generates synthetic metrics for simulation mode"""

    def __init__(self):
        self.config = SimulationConfig(enabled=False)
        self.baseline_metrics = {
            "FP32": {"latency": 150, "throughput": 10, "memory": 80, "cost": 0.0020},
            "FP16": {"latency": 75, "throughput": 20, "memory": 40, "cost": 0.0015},
            "INT8": {"latency": 50, "throughput": 30, "memory": 20, "cost": 0.0010},
            "INT4": {"latency": 30, "throughput": 40, "memory": 10, "cost": 0.0005},
        }

    def enable(self, request_rate: float = 10.0, synthetic_users: int = 75):
        """Enable simulation mode"""
        self.config.enabled = True
        self.config.request_rate = request_rate
        self.config.synthetic_users = synthetic_users

    def disable(self):
        """Disable simulation mode"""
        self.config.enabled = False

    def is_enabled(self) -> bool:
        """Check if simulation mode is active"""
        return self.config.enabled

    async def simulate_request(
        self,
        model_type: str,
        prompt: str = "Hello, how are you?"
    ) -> Tuple[str, float, float, float]:
        """
        Simulate an inference request with synthetic data

        Args:
            model_type: One of FP32, FP16, INT8, INT4
            prompt: Text prompt (used for response variety)

        Returns:
            Tuple of (response_text, latency_ms, tokens_per_second, cost)
        """
        baseline = self.baseline_metrics.get(model_type, self.baseline_metrics["FP16"])

        # Add realistic variation
        variation = self.config.latency_variation
        latency_ms = baseline["latency"] * (1 + random.uniform(-variation, variation))

        # Simulate processing delay
        await self._simulate_delay(latency_ms)

        # Generate synthetic response
        responses = [
            "This is a simulated response from the model.",
            "Simulation mode is active. This response is synthetic.",
            "In a real demo, this would be actual model output.",
            "Quantization affects both speed and quality of responses.",
            "Lower precision models run faster but may have quality trade-offs."
        ]
        response_text = random.choice(responses)

        # Calculate synthetic metrics
        tokens_per_second = baseline["throughput"] * (1 + random.uniform(-0.1, 0.1))
        cost = baseline["cost"] * (1 + random.uniform(-0.05, 0.05))

        return response_text, latency_ms, tokens_per_second, cost

    async def _simulate_delay(self, latency_ms: float):
        """Simulate processing delay to make it realistic"""
        import asyncio
        # Use a fraction of actual latency to keep demo responsive
        await asyncio.sleep(latency_ms / 1000 * 0.1)

    def get_synthetic_metrics(self, model_type: str, request_count: int) -> dict:
        """
        Generate synthetic real-time metrics

        Args:
            model_type: One of FP32, FP16, INT8, INT4
            request_count: Current request count for this model

        Returns:
            Dictionary of metrics
        """
        baseline = self.baseline_metrics.get(model_type, self.baseline_metrics["FP16"])

        # Simulate load-based variations
        load_factor = min(request_count / 100, 2.0)  # More load = higher latency
        variation = self.config.latency_variation

        return {
            "requests_per_second": baseline["throughput"] / (1 + load_factor * 0.3),
            "avg_latency_ms": baseline["latency"] * (1 + load_factor * 0.5),
            "p95_latency_ms": baseline["latency"] * (1 + load_factor * 0.7) * 1.2,
            "gpu_memory_gb": baseline["memory"] * (1 + load_factor * 0.2),
            "tokens_per_second": baseline["throughput"] * (1 + random.uniform(-0.1, 0.1)),
            "cost_per_request": baseline["cost"],
            "queue_depth": max(0, int(request_count * 0.1 * random.random())),
            "active_requests": max(1, int(request_count * 0.05)),
        }


# Global simulator instance
simulator = Simulator()
