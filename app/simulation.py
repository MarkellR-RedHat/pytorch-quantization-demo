"""Simulation mode for failsafe demo execution"""

import random
import asyncio
from typing import Tuple
from app.models import SimulationConfig


QUALITY_PROMPTS = {
    "complex_reasoning": {
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
        "FP16": "The farmer has 9 sheep left.\n\nStep 1: The farmer starts with 17 sheep.\nStep 2: The phrase 'all but 9' means every sheep except 9 ran away.\nStep 3: So 17 - 9 = 8 sheep ran away.\nStep 4: The 9 sheep that didn't run away are still with the farmer.\n\nThe answer is 9 sheep. The tricky part is that 'all but 9' means 9 remain, not that 9 ran away.",
        "INT4": "The farmer has 8 sheep left.\n\nStep 1: The farmer starts with 17 sheep.\nStep 2: 9 sheep run away.\nStep 3: 17 - 9 = 8 sheep remain.\n\nThe answer is 8 sheep.",
        "SPEC_DECODE": "The farmer has 9 sheep left.\n\nStep 1: The farmer starts with 17 sheep.\nStep 2: 'All but 9 run away' means every sheep except 9 ran away.\nStep 3: So 17 - 9 = 8 sheep ran away, leaving 9 behind.\nStep 4: The farmer still has the 9 sheep that stayed.\n\nThe answer is 9. The phrasing is intentionally misleading: 'all but 9' describes how many remain, not how many left.",
    },
    "code_generation": {
        "prompt": "Write a Python function that returns the second largest number in a list. Handle edge cases.",
        "FP16": "```python\ndef second_largest(nums):\n    if len(nums) < 2:\n        raise ValueError(\"Need at least two distinct values\")\n    unique = sorted(set(nums), reverse=True)\n    if len(unique) < 2:\n        raise ValueError(\"Need at least two distinct values\")\n    return unique[1]\n```\n\nThis handles duplicates by using `set()` first, then sorts in descending order. It raises an error if there aren't at least two distinct values in the list.",
        "INT4": "```python\ndef second_largest(nums):\n    nums.sort()\n    return nums[-2]\n```\n\nSort the list and return the second to last element.",
        "SPEC_DECODE": "```python\ndef second_largest(nums):\n    if len(nums) < 2:\n        raise ValueError(\"Need at least two distinct values\")\n    unique = sorted(set(nums), reverse=True)\n    if len(unique) < 2:\n        raise ValueError(\"Need at least two distinct values\")\n    return unique[1]\n```\n\nUsing `set()` removes duplicates so `[5, 5, 3]` correctly returns `3`. Raises `ValueError` when there aren't enough distinct values to have a meaningful second largest.",
    },
    "summarization": {
        "prompt": "Summarize the key trade-offs of model quantization for production LLM deployments in 3 bullet points.",
        "FP16": "- **Speed vs. Quality**: Lower precision (INT8/INT4) increases throughput and reduces latency, but aggressive quantization can degrade output quality on complex reasoning, nuanced instructions, and multi-step tasks.\n- **Cost vs. Capability**: Quantized models need less GPU memory and compute, cutting infrastructure costs by 2-8x, but may require additional validation pipelines to catch quality regressions before they affect users.\n- **Simplicity vs. Optimization**: Full precision models work out of the box, while quantized deployments require careful calibration, evaluation datasets, and ongoing monitoring to ensure the quality-performance trade-off stays within acceptable bounds.",
        "INT4": "- Speed vs Quality: Smaller models run faster but may lose accuracy\n- Cost vs Capability: Less GPU memory needed, lower costs\n- Simplicity vs Optimization: Quantization requires extra setup",
        "SPEC_DECODE": "- **Speed vs. Quality**: Lower-precision formats (INT8/INT4) boost throughput and cut latency, but aggressive quantization can hurt output quality on tasks that demand nuanced reasoning or multi-step logic.\n- **Cost vs. Capability**: Quantized models use significantly less GPU memory and compute, reducing serving costs by 2-8x. The trade-off is that teams need robust evaluation pipelines to detect regressions before they reach production.\n- **Simplicity vs. Optimization**: Full-precision models deploy with minimal tuning, while quantization requires calibration datasets, quality benchmarks, and ongoing monitoring to keep the performance-quality balance in check.",
    }
}


class Simulator:

    def __init__(self):
        self.config = SimulationConfig(enabled=False)
        self.baseline_metrics = {
            "FP16": {"latency": 95, "throughput": 18, "memory": 40, "cost": 0.0015},
            "INT4": {"latency": 45, "throughput": 35, "memory": 10, "cost": 0.0005},
            "SPEC_DECODE": {"latency": 55, "throughput": 30, "memory": 25, "cost": 0.0007},
        }

    def enable(self, request_rate: float = 10.0, synthetic_users: int = 75):
        self.config.enabled = True
        self.config.request_rate = request_rate
        self.config.synthetic_users = synthetic_users

    def disable(self):
        self.config.enabled = False

    def is_enabled(self) -> bool:
        return self.config.enabled

    async def simulate_request(
        self,
        model_type: str,
        prompt: str = "Hello, how are you?"
    ) -> Tuple[str, float, float, float]:
        baseline = self.baseline_metrics.get(model_type, self.baseline_metrics["FP16"])

        variation = self.config.latency_variation
        latency_ms = baseline["latency"] * (1 + random.uniform(-variation, variation))

        await self._simulate_delay(latency_ms)

        responses = [
            "This is a simulated response from the model.",
            "Simulation mode is active. This response is synthetic.",
            "In a real demo, this would be actual model output.",
            "Quantization affects both speed and quality of responses.",
            "Speculative decoding pairs a small draft model with a large verifier.",
        ]
        response_text = random.choice(responses)

        tokens_per_second = baseline["throughput"] * (1 + random.uniform(-0.1, 0.1))
        cost = baseline["cost"] * (1 + random.uniform(-0.05, 0.05))

        return response_text, latency_ms, tokens_per_second, cost

    async def _simulate_delay(self, latency_ms: float):
        await asyncio.sleep(latency_ms / 1000 * 0.1)

    def get_synthetic_metrics(self, model_type: str, request_count: int) -> dict:
        baseline = self.baseline_metrics.get(model_type, self.baseline_metrics["FP16"])

        load_factor = min(request_count / 100, 2.0)
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

    def get_quality_comparison(self, scenario: str = "complex_reasoning") -> dict:
        """Return pre-scripted quality comparison for the given scenario"""
        data = QUALITY_PROMPTS.get(scenario, QUALITY_PROMPTS["complex_reasoning"])
        return {
            "prompt": data["prompt"],
            "responses": {
                "FP16": data["FP16"],
                "INT4": data["INT4"],
                "SPEC_DECODE": data["SPEC_DECODE"],
            }
        }


simulator = Simulator()
