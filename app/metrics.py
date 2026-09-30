"""Metrics aggregation and tracking"""

import math
import time
from collections import defaultdict, deque

from app.config import ALL_VARIANTS, BASE_VARIANTS
from app.models import MetricsSnapshot

SAMPLE_SIZE = 200


def nearest_rank(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


class MetricsCollector:

    def __init__(self, window_s: float = 60.0):
        self.window_s = window_s
        self.reset()

    def reset(self):
        self.started = time.monotonic()
        self.request_counts: dict[str, int] = defaultdict(int)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.active_requests: dict[str, int] = defaultdict(int)
        self.latencies = {k: deque(maxlen=SAMPLE_SIZE) for k in ALL_VARIANTS}
        self.tokens_per_sec = {k: deque(maxlen=SAMPLE_SIZE) for k in ALL_VARIANTS}
        self.costs = {k: deque(maxlen=SAMPLE_SIZE) for k in ALL_VARIANTS}
        # (timestamp, completion tokens) for rate and aggregate throughput over the window
        self.completions = {k: deque(maxlen=10000) for k in ALL_VARIANTS}

    def record_request(
        self,
        model_type: str,
        latency_ms: float,
        tokens_per_second: float,
        cost: float | None = None,
        completion_tokens: int = 0,
    ):
        self.request_counts[model_type] += 1
        self.latencies[model_type].append(latency_ms)
        self.tokens_per_sec[model_type].append(tokens_per_second)
        if cost is not None:
            self.costs[model_type].append(cost)
        self.completions[model_type].append((time.monotonic(), completion_tokens))

    def record_error(self, model_type: str):
        self.error_counts[model_type] += 1

    def increment_active(self, model_type: str):
        self.active_requests[model_type] += 1

    def decrement_active(self, model_type: str):
        self.active_requests[model_type] = max(0, self.active_requests[model_type] - 1)

    def _window(self, model_type: str) -> tuple[list[tuple[float, int]], float]:
        now = time.monotonic()
        recent = [c for c in self.completions[model_type] if now - c[0] <= self.window_s]
        elapsed = min(self.window_s, now - self.started)
        return recent, max(elapsed, 1.0)

    def get_requests_per_second(self, model_type: str) -> float:
        """Completed requests per second over the last window (or since reset, if shorter)."""
        recent, elapsed = self._window(model_type)
        return len(recent) / elapsed

    def get_output_tokens_per_second(self, model_type: str) -> float:
        """Aggregate completion tokens per second across all streams over the window."""
        recent, elapsed = self._window(model_type)
        return sum(tokens for _, tokens in recent) / elapsed

    def get_avg_latency(self, model_type: str) -> float:
        values = self.latencies[model_type]
        return sum(values) / len(values) if values else 0.0

    def get_p50_latency(self, model_type: str) -> float:
        return nearest_rank(list(self.latencies[model_type]), 50)

    def get_p95_latency(self, model_type: str) -> float:
        return nearest_rank(list(self.latencies[model_type]), 95)

    def get_avg_tokens_per_sec(self, model_type: str) -> float:
        values = self.tokens_per_sec[model_type]
        return sum(values) / len(values) if values else 0.0

    def get_avg_cost(self, model_type: str) -> float | None:
        values = self.costs[model_type]
        return sum(values) / len(values) if values else None

    def get_snapshot(
        self,
        model_type: str,
        *,
        label: str | None = None,
        source: str = "simulated",
        basis: str = "",
        gpus: int = 1,
        weights_gib_per_gpu: float | None = None,
        include_cost: bool = False,
    ) -> MetricsSnapshot:
        tps = self.get_avg_tokens_per_sec(model_type)
        return MetricsSnapshot(
            model_type=model_type,
            label=label or model_type,
            source=source,
            basis=basis,
            gpus=gpus,
            weights_gib_per_gpu=weights_gib_per_gpu,
            requests_per_second=self.get_requests_per_second(model_type),
            avg_latency_ms=self.get_avg_latency(model_type),
            p50_latency_ms=self.get_p50_latency(model_type),
            p95_latency_ms=self.get_p95_latency(model_type),
            tokens_per_second=tps,
            tokens_per_second_per_gpu=tps / max(gpus, 1),
            output_tokens_per_second=self.get_output_tokens_per_second(model_type),
            cost_per_request=self.get_avg_cost(model_type) if include_cost else None,
            in_flight=self.active_requests[model_type],
            total_requests=self.request_counts[model_type],
            errors=self.error_counts[model_type],
        )

    def get_all_snapshots(self, variants: dict[str, dict] | None = None) -> dict[str, MetricsSnapshot]:
        """Snapshots keyed by variant; `variants` maps key -> get_snapshot keyword arguments."""
        variants = variants if variants is not None else {k: {} for k in BASE_VARIANTS}
        return {key: self.get_snapshot(key, **info) for key, info in variants.items()}

    def total_requests(self) -> int:
        return sum(self.request_counts.values())


metrics_collector = MetricsCollector()
