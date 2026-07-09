"""Metrics aggregation and tracking"""

import time
from collections import defaultdict, deque
from typing import Dict, List
from datetime import datetime
from app.models import MetricsSnapshot


class MetricsCollector:
    """Collects and aggregates metrics for all model variants"""

    def __init__(self, window_size: int = 60):
        """
        Initialize metrics collector

        Args:
            window_size: Number of seconds to keep in rolling window
        """
        self.window_size = window_size
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, deque] = {
            model: deque(maxlen=100) for model in ["FP32", "FP16", "INT8", "INT4"]
        }
        self.request_times: Dict[str, deque] = {
            model: deque(maxlen=1000) for model in ["FP32", "FP16", "INT8", "INT4"]
        }
        self.costs: Dict[str, List[float]] = defaultdict(list)
        self.tokens_per_sec: Dict[str, deque] = {
            model: deque(maxlen=100) for model in ["FP32", "FP16", "INT8", "INT4"]
        }
        self.active_requests: Dict[str, int] = defaultdict(int)

    def record_request(
        self,
        model_type: str,
        latency_ms: float,
        tokens_per_second: float,
        cost: float
    ):
        """Record a completed request"""
        self.request_counts[model_type] += 1
        self.latencies[model_type].append(latency_ms)
        self.request_times[model_type].append(time.time())
        self.tokens_per_sec[model_type].append(tokens_per_second)
        self.costs[model_type].append(cost)

    def increment_active(self, model_type: str):
        """Increment active request count"""
        self.active_requests[model_type] += 1

    def decrement_active(self, model_type: str):
        """Decrement active request count"""
        self.active_requests[model_type] = max(0, self.active_requests[model_type] - 1)

    def get_requests_per_second(self, model_type: str) -> float:
        """Calculate requests per second over the window"""
        times = list(self.request_times[model_type])
        if not times:
            return 0.0

        current_time = time.time()
        recent_times = [t for t in times if current_time - t <= self.window_size]

        if not recent_times:
            return 0.0

        return len(recent_times) / self.window_size

    def get_avg_latency(self, model_type: str) -> float:
        """Get average latency"""
        latencies = list(self.latencies[model_type])
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def get_p95_latency(self, model_type: str) -> float:
        """Get 95th percentile latency"""
        latencies = sorted(list(self.latencies[model_type]))
        if not latencies:
            return 0.0
        index = int(len(latencies) * 0.95)
        return latencies[min(index, len(latencies) - 1)]

    def get_avg_tokens_per_sec(self, model_type: str) -> float:
        """Get average tokens per second"""
        tps = list(self.tokens_per_sec[model_type])
        if not tps:
            return 0.0
        return sum(tps) / len(tps)

    def get_avg_cost(self, model_type: str) -> float:
        """Get average cost per request"""
        costs = self.costs[model_type][-100:]  # Last 100 requests
        if not costs:
            return 0.0
        return sum(costs) / len(costs)

    def get_snapshot(self, model_type: str, gpu_memory: float = 0.0) -> MetricsSnapshot:
        """Get current metrics snapshot for a model variant"""
        return MetricsSnapshot(
            model_type=model_type,
            requests_per_second=self.get_requests_per_second(model_type),
            avg_latency_ms=self.get_avg_latency(model_type),
            p95_latency_ms=self.get_p95_latency(model_type),
            gpu_memory_gb=gpu_memory,
            tokens_per_second=self.get_avg_tokens_per_sec(model_type),
            cost_per_request=self.get_avg_cost(model_type),
            queue_depth=0,  # Would need to query from model server
            active_requests=self.active_requests[model_type],
            total_requests=self.request_counts[model_type]
        )

    def get_all_snapshots(self) -> Dict[str, MetricsSnapshot]:
        """Get snapshots for all model variants"""
        # Simulated GPU memory usage (would be real in production)
        gpu_memory = {
            "FP32": 80.0,
            "FP16": 40.0,
            "INT8": 20.0,
            "INT4": 10.0
        }

        return {
            model: self.get_snapshot(model, gpu_memory[model])
            for model in ["FP32", "FP16", "INT8", "INT4"]
        }

    def reset(self):
        """Reset all metrics"""
        self.request_counts.clear()
        for deq in self.latencies.values():
            deq.clear()
        for deq in self.request_times.values():
            deq.clear()
        self.costs.clear()
        for deq in self.tokens_per_sec.values():
            deq.clear()
        self.active_requests.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()
