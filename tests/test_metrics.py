"""Tests for metrics collection"""

import pytest
from app.metrics import MetricsCollector


class TestMetricsCollector:
    """Test metrics collection functionality"""

    def test_initial_state(self):
        """Metrics should start at zero"""
        collector = MetricsCollector()
        assert collector.request_counts["FP32"] == 0
        assert collector.get_avg_latency("FP32") == 0.0

    def test_record_request(self):
        """Should record request metrics"""
        collector = MetricsCollector()
        collector.record_request("FP16", 75.0, 25.0, 0.0015)

        assert collector.request_counts["FP16"] == 1
        assert collector.get_avg_latency("FP16") == 75.0

    def test_multiple_requests(self):
        """Should aggregate multiple requests"""
        collector = MetricsCollector()
        collector.record_request("INT8", 50.0, 30.0, 0.001)
        collector.record_request("INT8", 60.0, 35.0, 0.001)

        assert collector.request_counts["INT8"] == 2
        assert collector.get_avg_latency("INT8") == 55.0

    def test_p95_latency(self):
        """Should calculate P95 latency"""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_request("FP32", float(i), 10.0, 0.002)

        p95 = collector.get_p95_latency("FP32")
        assert p95 >= 90  # Should be around 95

    def test_reset(self):
        """Should reset all metrics"""
        collector = MetricsCollector()
        collector.record_request("FP16", 75.0, 25.0, 0.0015)
        collector.reset()

        assert collector.request_counts["FP16"] == 0
        assert collector.get_avg_latency("FP16") == 0.0
