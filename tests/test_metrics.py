"""Tests for metrics collection"""

import pytest
from app.metrics import MetricsCollector


class TestMetricsCollector:

    def test_initial_state(self):
        collector = MetricsCollector()
        assert collector.request_counts["FP16"] == 0
        assert collector.get_avg_latency("FP16") == 0.0

    def test_record_request(self):
        collector = MetricsCollector()
        collector.record_request("FP16", 95.0, 18.0, 0.0015)
        assert collector.request_counts["FP16"] == 1
        assert collector.get_avg_latency("FP16") == 95.0

    def test_multiple_requests(self):
        collector = MetricsCollector()
        collector.record_request("INT4", 40.0, 35.0, 0.0005)
        collector.record_request("INT4", 50.0, 30.0, 0.0005)
        assert collector.request_counts["INT4"] == 2
        assert collector.get_avg_latency("INT4") == 45.0

    def test_spec_decode_requests(self):
        collector = MetricsCollector()
        collector.record_request("SPEC_DECODE", 55.0, 30.0, 0.0007)
        assert collector.request_counts["SPEC_DECODE"] == 1
        assert collector.get_avg_latency("SPEC_DECODE") == 55.0

    def test_p95_latency(self):
        collector = MetricsCollector()
        for i in range(100):
            collector.record_request("FP16", float(i), 10.0, 0.0015)
        p95 = collector.get_p95_latency("FP16")
        assert p95 >= 90

    def test_all_snapshots_returns_three_variants(self):
        collector = MetricsCollector()
        snapshots = collector.get_all_snapshots()
        assert set(snapshots.keys()) == {"FP16", "INT4", "SPEC_DECODE"}

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_request("FP16", 95.0, 18.0, 0.0015)
        collector.reset()
        assert collector.request_counts["FP16"] == 0
        assert collector.get_avg_latency("FP16") == 0.0
