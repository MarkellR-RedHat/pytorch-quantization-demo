"""Tests for metrics collection"""

from app.metrics import MetricsCollector, nearest_rank


class TestMetricsCollector:

    def test_initial_state(self):
        collector = MetricsCollector()
        assert collector.request_counts["BF16"] == 0
        assert collector.get_avg_latency("BF16") == 0.0
        assert collector.get_p95_latency("BF16") == 0.0

    def test_record_request(self):
        collector = MetricsCollector()
        collector.record_request("BF16", 5000.0, 46.0, completion_tokens=230)
        assert collector.request_counts["BF16"] == 1
        assert collector.get_avg_latency("BF16") == 5000.0

    def test_multiple_requests(self):
        collector = MetricsCollector()
        collector.record_request("INT4", 4000.0, 50.0)
        collector.record_request("INT4", 6000.0, 40.0)
        assert collector.request_counts["INT4"] == 2
        assert collector.get_avg_latency("INT4") == 5000.0
        assert collector.get_avg_tokens_per_sec("INT4") == 45.0

    def test_nearest_rank_percentiles(self):
        values = [float(i) for i in range(1, 101)]
        assert nearest_rank(values, 50) == 50.0
        assert nearest_rank(values, 95) == 95.0
        assert nearest_rank([7.0], 95) == 7.0

    def test_p95_of_twenty_samples_is_nineteenth(self):
        collector = MetricsCollector()
        for i in range(1, 21):
            collector.record_request("BF16", float(i), 10.0)
        assert collector.get_p95_latency("BF16") == 19.0

    def test_rate_uses_elapsed_time_not_fixed_window(self):
        collector = MetricsCollector(window_s=60)
        collector.started -= 10  # demo has run for 10 s
        for _ in range(20):
            collector.record_request("BF16", 100.0, 10.0, completion_tokens=100)
        assert 1.9 < collector.get_requests_per_second("BF16") <= 2.0
        assert 190 < collector.get_output_tokens_per_second("BF16") <= 200

    def test_snapshot_per_gpu_and_hidden_cost(self):
        collector = MetricsCollector()
        collector.record_request("BF16", 5000.0, 48.0, cost=0.01)
        snap = collector.get_snapshot("BF16", label="BF16", gpus=2)
        assert snap.label == "BF16"
        assert snap.tokens_per_second_per_gpu == 24.0
        assert snap.cost_per_request is None
        assert collector.get_snapshot("BF16", include_cost=True).cost_per_request == 0.01

    def test_errors_and_in_flight(self):
        collector = MetricsCollector()
        collector.increment_active("INT4")
        collector.record_error("INT4")
        snap = collector.get_snapshot("INT4")
        assert snap.in_flight == 1
        assert snap.errors == 1
        collector.decrement_active("INT4")
        collector.decrement_active("INT4")
        assert collector.get_snapshot("INT4").in_flight == 0

    def test_all_snapshots_default_variants(self):
        snapshots = MetricsCollector().get_all_snapshots()
        assert set(snapshots) == {"BF16", "INT4", "SPEC_DECODE"}

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_request("BF16", 95.0, 18.0)
        collector.record_error("BF16")
        collector.reset()
        assert collector.request_counts["BF16"] == 0
        assert collector.error_counts["BF16"] == 0
        assert collector.get_avg_latency("BF16") == 0.0
