"""Shared fixtures: every test starts from a clean, fast, simulated app"""

import pytest

from app import main
from app.config import settings


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(settings, "sim_time_scale", 0.0)
    monkeypatch.setattr(settings, "presenter_key", "")
    monkeypatch.setattr(settings, "gpu_hourly_usd", 0.0)
    monkeypatch.setattr(settings, "auto_traffic_rps", 0.5)
    main.simulator.enable()
    main.demo_state.is_running = False
    main.demo_state.start_time = None
    main.metrics_collector.reset()
    main.arena_votes.reset()
    main.arena_relay.reset()
    for limiter in (
        main.request_limiter,
        main.ip_limiter,
        main.vote_limiter,
        main.hard_client_limiter,
        main.hard_global_limiter,
    ):
        limiter.reset()
    yield
    main.stop_auto_traffic()
    main.demo_state.is_running = False
