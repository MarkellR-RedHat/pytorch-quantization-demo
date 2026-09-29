"""Tests for the token bucket limiter"""

from app.ratelimit import RateLimiter


def test_burst_then_throttle_then_refill():
    limiter = RateLimiter(rate=4, burst=8)
    assert all(limiter.take("a", now=0.0) == 0 for _ in range(8))
    assert limiter.take("a", now=0.0) > 0
    assert limiter.take("b", now=0.0) == 0
    assert limiter.take("a", now=0.25) == 0


def test_retry_after_and_refund():
    limiter = RateLimiter(rate=0.25, burst=1)
    assert limiter.take("x", now=0.0) == 0
    assert limiter.take("x", now=0.0) == 4.0
    limiter.refund("x", now=0.0)
    assert limiter.take("x", now=0.0) == 0


def test_prunes_full_buckets():
    limiter = RateLimiter(rate=100, burst=1, max_keys=10)
    for i in range(50):
        limiter.take(str(i), now=0.0)
    limiter.take("late", now=100.0)
    assert len(limiter.buckets) <= 10
