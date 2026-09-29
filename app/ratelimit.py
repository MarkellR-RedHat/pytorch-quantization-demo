"""In-memory token bucket rate limiting (single replica by design)"""

import time


class RateLimiter:
    """Token bucket per key. take() returns 0 when allowed, else seconds until a token is available."""

    def __init__(self, rate: float, burst: float, max_keys: int = 20000):
        self.rate = rate
        self.burst = burst
        self.max_keys = max_keys
        self.buckets: dict[str, tuple[float, float]] = {}

    def _level(self, key: str, now: float) -> float:
        tokens, last = self.buckets.get(key, (self.burst, now))
        return min(self.burst, tokens + (now - last) * self.rate)

    def take(self, key: str, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        tokens = self._level(key, now)
        if tokens >= 1:
            self.buckets[key] = (tokens - 1, now)
            self._prune(now)
            return 0.0
        self.buckets[key] = (tokens, now)
        return (1 - tokens) / self.rate

    def refund(self, key: str, now: float | None = None):
        now = time.monotonic() if now is None else now
        self.buckets[key] = (min(self.burst, self._level(key, now) + 1), now)

    def reset(self):
        self.buckets.clear()

    def _prune(self, now: float):
        if len(self.buckets) <= self.max_keys:
            return
        full = [k for k in self.buckets if self._level(k, now) >= self.burst]
        for k in full:
            del self.buckets[k]
