"""Benchmark data that grounds simulation mode and the dashboard labels"""

import json
import logging
import math
import re
from itertools import pairwise
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GPUS = {"FP16": 2, "FP8": 1, "INT4": 1, "SPEC_DECODE": 2}
Z95 = 1.6448536269514722
SWEEP_LATENCY_KEYS = ("mean_e2el_ms", "median_e2el_ms")


def lognormal_sigma(mean: float, p95: float) -> float:
    """Sigma of a lognormal whose mean and 95th percentile match the benchmark."""
    if mean <= 0 or p95 <= mean:
        return 0.05
    d = math.log(p95 / mean)
    disc = Z95 * Z95 - 2 * d
    if disc <= 0:
        return Z95
    return Z95 - math.sqrt(disc)


def _interp(points: list[tuple[int, float]], n: float) -> float:
    if n <= points[0][0]:
        return points[0][1]
    for (n0, y0), (n1, y1) in pairwise(points):
        if n <= n1:
            return y0 + (y1 - y0) * (n - n0) / (n1 - n0)
    return points[-1][1]


class BenchmarkData:
    """benchmark_results.json plus optional `vllm bench serve` sweeps in bench/<VARIANT>/c<N>.json"""

    def __init__(self, path: Path, bench_dir: Path):
        self.path = Path(path)
        self.bench_dir = Path(bench_dir)
        self.reload()

    def reload(self):
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"Benchmark file {self.path} not loaded: {e}")
            raw = {}
        self.raw = raw if isinstance(raw, dict) else {}
        variants = self.raw.get("variants", {})
        self.variants: dict[str, dict] = variants if isinstance(variants, dict) else {}
        self.sweeps = {key: self._load_sweep(key) for key in DEFAULT_GPUS}

    def _load_sweep(self, key: str) -> list[tuple[int, float]]:
        folder = self.bench_dir / key
        points = []
        if not folder.is_dir():
            return points
        for f in folder.glob("c*.json"):
            m = re.fullmatch(r"c(\d+)\.json", f.name)
            if not m:
                continue
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                logger.warning(f"Skipping unreadable sweep file {f}")
                continue
            values = [data[k] for k in SWEEP_LATENCY_KEYS if isinstance(data.get(k), int | float)]
            latency = values[0] if values else None
            if latency:
                points.append((int(m.group(1)), float(latency)))
        return sorted(points)

    def meta(self, label=lambda key: key) -> dict:
        info = {k: self.raw.get(k) for k in ("model", "gpu", "vllm_version", "date", "notes")}
        info["variants"] = {key: self.summary(key, label(key)) for key in DEFAULT_GPUS if self.has(key)}
        info["load"] = {key: pts for key in DEFAULT_GPUS if (pts := self.load_points(key))}
        return info

    def load_points(self, key: str) -> list[dict]:
        """Concurrency sweep for the dashboard: output tokens/s (total and per GPU) and median latency."""
        folder = self.bench_dir / key
        if not folder.is_dir():
            return []
        gpus = self.gpus(key)
        points = []
        for f in folder.glob("c*.json"):
            m = re.fullmatch(r"c(\d+)\.json", f.name)
            if not m:
                continue
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            tput = data.get("output_throughput")
            if not isinstance(tput, int | float):
                continue
            found = [data[k] for k in SWEEP_LATENCY_KEYS if isinstance(data.get(k), int | float)]
            latency = found[0] if found else None
            points.append({
                "concurrency": int(m.group(1)),
                "output_tokens_per_second": round(float(tput), 1),
                "output_tokens_per_second_per_gpu": round(float(tput) / gpus, 1),
                "latency_ms": round(float(latency), 1) if latency else None,
            })
        return sorted(points, key=lambda p: p["concurrency"])

    def summary(self, key: str, label: str) -> dict:
        v = self.variant(key)
        gpus = self.gpus(key)
        tps = v.get("throughput_tps")
        return {
            "label": label,
            "gpus": gpus,
            "avg_latency_ms": v.get("avg_latency_ms"),
            "p95_latency_ms": v.get("p95_latency_ms"),
            "throughput_tps": tps,
            "tokens_per_second_per_gpu": round(tps / gpus, 2) if isinstance(tps, int | float) else None,
            "avg_tokens_per_request": v.get("avg_tokens_per_request"),
            "weights_gb": self.weights_gb(key),
        }

    def has(self, key: str) -> bool:
        return bool(self.variants.get(key, {}).get("avg_latency_ms"))

    def variant(self, key: str) -> dict:
        return self.variants.get(key, {})

    def gpus(self, key: str) -> int:
        return int(self.variant(key).get("gpus") or DEFAULT_GPUS.get(key, 1))

    def weights_gb(self, key: str) -> float | None:
        value = self.variant(key).get("weights_gb")
        return float(value) if isinstance(value, int | float) else None

    def latency_model(self, key: str, concurrency: int = 1) -> tuple[float, float, str]:
        """(mean latency ms, lognormal sigma, basis label) for a variant at the given concurrency."""
        v = self.variant(key)
        mean = float(v.get("avg_latency_ms") or 0.0)
        sigma = lognormal_sigma(mean, float(v.get("p95_latency_ms") or 0.0))
        sweep = self.sweeps.get(key)
        n = max(1, int(concurrency))
        if sweep:
            return _interp(sweep, n), sigma, f"benchmark c≈{n}"
        return mean, sigma, "benchmark · 1 stream"


benchmark_data = BenchmarkData(
    settings.resolve(settings.benchmark_file),
    settings.resolve(settings.bench_dir),
)
