"""Benchmark data that grounds simulation mode and the dashboard labels"""

import json
import logging
import math
import re
from itertools import pairwise
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GPUS = {"BF16": 2, "FP8": 1, "INT4": 1, "SPEC_DECODE": 2}
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


def _num(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _round(value) -> float | None:
    return round(float(value), 1) if _num(value) else None


def at_target(points: list[dict], target_ms: float) -> dict | None:
    """The most output tokens/s per GPU a setup delivered while its tail time per output token stayed
    under the target: the number that sets cost, since a GPU is only as useful as the load it can take
    without users noticing."""
    ok = [p for p in points if p.get("tpot_tail_ms") is not None and p["tpot_tail_ms"] <= target_ms]
    if not ok:
        return None
    # a point whose run shared the pod with another sweep is a lower bound (the GPU was serving both);
    # a clean point wins when there is one, otherwise the best lower bound, labeled as one
    clean = [p for p in ok if not p.get("overlapped_with")]
    best = max(clean or ok, key=lambda p: p["output_tokens_per_second_per_gpu"])
    return {
        "concurrency": best["concurrency"],
        "output_tokens_per_second_per_gpu": best["output_tokens_per_second_per_gpu"],
        "output_tokens_per_second_per_h200": best.get("output_tokens_per_second_per_h200"),
        "slices_per_h200": best.get("slices_per_h200", 1),
        "tpot_tail_ms": best["tpot_tail_ms"],
        "tpot_tail_kind": best["tpot_tail_kind"],
        "lower_bound": bool(best.get("overlapped_with")),
        "overlapped_with": best.get("overlapped_with") or [],
    }


def _per_request(data: dict, key: str, done: int) -> int | None:
    value = data.get(key)
    return round(value / done) if done and isinstance(value, int | float) else None


class BenchmarkData:
    """benchmark_results.json plus the `vllm bench serve` sweeps under bench_dir/<VARIANT>/c<N>.json"""

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

    def sweep_folder(self, key: str) -> Path:
        # a column can point at another folder's sweep (the INT4 column shows Red Hat's build)
        return self.bench_dir / (self.variant(key).get("sweep_dir") or key)

    def _load_sweep(self, key: str) -> list[tuple[int, float]]:
        folder = self.sweep_folder(key)
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
        target = info["tpot_target_ms"] = settings.tpot_target_ms
        info["at_target"] = {k: best for k, pts in info["load"].items() if (best := at_target(pts, target))}
        return info

    def load_points(self, key: str) -> list[dict]:
        """Concurrency sweep for the dashboard: output tokens/s (total and per GPU) and median latency."""
        folder = self.sweep_folder(key)
        if not folder.is_dir():
            return []
        gpus = self.gpus(key)
        slices = self.slices_per_h200(key)
        runs = self.variant(key).get("sweep_runs") or {}
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
            keys = ("median_e2el_ms", "mean_e2el_ms")
            found = [data[k] for k in keys if isinstance(data.get(k), int | float)]
            latency = found[0] if found else None
            done = data.get("completed") or 0
            tail_kind = next((k for k in ("p95", "p99") if _num(data.get(f"{k}_tpot_ms"))), None)
            points.append({
                "concurrency": int(m.group(1)),
                "output_tokens_per_second": round(float(tput), 1),
                "output_tokens_per_second_per_gpu": round(float(tput) / gpus, 1),
                # per device times the devices per card: arithmetic, not a measurement (equal to per GPU
                # on a full card); the dashboard labels it so and never puts it in a ratio with per-slice
                "output_tokens_per_second_per_h200": round(float(tput) / gpus * slices, 1),
                "slices_per_h200": slices,
                "latency_ms": round(float(latency), 1) if latency else None,
                "latency_kind": "median" if isinstance(data.get("median_e2el_ms"), int | float) else "mean",
                "latency_p95_ms": _round(data.get("p95_e2el_ms")),
                "avg_input_tokens": _per_request(data, "total_input_tokens", done),
                "avg_output_tokens": _per_request(data, "total_output_tokens", done),
                "tpot_median_ms": _round(data.get("median_tpot_ms")),
                "tpot_tail_ms": _round(data.get(f"{tail_kind}_tpot_ms")) if tail_kind else None,
                "tpot_tail_kind": tail_kind,
                # other vllm bench serve runs that shared the pod during this one (from the file's window)
                "overlapped_with": (runs.get(f.name) or {}).get("overlapped_with") or [],
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
            "slices_per_h200": self.slices_per_h200(key),
            "per_h200": v.get("per_h200"),
            "device": v.get("device"),
            "avg_tokens_per_request": v.get("avg_tokens_per_request"),
            "weights_gib_per_gpu": self.weights_gib_per_gpu(key),
            "mean_acceptance_length": v.get("mean_acceptance_length"),
            "num_speculative_tokens": v.get("num_speculative_tokens"),
            "spec_tokens_measured": v.get("spec_tokens_measured"),
            # single-stream speed at every other number of speculative tokens the run tried
            **{k: val for k, val in v.items() if re.fullmatch(r"throughput_tps_k\d+", k)},
            # round-2 fields, passed through as they are so every number on screen traces to the file
            "ttft_ms_avg": v.get("ttft_ms_avg"),
            "speed_vs_baseline": v.get("speed_vs_baseline"),
            "speed_vs_baseline_t0_7": v.get("speed_vs_baseline_t0_7"),
            "throughput_tps_t0_7": (v.get("at_temperature_0_7") or {}).get("throughput_tps"),
            "prompts": v.get("prompts"),
            "checkpoint": v.get("checkpoint"),
            "build": v.get("build"),
            "kernel": v.get("kernel"),
            "accuracy": v.get("accuracy"),
            "acceptance": v.get("acceptance"),
            "reference": self._reference(v.get("reference")),
            "int4_35": self._footnote(v.get("int4_35")),
            "enforce_eager": v.get("enforce_eager"),
            "kv_cache_tokens": v.get("kv_cache_tokens"),
            "overlap_note": v.get("overlap_note"),
        }

    @staticmethod
    def _footnote(run: dict | None) -> dict | None:
        """The Qwen INT4 card's footnote: the same checkpoint on a 35 GB slice."""
        if not run:
            return None
        return {
            "device": run.get("device"),
            "throughput_tps": run.get("throughput_tps"),
            "speed_vs_baseline": run.get("speed_vs_baseline"),
            "weights_gib_per_gpu": run.get("weights_gib_per_gpu"),
            "kv_cache_tokens": run.get("kv_cache_tokens"),
            "per_h200": run.get("per_h200"),
            "enforce_eager": run.get("enforce_eager"),
            "note": run.get("note"),
        }

    @staticmethod
    def _reference(ref: dict | None) -> dict | None:
        """The naive pick's numbers, shown as one line under the setup that replaced it."""
        if not ref:
            return None
        acc = ref.get("accuracy") or {}
        return {
            "build": ref.get("build"),
            "checkpoint": ref.get("checkpoint"),
            "kernel": ref.get("kernel"),
            "throughput_tps": ref.get("throughput_tps"),
            "speed_vs_baseline": ref.get("speed_vs_baseline"),
            "gsm8k": (acc.get("gsm8k") or {}).get("score"),
            "mmlu_pro": (acc.get("mmlu_pro") or {}).get("score"),
        }

    def has(self, key: str) -> bool:
        return bool(self.variants.get(key, {}).get("avg_latency_ms"))

    def variant(self, key: str) -> dict:
        return self.variants.get(key, {})

    def gpus(self, key: str) -> int:
        return int(self.variant(key).get("gpus") or DEFAULT_GPUS.get(key, 1))

    def slices_per_h200(self, key: str) -> int:
        """How many of the setup's device fit one H200: 1 for a full card, 2 for 71 GB slices, 3 for 35 GB."""
        return int(self.variant(key).get("slices_per_h200") or 1)

    def weights_gib_per_gpu(self, key: str) -> float | None:
        """Model weights per GPU rank in GiB, from the vLLM startup log ("Model loading took")."""
        value = self.variant(key).get("weights_gib_per_gpu")
        return float(value) if isinstance(value, int | float) else None

    def latency_model(self, key: str, concurrency: int = 1) -> tuple[float, float, str]:
        """(mean latency ms, lognormal sigma, basis label) for a variant at the given concurrency."""
        v = self.variant(key)
        mean = float(v.get("avg_latency_ms") or 0.0)
        sweep = self.sweeps.get(key)
        n = max(1, int(concurrency))
        if sweep:
            # the sweep's fixed-length prompts give the request-to-request jitter; the single-stream
            # p95 spans ShareGPT prompts of very different lengths, which isn't jitter
            return _interp(sweep, n), self.sweep_sigma(key), f"benchmark c≈{n}"
        return mean, lognormal_sigma(mean, float(v.get("p95_latency_ms") or 0.0)), "benchmark · 1 stream"

    def sweep_sigma(self, key: str) -> float:
        points = self.load_points(key)
        first = points[0] if points else None
        if not first or not first.get("latency_ms") or not first.get("latency_p95_ms"):
            return 0.05
        return min(max(lognormal_sigma(first["latency_ms"], first["latency_p95_ms"]), 0.02), 0.35)


benchmark_data = BenchmarkData(
    settings.resolve(settings.benchmark_file),
    settings.resolve(settings.bench_dir),
)
