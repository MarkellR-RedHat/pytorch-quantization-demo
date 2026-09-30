"""Two tracks, one demo: the Llama 3.1 70B track (two full H200s and one) and the Qwen3.8-27B track
(one full H200 and 71 GB MIG slices). A track owns its benchmark file, its recordings folder, its
sweeps, the device each setup runs on, and the wording of its router lanes. The setup keys are the
same in both, so the app code doesn't fork."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.benchmark import BenchmarkData
from app.config import settings

logger = logging.getLogger(__name__)


class TrackPending(Exception):
    """The track's data hasn't landed, and pending tracks aren't allowed on screen."""


@dataclass(frozen=True)
class Track:
    key: str
    title: str
    subtitle: str
    model: str
    short: str  # what the corner badge calls it
    checkpoint: str
    devices: dict[str, tuple[str, int]]  # setup -> (device name, count)
    lanes: dict[str, str]  # setup -> the router lane text on the Numbers strip
    paths: dict[str, str] | None = None  # settings field names for benchmark_file, quality_dir, bench_dir
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def _setting(self, name: str) -> str:
        return getattr(settings, self.paths[name] if self.paths else name)

    @property
    def benchmark_file(self) -> Path:
        return settings.resolve(self._setting("benchmark_file"))

    @property
    def quality_dir(self) -> Path:
        return settings.resolve(self._setting("quality_dir"))

    @property
    def bench_dir(self) -> Path:
        return settings.resolve(self._setting("bench_dir"))

    def missing(self) -> list[str]:
        out = []
        if not self.benchmark_file.is_file():
            out.append(f"benchmark file {self._setting('benchmark_file')}")
        if not self.quality_dir.is_dir():
            out.append(f"recordings folder {self._setting('quality_dir')}")
        return out

    @property
    def status(self) -> str:
        return "pending" if self.missing() else "ready"

    @property
    def benchmark(self) -> BenchmarkData:
        """Loaded on first use and reloaded when the file it points at changes."""
        key = (str(self.benchmark_file), str(self.bench_dir))
        if self._cache.get("key") != key:
            self._cache["key"] = key
            self._cache["data"] = BenchmarkData(self.benchmark_file, self.bench_dir)
        return self._cache["data"]

    def device(self, setup: str) -> dict:
        name, count = self.devices.get(setup, ("H200", 1))
        return {"name": name, "count": count}


H200 = "H200"
SLICE = "71 GB slice"

TRACKS = {
    "llama": Track(
        key="llama",
        title="Not Every Question Needs Two GPUs",
        subtitle="When to use BF16, INT4, or speculative decoding, and when a router earns its keep",
        model="Llama 3.1 70B Instruct",
        short="Llama 70B",
        checkpoint="meta-llama/Meta-Llama-3.1-70B-Instruct",
        devices={"BF16": (H200, 2), "FP8": (H200, 1), "INT4": (H200, 1), "SPEC_DECODE": (H200, 2)},
        lanes={
            "BF16": "A wrong answer is expensive",
            "FP8": "Everyday questions",
            "INT4": "73 GB won't fit",
            "SPEC_DECODE": "Latency-sensitive, low traffic",
        },
    ),
    "qwen": Track(
        key="qwen",
        title="Not Every Question Needs the Whole GPU",
        subtitle="When a 71 GB slice is enough, when it isn't, and when a router earns its keep",
        model="Qwen3.8-27B",
        short="Qwen 27B",
        checkpoint="Qwen/Qwen3.8-27B",
        devices={"BF16": (H200, 1), "FP8": (SLICE, 1), "INT4": (SLICE, 1), "SPEC_DECODE": (H200, 1)},
        lanes={
            "BF16": "A wrong answer is expensive",
            "FP8": "Everyday questions",
            "INT4": "The smallest slice",
            "SPEC_DECODE": "Latency-sensitive, low traffic",
        },
        paths={
            "benchmark_file": "qwen_benchmark_file",
            "quality_dir": "qwen_quality_dir",
            "bench_dir": "qwen_bench_dir",
        },
    ),
}

_active: Track = TRACKS["llama"]
_listeners: list = []


def active() -> Track:
    return _active


def on_change(callback) -> None:
    """Called with the new track after every switch (the app rebinds its simulator to it)."""
    _listeners.append(callback)


def select(key: str, allow_pending: bool | None = None) -> Track:
    global _active
    track = TRACKS[key]
    allow = settings.allow_pending_tracks if allow_pending is None else allow_pending
    if track.status == "pending" and not allow:
        raise TrackPending(f"track {key} is pending: missing {', '.join(track.missing())}")
    _active = track
    for callback in _listeners:
        callback(track)
    return track


def select_from_settings() -> Track:
    """The start track from TRACK, falling back to llama when the chosen track has no data yet."""
    try:
        return select(settings.track)
    except (KeyError, TrackPending) as e:
        logger.warning(f"TRACK={settings.track!r} not usable ({e}); starting on llama")
        return select("llama")


def listing() -> list[dict]:
    return [
        {"key": t.key, "title": t.title, "model": t.model, "status": t.status, "missing": t.missing()}
        for t in TRACKS.values()
    ]


def describe(track: Track | None = None) -> dict:
    t = track or _active
    return {
        "key": t.key,
        "title": t.title,
        "subtitle": t.subtitle,
        "model": t.model,
        "short": t.short,
        "checkpoint": t.checkpoint,
        "status": t.status,
        "lanes": t.lanes,
        "tracks": listing(),
    }
