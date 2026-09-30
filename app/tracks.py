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
    devices: dict[str, tuple[str, int, int]]  # setup -> (device name, count, devices of that kind per H200)
    lanes: dict[str, str]  # setup -> the router lane text on the Numbers strip
    paths: dict[str, str] | None = None  # settings field names for benchmark_file, quality_dir, bench_dir
    captures: dict[str, str] = field(default_factory=dict)  # setup -> recordings folder, if not the setup key
    labels: dict[str, str] = field(default_factory=dict)  # setup -> on-screen label, when not the default
    builds: dict[str, str] = field(default_factory=dict)  # setup -> which build it runs, when worth saying
    copy: dict[str, dict[str, str]] = field(default_factory=dict)  # setup -> the card wording (see COPY)
    router_note: str = ""  # the line under the router strip: the next lanes, untested here
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
        """What the setup runs on. per_h200 is how many of that device fit one card (1 for a full H200):
        the factor behind every "per H200" number, which the app labels as arithmetic wherever it shows it."""
        name, count, per_h200 = self.devices.get(setup, (H200, 1, 1))
        return {"name": name, "count": count, "per_h200": per_h200}


TITLE = "Not Every Question Needs the Whole GPU"  # one title for both tracks
H200 = "H200"
SLICE_71 = "71 GB slice"  # nvidia.com/mig-3g.71gb, two per H200
SLICE_35 = "35 GB slice"  # nvidia.com/mig-2g.35gb, three per H200

# The wording on the cards, per setup: "role" under the name on Ask, "gets" / "best" / "watch" on Numbers
# ("bestWithFP8" when FP8 is on screen), "acc" / "accNote" when the accuracy cell isn't a measured score,
# and for spec decode what drafts ("drafter"), what checks ("pass") and the long form for the footnote.
# Lines that carry a ratio are computed from the benchmark in the page instead, so words never drift
# from the numbers under them. The Llama copy is the approved Sep 29 wording; the Qwen copy is a DRAFT
# that states only what the setup is, pending the measured numbers and b7's approval.
LLAMA_COPY = {
    "BF16": {
        "role": "The reference",
        "gets": "The reference the others are measured against",
        "best": "Your hardest questions, until the others are tested on them",
        "watch": "Every replica needs two GPUs",
        "acc": "100%", "accNote": "the reference",
    },
    "INT4": {
        "role": "Half the GPUs",
        "gets": "Half the GPUs, and about the same output per GPU under load",
        "best": "Everyday chat and easy questions",
        "bestWithFP8": "When 73 GB of weights won't fit: memory-tight GPUs",
        "watch": "3 to 4 points lower on 280 MMLU-Pro questions, too few to call it, so the hardest "
                 "questions stay on BF16 until it's tested further",
    },
    "SPEC_DECODE": {
        "role": "Same 2 GPUs, 70B + 8B draft",
        "gets": "Faster answers on the same GPUs, with BF16 quality",
        "best": "Latency-sensitive, low-traffic work",
        "watch": "A slower first token, and about half of BF16's tokens per GPU under load: it's a latency "
                 "tool",
        "acc": "= BF16", "accNote": "by design, the 70B checks every token",
        "drafter": "8B draft", "pass": "70B pass", "drafterLong": "Llama 3.1 8B as the draft",
    },
    "FP8": {
        "role": "One GPU, 8-bit",
        "gets": "BF16 speed on one GPU, and the most tokens per GPU under load",
        "best": "Everyday chat and easy questions, on Ada, Hopper and newer",
        "watch": "Needs FP8 tensor cores (Ada, Hopper and newer; on A100 vLLM falls back to a slower "
                 "weight-only kernel) and 73 GB for the weights, so less KV cache room than INT4",
    },
}
QWEN_COPY = {
    "BF16": {
        "role": "The reference, one full H200",
        "gets": "The reference the others are measured against",
        "best": "Your hardest questions, until the others are tested on them",
        "watch": "Needs a full H200: 52 GiB of weights at 32K context",
        "acc": "100%", "accNote": "the reference",
    },
    "INT4": {
        "role": "A 35 GB slice, three per H200",
        "gets": "The smallest slice, three per H200, with 4-bit weights",
        "best": "When 71 GB is too much: the smallest slice that serves the model",
        "watch": "Speed here is the 35 GB slice's; accuracy is the checkpoint's, measured on a 71 GB slice "
                 "(the footnote has the 71 GB like-for-like line against FP8)",
    },
    "SPEC_DECODE": {
        "role": "Same H200, the model's own MTP head",
        "gets": "Faster answers on the same GPU, with BF16 quality",
        "best": "Latency-sensitive, low-traffic work",
        "watch": "A slower first token, and fewer of BF16's tokens per GPU under load: it's a latency tool",
        "acc": "= BF16", "accNote": "by design, the 27B checks every token",
        "drafter": "MTP head", "pass": "27B pass", "drafterLong": "the model's own MTP head as the draft",
    },
    "FP8": {
        "role": "A 71 GB slice, two per H200",
        "gets": "8-bit weights and activations on a 71 GB slice, two per H200",
        "best": "Everyday chat and easy questions, on Ada, Hopper and newer",
        "watch": "Needs FP8 tensor cores (Ada, Hopper and newer; on A100 vLLM falls back to a slower "
                 "weight-only kernel)",
    },
}

TRACKS = {
    "llama": Track(
        key="llama",
        title=TITLE,
        subtitle="When to use BF16, FP8, INT4, or speculative decoding, and when a router earns its keep",
        model="Llama 3.1 70B Instruct",
        short="Llama 70B",
        checkpoint="meta-llama/Meta-Llama-3.1-70B-Instruct",
        devices={
            "BF16": (H200, 2, 1), "FP8": (H200, 1, 1), "INT4": (H200, 1, 1), "SPEC_DECODE": (H200, 2, 1),
        },
        lanes={
            "BF16": "A wrong answer is expensive",
            "FP8": "Everyday questions",
            "INT4": "73 GB won't fit",
            "SPEC_DECODE": "Latency-sensitive, low traffic",
        },
        copy=LLAMA_COPY,
        router_note=(
            "A router pays off once your traffic is big and mixed enough to run more than one pool. With "
            "small traffic, pick the one setup that fits most of your questions. INT4 + spec decode, a draft "
            "model on an INT4 target, is the obvious next lane, untested here."
        ),
    ),
    "qwen": Track(
        key="qwen",
        title=TITLE,
        subtitle="When a 71 GB slice is enough, when it isn't, and when a router earns its keep",
        model="Qwen3.8-27B",
        short="Qwen 27B",
        checkpoint="Qwen/Qwen3.8-27B",
        # the INT4 card is the 35 GB slice: its speed, load numbers and recordings come from that run
        # (INT4_35 in the raw folder), its accuracy from the same checkpoint on a 71 GB slice
        devices={
            "BF16": (H200, 1, 1), "FP8": (SLICE_71, 1, 2), "INT4": (SLICE_35, 1, 3),
            "SPEC_DECODE": (H200, 1, 1),
        },
        lanes={
            "BF16": "A wrong answer is expensive",
            "FP8": "Everyday questions",
            "INT4": "When 71 GB is too much",
            "SPEC_DECODE": "Latency-sensitive, low traffic",
        },
        paths={
            "benchmark_file": "qwen_benchmark_file",
            "quality_dir": "qwen_quality_dir",
            "bench_dir": "qwen_bench_dir",
        },
        captures={"INT4": "INT4_35"},
        # RedHatAI/Qwen3.8-27B-INT4 is an LLM Compressor build (compressed-tensors, AWQ smoothing + GPTQ,
        # W4A16), unlike the Llama W4A16, which is AutoGPTQ format
        labels={"INT4": "INT4 (LLM Compressor W4A16)"},
        builds={"INT4": "Red Hat's LLM Compressor W4A16 build (AWQ smoothing + GPTQ)"},
        copy=QWEN_COPY,
        # the DSpark card (RedHatAI/Qwen3.8-27B-speculator.dspark, evaluated on vLLM 0.29.0) says DSpark
        # "consistently delivers higher throughput and better interactivity than MTP with the same eight
        # speculative tokens"; the NVFP4 card evaluates on a B200 and recommends the DSpark draft on top
        router_note=(
            "A router pays off once your traffic is big and mixed enough to run more than one pool. With "
            "small traffic, pick the one setup that fits most of your questions. Two next lanes, untested "
            "here, from Red Hat's model cards: the DSpark speculator (vLLM 0.29+), which Red Hat reports "
            "ahead of native MTP at the same eight speculative tokens, and NVFP4 on Blackwell, which stacks "
            "with it."
        ),
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
        "copy": t.copy,
        "router_note": t.router_note,
        "tracks": listing(),
    }
