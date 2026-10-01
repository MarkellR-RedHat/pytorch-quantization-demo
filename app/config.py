from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Bearer token for the vLLM endpoints, if they need one
    openshift_ai_token: str = ""

    # Model endpoints, one vLLM OpenAI-compatible chat completions URL per variant
    model_bf16_endpoint: str = ""
    model_fp8_endpoint: str = ""
    model_int4_endpoint: str = ""
    model_spec_decode_endpoint: str = ""

    # Served model names (vLLM --served-model-name); empty means the variant key lowercased
    model_bf16_name: str = ""
    model_fp8_name: str = ""
    model_int4_name: str = ""
    model_spec_decode_name: str = ""

    # Per-setup backup switch: "recorded" never calls that endpoint and plays the preset recordings
    # instead, labeled as recorded. Use it when a setup's GPUs get pulled on the day.
    model_bf16_mode: str = "live"
    model_fp8_mode: str = "live"
    model_int4_mode: str = "live"
    model_spec_decode_mode: str = "live"

    # Which folder under quality/ holds the INT4 column's recordings: INT4 for the community AWQ
    # build, INT4_RH for Red Hat's W4A16 build (GPTQ via AutoGPTQ). It also names the build on screen.
    model_int4_captures: str = "INT4_RH"

    # The Qwen track's endpoints, served names and modes: the same four columns, its own pods. They
    # default to recorded, so the track plays its recordings unless an endpoint is given and its mode
    # set to live (the same rules as the Llama settings above, read when the Qwen track is on screen).
    qwen_model_bf16_endpoint: str = ""
    qwen_model_fp8_endpoint: str = ""
    qwen_model_int4_endpoint: str = ""
    qwen_model_spec_decode_endpoint: str = ""
    qwen_model_bf16_name: str = "qwen-bf16"
    qwen_model_fp8_name: str = "qwen-fp8"
    qwen_model_int4_name: str = "qwen-int4"
    qwen_model_spec_decode_name: str = "qwen-mtp"
    qwen_model_bf16_mode: str = "recorded"
    qwen_model_fp8_mode: str = "recorded"
    qwen_model_int4_mode: str = "recorded"
    qwen_model_spec_decode_mode: str = "recorded"

    # Demo
    simulation_mode: bool = False
    presenter_key: str = ""
    trust_proxy: bool = False

    # Traffic and protection
    auto_traffic: bool = True
    auto_traffic_rps: float = 0.5
    max_inflight_per_variant: int = 32

    # Cost model: hourly price of one GPU. 0 hides cost everywhere.
    gpu_hourly_usd: float = 0.0
    # Latency target for the load test: the tail (p95, or p99 when that's all the run recorded) time
    # per output token a setup must stay under for its throughput to count. 50 ms is 20 tokens/s
    # per user, comfortably faster than anyone reads.
    tpot_target_ms: float = 50.0

    # Data files (relative paths resolve against the repo root)
    benchmark_file: str = "benchmark_results.json"
    bench_dir: str = "bench/raw/2026-09-29-r2/sweeps"
    quality_dir: str = "quality"

    # The second track. Its files land when the Qwen run comes back; until then the track is pending
    # and stays off screen unless ALLOW_PENDING_TRACKS is set.
    track: str = "llama"
    allow_pending_tracks: bool = False
    qwen_benchmark_file: str = "benchmark_results.qwen.json"
    qwen_quality_dir: str = "quality/qwen"
    qwen_bench_dir: str = "bench/raw/2026-09-30-qwen-r1/sweeps"
    sim_time_scale: float = 1.0

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    def _field(self, key: str, what: str, default=""):
        """The setting for a column on the track on screen: model_<key>_<what>, or the track's prefixed
        copy of it (qwen_model_<key>_<what>) when the Qwen track is up."""
        return getattr(self, f"{_track().settings_prefix}model_{key.lower()}_{what}", default)

    def endpoint_for(self, key: str) -> str:
        return self._field(key, "endpoint")

    def served_name_for(self, key: str) -> str:
        return self._field(key, "name") or key.lower()

    def mode_for(self, key: str) -> str:
        mode = str(self._field(key, "mode", "live")).strip().lower()
        return "recorded" if mode == "recorded" else "live"

    def captures_for(self, key: str) -> str:
        """The folder under the track's recordings dir holding this column's recordings."""
        if folder := _track().captures.get(key):
            return folder
        if key == "INT4":
            return self.model_int4_captures.strip() or "INT4"
        return key


def _track():
    from app import tracks  # here and not at the top: tracks imports this module

    return tracks.active()


BASE_VARIANTS = ["BF16", "INT4", "SPEC_DECODE"]
ALL_VARIANTS = ["BF16", "FP8", "INT4", "SPEC_DECODE"]

settings = Settings()


INT4_BUILDS = {
    "INT4": "hugging-quants AWQ build",
    "INT4_RH": "Red Hat's validated build: W4A16, GPTQ via AutoGPTQ",
}


def build_note(key: str) -> str | None:
    """Which checkpoint a column runs, when there's more than one it could be."""
    if build := _track().builds.get(key):
        return build
    if key != "INT4":
        return None
    return INT4_BUILDS.get(settings.captures_for(key), settings.captures_for(key))


def variant_label(key: str) -> str:
    """The label of a live (or recorded) column, which follows the INT4 build it runs."""
    if label := _track().labels.get(key):
        return label
    if key == "INT4" and settings.captures_for(key) == "INT4_RH":
        return "INT4 (GPTQ via AutoGPTQ)"
    return benchmark_label(key)


def benchmark_label(key: str) -> str:
    """The label of a setup's benchmark numbers. On the Llama track the Sep 29 INT4 numbers are the AWQ
    build's, whatever the INT4 column runs today; a track that measured what it runs names it once."""
    if label := _track().labels.get(key):
        return label
    labels = {
        "BF16": "BF16",
        "FP8": "FP8",
        "INT4": "INT4 AWQ",
        "SPEC_DECODE": "Spec Decode",
    }
    return labels.get(key, key)
