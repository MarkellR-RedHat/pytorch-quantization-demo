"""Application configuration"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

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

    # OpenShift AI (live mode)
    openshift_ai_endpoint: str = ""
    openshift_ai_token: str = ""

    # Model endpoints, one vLLM OpenAI-compatible chat completions URL per variant
    model_fp16_endpoint: str = ""
    model_fp8_endpoint: str = ""
    model_int4_endpoint: str = ""
    model_spec_decode_endpoint: str = ""

    # Served model names (vLLM --served-model-name); empty means the variant key lowercased
    model_fp16_name: str = ""
    model_fp8_name: str = ""
    model_int4_name: str = ""
    model_spec_decode_name: str = ""

    # Per-setup backup switch: "recorded" never calls that endpoint and plays the preset recordings
    # instead, labeled as recorded. Use it when a setup's GPUs get pulled on the day.
    model_fp16_mode: str = "live"
    model_fp8_mode: str = "live"
    model_int4_mode: str = "live"
    model_spec_decode_mode: str = "live"

    # Which folder under quality/ holds the INT4 column's recordings: INT4 for the community AWQ
    # build, INT4_RH for Red Hat's LLM Compressor build. It also names the build on screen.
    model_int4_captures: str = "INT4"

    # Demo
    simulation_mode: bool = False
    enable_prometheus: bool = True
    baseline_label: str = "BF16"
    presenter_key: str = ""
    trust_proxy: bool = False

    # Traffic and protection
    auto_traffic: bool = True
    auto_traffic_rps: float = 0.5
    max_inflight_per_variant: int = 32

    # Cost model: hourly price of one GPU. 0 hides cost everywhere.
    gpu_hourly_usd: float = 0.0

    # Data files (relative paths resolve against the repo root)
    benchmark_file: str = "benchmark_results.json"
    bench_dir: str = "bench"
    quality_dir: str = "quality"
    sim_time_scale: float = 1.0

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    def endpoint_for(self, key: str) -> str:
        return getattr(self, f"model_{key.lower()}_endpoint", "")

    def served_name_for(self, key: str) -> str:
        return getattr(self, f"model_{key.lower()}_name", "") or key.lower()

    def mode_for(self, key: str) -> str:
        mode = str(getattr(self, f"model_{key.lower()}_mode", "live")).strip().lower()
        return "recorded" if mode == "recorded" else "live"

    def captures_for(self, key: str) -> str:
        """The quality/ folder holding this column's recordings."""
        if key == "INT4":
            return self.model_int4_captures.strip() or "INT4"
        return key


BASE_VARIANTS = ["FP16", "INT4", "SPEC_DECODE"]
ALL_VARIANTS = ["FP16", "FP8", "INT4", "SPEC_DECODE"]
# Kept for backwards compatibility with older imports
MODEL_VARIANTS = BASE_VARIANTS

settings = Settings()


INT4_BUILDS = {
    "INT4": "hugging-quants AWQ build",
    "INT4_RH": "Red Hat LLM Compressor build",
}


def build_note(key: str) -> str | None:
    """Which checkpoint a column runs, when there's more than one it could be."""
    if key != "INT4":
        return None
    return INT4_BUILDS.get(settings.captures_for(key), settings.captures_for(key))


def variant_label(key: str) -> str:
    """The label of a live (or recorded) column, which follows the INT4 build it runs."""
    if key == "INT4" and settings.captures_for(key) == "INT4_RH":
        return "INT4 (LLM Compressor)"
    return benchmark_label(key)


def benchmark_label(key: str) -> str:
    """The label of a setup's benchmark numbers. The Sep 29 INT4 numbers are the AWQ build's,
    whatever the INT4 column runs today."""
    labels = {
        "FP16": settings.baseline_label,
        "FP8": "FP8",
        "INT4": "INT4 AWQ",
        "SPEC_DECODE": "Spec Decode",
    }
    return labels.get(key, key)
