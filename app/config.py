"""Application configuration"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # OpenShift AI Configuration
    openshift_ai_endpoint: str
    openshift_ai_token: str

    # Model Endpoints (3 variants)
    model_fp16_endpoint: str
    model_int4_endpoint: str
    model_spec_decode_endpoint: str

    # Demo Configuration
    simulation_mode: bool = False
    enable_prometheus: bool = True

    # Cost Configuration (per 1000 tokens)
    cost_fp16_per_1k: float = 0.0015
    cost_int4_per_1k: float = 0.0005
    cost_spec_decode_per_1k: float = 0.0007

    @property
    def model_endpoints(self) -> dict[str, str]:
        return {
            "FP16": self.model_fp16_endpoint,
            "INT4": self.model_int4_endpoint,
            "SPEC_DECODE": self.model_spec_decode_endpoint,
        }

    @property
    def cost_per_1k(self) -> dict[str, float]:
        return {
            "FP16": self.cost_fp16_per_1k,
            "INT4": self.cost_int4_per_1k,
            "SPEC_DECODE": self.cost_spec_decode_per_1k,
        }


MODEL_VARIANTS = ["FP16", "INT4", "SPEC_DECODE"]

# Global settings instance
settings = Settings()
