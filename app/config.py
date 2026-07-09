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

    # Model Endpoints
    model_fp32_endpoint: str
    model_fp16_endpoint: str
    model_int8_endpoint: str
    model_int4_endpoint: str

    # Demo Configuration
    simulation_mode: bool = False
    enable_prometheus: bool = True

    # Cost Configuration (per 1000 tokens)
    cost_fp32_per_1k: float = 0.0020  # $0.002 per 1k tokens
    cost_fp16_per_1k: float = 0.0015  # $0.0015 per 1k tokens
    cost_int8_per_1k: float = 0.0010  # $0.001 per 1k tokens
    cost_int4_per_1k: float = 0.0005  # $0.0005 per 1k tokens

    @property
    def model_endpoints(self) -> dict[str, str]:
        """Get all model endpoints as a dictionary"""
        return {
            "FP32": self.model_fp32_endpoint,
            "FP16": self.model_fp16_endpoint,
            "INT8": self.model_int8_endpoint,
            "INT4": self.model_int4_endpoint,
        }

    @property
    def cost_per_1k(self) -> dict[str, float]:
        """Get cost per 1k tokens for each model type"""
        return {
            "FP32": self.cost_fp32_per_1k,
            "FP16": self.cost_fp16_per_1k,
            "INT8": self.cost_int8_per_1k,
            "INT4": self.cost_int4_per_1k,
        }


# Global settings instance
settings = Settings()
