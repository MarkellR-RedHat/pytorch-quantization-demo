"""Pydantic models for request/response validation"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VariantKey = Literal["FP16", "FP8", "INT4", "SPEC_DECODE"]
PromptId = Literal["chat", "reasoning", "code", "summary"]


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    source: Literal["simulated", "live"]
    response_text: str
    latency_ms: float
    tokens_per_second: float
    completion_tokens: int
    cost_per_request: float | None
    timestamp: datetime


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    label: str
    source: Literal["simulated", "live"]
    basis: str
    gpus: int
    weights_gb: float | None
    requests_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    output_tokens_per_second: float
    cost_per_request: float | None
    in_flight: int
    total_requests: int
    errors: int


class DemoState(BaseModel):
    is_running: bool
    simulation_mode: bool
    participant_count: int
    total_requests: int
    start_time: datetime | None = None


class AskRequest(BaseModel):
    """Presenter-only: a question typed on the presenter laptop, or one of the preset questions."""

    prompt: str = Field(default="", max_length=2000)
    preset: Literal["reasoning", "code", "summary"] | None = None
