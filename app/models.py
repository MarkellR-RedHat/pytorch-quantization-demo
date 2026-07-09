"""Pydantic models for request/response validation"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class InferenceRequest(BaseModel):
    """Request to send inference to a model variant"""
    model_type: Literal["FP32", "FP16", "INT8", "INT4"]
    prompt: Optional[str] = "Hello, how are you?"
    user_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response from model inference"""
    model_type: str
    response_text: str
    latency_ms: float
    tokens_per_second: float
    cost_per_request: float
    timestamp: datetime


class MetricsSnapshot(BaseModel):
    """Real-time metrics for a model variant"""
    model_type: str
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    gpu_memory_gb: float
    tokens_per_second: float
    cost_per_request: float
    queue_depth: int
    active_requests: int
    total_requests: int


class DemoState(BaseModel):
    """Overall demo state"""
    is_running: bool
    simulation_mode: bool
    participant_count: int
    total_requests: int
    start_time: Optional[datetime] = None
    metrics: dict[str, MetricsSnapshot] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Configuration for simulation mode"""
    enabled: bool
    request_rate: float = 10.0  # requests per second
    latency_variation: float = 0.2  # 20% variation
    synthetic_users: int = 75
