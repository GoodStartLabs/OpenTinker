"""
Shared Pydantic models for OpenTinker services.

Data models shared across API gateway and worker components.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class TrainingTask(str, Enum):
    """Supported training tasks."""
    GOMOKU = "gomoku"
    ALFWORLD = "alfworld"
    MATH = "math"
    MATH_TOOL = "math_tool"
    GEO3K = "geo3k"


class JobStatus(str, Enum):
    """Job status states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """Training configuration parameters."""
    experiment_name: Optional[str] = Field(None, description="Experiment name for tracking")
    batch_size: Optional[int] = Field(None, ge=1, description="Training batch size")
    num_epochs: Optional[int] = Field(None, ge=1, description="Number of training epochs")
    learning_rate: Optional[float] = Field(None, gt=0, description="Learning rate")
    temperature: Optional[float] = Field(None, gt=0, description="Sampling temperature")
    max_new_tokens: Optional[int] = Field(None, ge=1, description="Max tokens to generate")

    class Config:
        use_enum_values = True


class JobMetrics(BaseModel):
    """Training job metrics."""
    loss: Optional[float] = None
    reward: Optional[float] = None
    accuracy: Optional[float] = None
    steps_completed: Optional[int] = None
    epochs_completed: Optional[int] = None
    gpu_utilization: Optional[float] = None
    duration_seconds: Optional[float] = None


class JobResult(BaseModel):
    """Complete job result."""
    status: JobStatus
    job_id: str
    celery_task_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metrics: Optional[JobMetrics] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    class Config:
        use_enum_values = True


class HealthStatus(BaseModel):
    """Health check status."""
    status: str = Field(..., description="Overall health status")
    component: str = Field(..., description="Component name")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional details")
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "component": "api",
                "details": {"celery": "connected", "uptime": "3h 24m"},
                "timestamp": "2025-01-28T10:30:00Z"
            }
        }
