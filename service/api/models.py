"""
Pydantic models for OpenTinker Training API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TrainingRequest(BaseModel):
    """Request model for training job submission."""
    task: str = Field(..., description="Training task: 'gomoku' or 'alfworld'")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hydra config overrides for training"
    )
    num_gpus: int = Field(4, ge=1, le=8, description="Number of H100 GPUs to allocate")
    checkpoint_path: Optional[str] = Field(
        None,
        description="Path to checkpoint for resuming training"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task": "gomoku",
                "num_gpus": 4,
                "config": {
                    "experiment_name": "gomoku_training_v1",
                    "batch_size": 32,
                    "num_epochs": 10
                }
            }
        }


class TrainingResponse(BaseModel):
    """Response model for training job submission."""
    status: str = Field(..., description="Job status: queued, running, completed, or failed")
    job_id: str = Field(..., description="Unique job identifier for tracking")
    message: str = Field(..., description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "queued",
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Training job queued for H100 GPU cluster in us-east4-a"
            }
        }
