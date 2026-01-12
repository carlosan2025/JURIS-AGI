"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a solve job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class GridData(BaseModel):
    """A 2D grid of integers (0-9)."""
    data: List[List[int]] = Field(..., description="2D array of color values (0-9)")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            }
        }


class TrainPair(BaseModel):
    """A training input/output pair."""
    input: GridData
    output: GridData


class TestPair(BaseModel):
    """A test input (output may be unknown)."""
    input: GridData
    output: Optional[GridData] = None


class TaskPayload(BaseModel):
    """ARC task payload."""
    train: List[TrainPair] = Field(..., description="Training pairs")
    test: List[TestPair] = Field(default_factory=list, description="Test pairs")

    class Config:
        json_schema_extra = {
            "example": {
                "train": [
                    {
                        "input": {"data": [[1, 2], [3, 4]]},
                        "output": {"data": [[3, 1], [4, 2]]}
                    }
                ],
                "test": [
                    {"input": {"data": [[5, 6], [7, 8]]}}
                ]
            }
        }


class BudgetConfig(BaseModel):
    """Budget configuration for solving."""
    max_time_seconds: float = Field(default=60.0, ge=1.0, le=600.0, description="Maximum time in seconds")
    max_iterations: int = Field(default=1000, ge=10, le=100000, description="Maximum synthesis iterations")
    beam_width: int = Field(default=50, ge=1, le=500, description="Beam search width")
    max_depth: int = Field(default=4, ge=1, le=10, description="Maximum program depth")


class SolveRequest(BaseModel):
    """Request to solve an ARC task."""
    task_id: Optional[str] = Field(default=None, description="Optional task identifier")
    task: TaskPayload = Field(..., description="The ARC task to solve")
    budget: BudgetConfig = Field(default_factory=BudgetConfig, description="Compute budget")
    use_neural: bool = Field(default=True, description="Use neural components if available")
    return_trace: bool = Field(default=True, description="Include execution trace in result")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "example_rotation",
                "task": {
                    "train": [
                        {
                            "input": {"data": [[1, 2], [3, 4]]},
                            "output": {"data": [[3, 1], [4, 2]]}
                        }
                    ],
                    "test": [
                        {"input": {"data": [[5, 6], [7, 8]]}}
                    ]
                },
                "budget": {
                    "max_time_seconds": 30.0,
                    "max_iterations": 500
                }
            }
        }


class SolveResponse(BaseModel):
    """Response after submitting a solve request."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    estimated_time_seconds: Optional[float] = Field(default=None, description="Estimated completion time")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "pending",
                "created_at": "2024-01-15T10:30:00Z",
                "estimated_time_seconds": 30.0
            }
        }


class PredictionResult(BaseModel):
    """A prediction for a test input."""
    test_index: int = Field(..., description="Index of the test pair")
    prediction: GridData = Field(..., description="Predicted output grid")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class JobResult(BaseModel):
    """Result of a completed job."""
    job_id: str
    status: JobStatus
    task_id: Optional[str] = None

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runtime_seconds: Optional[float] = None

    # Result
    success: bool = False
    predictions: List[PredictionResult] = Field(default_factory=list)
    program: Optional[str] = Field(default=None, description="Synthesized program")

    # Metrics
    robustness_score: Optional[float] = None
    regime: Optional[str] = None
    synthesis_iterations: Optional[int] = None

    # Error info
    error_message: Optional[str] = None

    # Artifacts
    trace_url: Optional[str] = Field(default=None, description="URL to full execution trace")
    result_url: Optional[str] = Field(default=None, description="URL to result JSON artifact")
    trace_data: Optional[Dict[str, Any]] = Field(default=None, description="Inline trace data")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "completed",
                "task_id": "example_rotation",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:15Z",
                "runtime_seconds": 15.2,
                "success": True,
                "predictions": [
                    {
                        "test_index": 0,
                        "prediction": {"data": [[7, 5], [8, 6]]},
                        "confidence": 0.95
                    }
                ],
                "program": "rotate90(1)",
                "robustness_score": 0.92
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    torch_available: bool = Field(..., description="Whether PyTorch is available")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    worker_count: int = Field(default=0, description="Number of active workers")
    pending_jobs: int = Field(default=0, description="Number of pending jobs")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "gpu_available": True,
                "torch_available": True,
                "redis_connected": True,
                "worker_count": 2,
                "pending_jobs": 5
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
