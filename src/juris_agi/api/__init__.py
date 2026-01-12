"""JURIS-AGI API module for cloud and local deployment."""

from .server import app, create_app
from .worker import JurisWorker
from .models import (
    SolveRequest,
    SolveResponse,
    JobStatus,
    JobResult,
    HealthResponse,
)
from .local_config import LocalPoCConfig, get_local_config, is_local_poc_mode
from .local_server import create_local_app

__all__ = [
    # Cloud API
    "app",
    "create_app",
    "JurisWorker",
    # Models
    "SolveRequest",
    "SolveResponse",
    "JobStatus",
    "JobResult",
    "HealthResponse",
    # Local PoC
    "LocalPoCConfig",
    "get_local_config",
    "is_local_poc_mode",
    "create_local_app",
]
