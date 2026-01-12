"""Core types and interfaces for JURIS-AGI."""

from .types import (
    Grid,
    Color,
    Point,
    BoundingBox,
    GridObject,
    ARCTask,
    ARCPair,
    AuditTrace,
    SolverResult,
    Constraint,
    ConstraintType,
)
from .state import MultiViewState, MultiViewStateBuilder, build_multiview_state
from .metrics import (
    compute_exact_match,
    compute_pixel_accuracy,
    compute_grid_diff,
    evaluate_on_pairs,
    score_solution,
)
from .trace import SolveTrace, TraceEntry, TraceWriter, TraceContext
from .storage import (
    StorageConfig,
    StorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
    StorageClient,
    ArtifactMetadata,
)
from .model_registry import (
    ModelRegistry,
    ModelEntry,
    ModelVersion,
)

__all__ = [
    # Types
    "Grid",
    "Color",
    "Point",
    "BoundingBox",
    "GridObject",
    "ARCTask",
    "ARCPair",
    "AuditTrace",
    "SolverResult",
    "Constraint",
    "ConstraintType",
    # State
    "MultiViewState",
    "MultiViewStateBuilder",
    "build_multiview_state",
    # Metrics
    "compute_exact_match",
    "compute_pixel_accuracy",
    "compute_grid_diff",
    "evaluate_on_pairs",
    "score_solution",
    # Trace
    "SolveTrace",
    "TraceEntry",
    "TraceWriter",
    "TraceContext",
    # Storage
    "StorageConfig",
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    "StorageClient",
    "ArtifactMetadata",
    # Model Registry
    "ModelRegistry",
    "ModelEntry",
    "ModelVersion",
]
