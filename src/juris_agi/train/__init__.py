"""Training utilities for JURIS-AGI neural components."""

from .train_sketcher import (
    SketcherTrainer,
    SketcherTrainingConfig,
    generate_synthetic_task,
)
from .train_critic import (
    CriticTrainer,
    CriticTrainingConfig,
)

__all__ = [
    "SketcherTrainer",
    "SketcherTrainingConfig",
    "generate_synthetic_task",
    "CriticTrainer",
    "CriticTrainingConfig",
]
