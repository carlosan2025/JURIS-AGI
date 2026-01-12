"""WME (World Model Expert) module - provides priors and counterfactuals."""

from .priors import (
    PriorKnowledge,
    ARCPriors,
    TransformationPrior,
    # New propose_priors API
    ProposedPriors,
    ProposedInvariant,
    TransformFamily,
    propose_priors,
    extract_invariants_from_task,
)
from .world_model import (
    WorldModel,
    WorldModelState,
)
from .counterfactuals import (
    CounterfactualGenerator,
    Counterfactual,
    # New invariant-preserving API
    InvariantSpec,
    InvariantPreservingGenerator,
    generate_counterfactuals,
)
from .robustness import (
    RobustnessChecker,
    RobustnessResult,
    # New robustness_check API
    CounterfactualRobustnessResult,
    robustness_check,
    quick_robustness_check,
)

__all__ = [
    # Priors
    "PriorKnowledge",
    "ARCPriors",
    "TransformationPrior",
    "ProposedPriors",
    "ProposedInvariant",
    "TransformFamily",
    "propose_priors",
    "extract_invariants_from_task",
    # World Model
    "WorldModel",
    "WorldModelState",
    # Counterfactuals
    "CounterfactualGenerator",
    "Counterfactual",
    "InvariantSpec",
    "InvariantPreservingGenerator",
    "generate_counterfactuals",
    # Robustness
    "RobustnessChecker",
    "RobustnessResult",
    "CounterfactualRobustnessResult",
    "robustness_check",
    "quick_robustness_check",
]
