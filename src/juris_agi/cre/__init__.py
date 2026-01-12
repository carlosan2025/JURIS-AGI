"""CRE (Certified Reasoning Expert) module."""

from .synthesizer import (
    Synthesizer,
    BeamSearchSynthesizer,
    SynthesisResult,
    SynthesisConfig,
)
from .critic_symbolic import (
    SymbolicCritic,
    CriticResult,
    compute_symbolic_diff,
)
from .refinement import (
    RefinementEngine,
    RefinementResult,
    EditOperation,
)
from .sketcher_model import (
    SketcherModel,
    ProgramSketch,
    SketcherConfig,
    HeuristicSketcher,
    get_sketcher,
    TORCH_AVAILABLE,
)
from .critic_neural import (
    NeuralCritic,
    NeuralCriticScore,
    CriticConfig,
    StubNeuralCritic,
    EnsembleNeuralCritic,
    get_critic,
)

__all__ = [
    # Synthesizer
    "Synthesizer",
    "BeamSearchSynthesizer",
    "SynthesisResult",
    "SynthesisConfig",
    # Symbolic Critic
    "SymbolicCritic",
    "CriticResult",
    "compute_symbolic_diff",
    # Refinement
    "RefinementEngine",
    "RefinementResult",
    "EditOperation",
    # Neural Sketcher
    "SketcherModel",
    "ProgramSketch",
    "SketcherConfig",
    "HeuristicSketcher",
    "get_sketcher",
    "TORCH_AVAILABLE",
    # Neural Critic
    "NeuralCritic",
    "NeuralCriticScore",
    "CriticConfig",
    "StubNeuralCritic",
    "EnsembleNeuralCritic",
    "get_critic",
]
