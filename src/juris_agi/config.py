"""
JURIS-AGI Configuration.

Provides centralized configuration with reproducibility settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# Default random seed for reproducibility
DEFAULT_SEED = 42


@dataclass
class JurisAGIConfig:
    """
    Global configuration for JURIS-AGI.

    Supports reproducibility through fixed seeds.
    """

    # Reproducibility
    random_seed: int = DEFAULT_SEED
    """Random seed for deterministic analysis."""

    # Analysis settings
    num_counterfactuals: int = 20
    """Number of counterfactuals to generate."""

    max_perturbations_per_claim: int = 5
    """Maximum perturbations to try per claim."""

    # Safeguard thresholds
    min_confidence_for_recommendation: float = 0.4
    """Minimum confidence to provide a recommendation."""

    min_claims_for_analysis: int = 5
    """Minimum number of claims required for analysis."""

    low_confidence_threshold: float = 0.6
    """Threshold below which to show low confidence warning."""

    high_uncertainty_threshold: float = 0.35
    """High uncertainty threshold for refusal."""

    # Required claim types for minimal viable analysis
    required_claim_types: list = field(
        default_factory=lambda: ["traction", "team_quality"]
    )

    # Trace settings
    always_generate_trace: bool = True
    """Always generate decision trace."""

    store_all_counterfactuals: bool = False
    """Store all tested counterfactuals in trace."""

    @classmethod
    def from_env(cls) -> "JurisAGIConfig":
        """Load configuration from environment variables."""
        return cls(
            random_seed=int(os.getenv("JURIS_RANDOM_SEED", str(DEFAULT_SEED))),
            num_counterfactuals=int(os.getenv("JURIS_NUM_COUNTERFACTUALS", "20")),
            min_confidence_for_recommendation=float(
                os.getenv("JURIS_MIN_CONFIDENCE", "0.4")
            ),
            min_claims_for_analysis=int(os.getenv("JURIS_MIN_CLAIMS", "5")),
            always_generate_trace=os.getenv("JURIS_ALWAYS_TRACE", "true").lower()
            == "true",
        )


# Global config instance
_config: Optional[JurisAGIConfig] = None


def get_config() -> JurisAGIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = JurisAGIConfig.from_env()
    return _config


def set_config(config: JurisAGIConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to default (reload from env)."""
    global _config
    _config = None
