"""
Configuration schema for JURIS-AGI project-level settings.

Provides pydantic models for configuring scaling vs accuracy tradeoffs
across the reasoning pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Current config schema version
CONFIG_VERSION = "1.0.0"


class PresetName(str, Enum):
    """Available configuration presets."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    CUSTOM = "custom"


class ContextDials(BaseModel):
    """
    Configuration for context building and claim selection.

    Controls how claims are filtered and selected from the evidence graph
    before reasoning begins.
    """

    max_claims: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum number of claims to include in reasoning context"
    )

    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for claim inclusion"
    )

    per_bucket_cap: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum claims per ontology bucket"
    )

    required_buckets: list[str] = Field(
        default_factory=list,
        description="Ontology buckets that must have at least one claim"
    )

    conflict_resolution: str = Field(
        default="highest_confidence",
        description="Strategy for resolving conflicting claims: highest_confidence, most_recent, manual"
    )

    include_low_confidence: bool = Field(
        default=False,
        description="Include low-confidence claims with reduced weight"
    )

    low_confidence_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight multiplier for low-confidence claims when included"
    )

    @field_validator("conflict_resolution")
    @classmethod
    def validate_conflict_resolution(cls, v: str) -> str:
        valid = {"highest_confidence", "most_recent", "manual", "weighted_average"}
        if v not in valid:
            raise ValueError(f"conflict_resolution must be one of {valid}")
        return v


class SearchDials(BaseModel):
    """
    Configuration for hypothesis search and policy learning.

    Controls the breadth and depth of the search space exploration.
    """

    max_hypotheses: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of policy hypotheses to generate"
    )

    beam_width: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Beam width for beam search during policy learning"
    )

    max_rules_per_policy: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum rules allowed in a single policy"
    )

    max_search_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum iterations for policy search"
    )

    early_stopping_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Coverage threshold to trigger early stopping"
    )

    mdl_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for MDL (Minimum Description Length) in scoring"
    )

    coverage_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for coverage in scoring"
    )

    diversity_bonus: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Bonus for hypothesis diversity"
    )

    use_neural_guidance: bool = Field(
        default=True,
        description="Use neural network to guide symbolic search"
    )

    neural_candidate_limit: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum candidates from neural proposer"
    )


class NumericDials(BaseModel):
    """
    Configuration for numeric threshold discovery.

    Controls how thresholds are discovered for numeric predicates.
    """

    max_thresholds_per_field: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum thresholds to consider per numeric field"
    )

    threshold_discovery_method: str = Field(
        default="information_gain",
        description="Method for threshold discovery: information_gain, percentile, clustering"
    )

    percentile_candidates: list[int] = Field(
        default_factory=lambda: [10, 25, 50, 75, 90],
        description="Percentiles to consider when using percentile method"
    )

    min_samples_per_split: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Minimum samples required on each side of a threshold"
    )

    numeric_precision: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Decimal places for threshold rounding"
    )

    use_trend_predicates: bool = Field(
        default=True,
        description="Enable time-series trend predicates"
    )

    trend_window_size: int = Field(
        default=3,
        ge=2,
        le=12,
        description="Window size for trend calculation"
    )

    @field_validator("threshold_discovery_method")
    @classmethod
    def validate_discovery_method(cls, v: str) -> str:
        valid = {"information_gain", "percentile", "clustering", "entropy"}
        if v not in valid:
            raise ValueError(f"threshold_discovery_method must be one of {valid}")
        return v


class RobustnessDials(BaseModel):
    """
    Configuration for robustness and uncertainty analysis.

    Controls the depth of counterfactual analysis and uncertainty quantification.
    """

    num_counterfactuals: int = Field(
        default=20,
        ge=0,
        le=200,
        description="Number of counterfactual scenarios to test"
    )

    perturbation_magnitude: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Default magnitude for value perturbations"
    )

    confidence_perturbation: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Magnitude for confidence score perturbations"
    )

    polarity_flip_probability: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Probability of testing polarity flips"
    )

    enable_uncertainty_decomposition: bool = Field(
        default=True,
        description="Enable epistemic/aleatoric uncertainty decomposition"
    )

    bootstrap_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of bootstrap samples for uncertainty estimation"
    )

    stability_margin_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Minimum stability margin for confident decisions"
    )

    min_confidence_for_decision: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to issue a decision vs defer"
    )

    sensitivity_analysis_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Depth of claim sensitivity analysis (0=disabled)"
    )


class HierarchyDials(BaseModel):
    """
    Configuration for hierarchical reasoning.

    Controls sector and stage-specific policy learning.
    """

    enable_hierarchy: bool = Field(
        default=True,
        description="Enable hierarchical (sector/stage) policy learning"
    )

    min_partition_size: int = Field(
        default=5,
        ge=2,
        le=50,
        description="Minimum examples required to create a partition"
    )

    max_hierarchy_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum depth of policy hierarchy"
    )

    partition_by: list[str] = Field(
        default_factory=lambda: ["sector", "stage"],
        description="Fields to partition by for hierarchical policies"
    )

    fallback_to_global: bool = Field(
        default=True,
        description="Fall back to global policy when partition is too small"
    )


class PerformanceDials(BaseModel):
    """
    Configuration for performance and resource usage.

    Controls parallelization and caching behavior.
    """

    enable_caching: bool = Field(
        default=True,
        description="Enable caching of intermediate results"
    )

    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds"
    )

    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum parallel workers for computation"
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for neural network inference"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Maximum execution time for a single job"
    )

    enable_progress_events: bool = Field(
        default=True,
        description="Emit progress events during execution"
    )

    event_granularity: str = Field(
        default="normal",
        description="Event emission granularity: minimal, normal, verbose"
    )

    @field_validator("event_granularity")
    @classmethod
    def validate_event_granularity(cls, v: str) -> str:
        valid = {"minimal", "normal", "verbose"}
        if v not in valid:
            raise ValueError(f"event_granularity must be one of {valid}")
        return v


class ProjectConfig(BaseModel):
    """
    Top-level project configuration.

    Aggregates all dial configurations with versioning and metadata.
    """

    config_version: str = Field(
        default=CONFIG_VERSION,
        description="Schema version for this configuration"
    )

    preset: PresetName = Field(
        default=PresetName.BALANCED,
        description="Base preset (values can be overridden)"
    )

    project_name: Optional[str] = Field(
        default=None,
        description="Optional project name for identification"
    )

    description: Optional[str] = Field(
        default=None,
        description="Optional description of this configuration"
    )

    context: ContextDials = Field(
        default_factory=ContextDials,
        description="Context building configuration"
    )

    search: SearchDials = Field(
        default_factory=SearchDials,
        description="Hypothesis search configuration"
    )

    numeric: NumericDials = Field(
        default_factory=NumericDials,
        description="Numeric threshold configuration"
    )

    robustness: RobustnessDials = Field(
        default_factory=RobustnessDials,
        description="Robustness analysis configuration"
    )

    hierarchy: HierarchyDials = Field(
        default_factory=HierarchyDials,
        description="Hierarchical reasoning configuration"
    )

    performance: PerformanceDials = Field(
        default_factory=PerformanceDials,
        description="Performance and resource configuration"
    )

    # Custom overrides (for advanced users)
    custom_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom key-value overrides for advanced tuning"
    )

    @model_validator(mode="after")
    def set_preset_marker(self) -> "ProjectConfig":
        """Mark as custom if any values differ from preset defaults."""
        # This is handled by the validation module
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        """Create from dictionary."""
        return cls.model_validate(data)

    def merge_with(self, overrides: dict[str, Any]) -> "ProjectConfig":
        """
        Create a new config by merging overrides onto this config.

        Args:
            overrides: Dictionary of values to override

        Returns:
            New ProjectConfig with merged values
        """
        from .validate import merge_configs
        return merge_configs(self, overrides)


# Type aliases for convenience
ConfigDict = dict[str, Any]
