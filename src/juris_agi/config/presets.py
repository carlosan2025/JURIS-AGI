"""
Configuration presets for JURIS-AGI.

Defines Fast, Balanced, and Thorough presets with sensible defaults
for different use cases.
"""

from __future__ import annotations

from typing import Any

from .schema import (
    CONFIG_VERSION,
    PresetName,
    ProjectConfig,
    ContextDials,
    SearchDials,
    NumericDials,
    RobustnessDials,
    HierarchyDials,
    PerformanceDials,
)


def get_fast_preset() -> ProjectConfig:
    """
    Fast preset optimized for quick turnaround.

    Use when:
    - You need rapid initial screening
    - Processing many deals in batch
    - Latency is critical
    - Accuracy can be traded for speed

    Tradeoffs:
    - Fewer hypotheses explored
    - Less counterfactual analysis
    - No hierarchy learning
    - Minimal uncertainty decomposition
    """
    return ProjectConfig(
        config_version=CONFIG_VERSION,
        preset=PresetName.FAST,
        description="Optimized for quick turnaround with acceptable accuracy",
        context=ContextDials(
            max_claims=30,
            min_confidence=0.6,
            per_bucket_cap=5,
            conflict_resolution="highest_confidence",
            include_low_confidence=False,
        ),
        search=SearchDials(
            max_hypotheses=1,
            beam_width=3,
            max_rules_per_policy=5,
            max_search_iterations=50,
            early_stopping_threshold=0.9,
            mdl_weight=1.0,
            coverage_weight=1.0,
            diversity_bonus=0.0,
            use_neural_guidance=True,
            neural_candidate_limit=25,
        ),
        numeric=NumericDials(
            max_thresholds_per_field=3,
            threshold_discovery_method="percentile",
            percentile_candidates=[25, 50, 75],
            min_samples_per_split=2,
            numeric_precision=1,
            use_trend_predicates=False,
        ),
        robustness=RobustnessDials(
            num_counterfactuals=5,
            perturbation_magnitude=0.15,
            confidence_perturbation=0.1,
            polarity_flip_probability=0.05,
            enable_uncertainty_decomposition=False,
            bootstrap_samples=20,
            stability_margin_threshold=0.15,
            min_confidence_for_decision=0.55,
            sensitivity_analysis_depth=0,
        ),
        hierarchy=HierarchyDials(
            enable_hierarchy=False,
            min_partition_size=10,
            max_hierarchy_depth=1,
            partition_by=[],
            fallback_to_global=True,
        ),
        performance=PerformanceDials(
            enable_caching=True,
            cache_ttl_seconds=1800,
            max_workers=4,
            batch_size=64,
            timeout_seconds=60,
            enable_progress_events=True,
            event_granularity="minimal",
        ),
    )


def get_balanced_preset() -> ProjectConfig:
    """
    Balanced preset for typical use cases.

    Use when:
    - Standard deal analysis workflow
    - Need reasonable accuracy with acceptable latency
    - Default choice for most situations

    Tradeoffs:
    - Moderate hypothesis exploration
    - Basic counterfactual analysis
    - Optional hierarchy learning
    - Standard uncertainty analysis
    """
    return ProjectConfig(
        config_version=CONFIG_VERSION,
        preset=PresetName.BALANCED,
        description="Balanced tradeoff between speed and accuracy for typical use",
        context=ContextDials(
            max_claims=50,
            min_confidence=0.5,
            per_bucket_cap=10,
            conflict_resolution="highest_confidence",
            include_low_confidence=False,
        ),
        search=SearchDials(
            max_hypotheses=3,
            beam_width=5,
            max_rules_per_policy=10,
            max_search_iterations=100,
            early_stopping_threshold=0.95,
            mdl_weight=1.0,
            coverage_weight=1.0,
            diversity_bonus=0.1,
            use_neural_guidance=True,
            neural_candidate_limit=50,
        ),
        numeric=NumericDials(
            max_thresholds_per_field=5,
            threshold_discovery_method="information_gain",
            percentile_candidates=[10, 25, 50, 75, 90],
            min_samples_per_split=3,
            numeric_precision=2,
            use_trend_predicates=True,
            trend_window_size=3,
        ),
        robustness=RobustnessDials(
            num_counterfactuals=20,
            perturbation_magnitude=0.2,
            confidence_perturbation=0.1,
            polarity_flip_probability=0.1,
            enable_uncertainty_decomposition=True,
            bootstrap_samples=100,
            stability_margin_threshold=0.1,
            min_confidence_for_decision=0.6,
            sensitivity_analysis_depth=2,
        ),
        hierarchy=HierarchyDials(
            enable_hierarchy=True,
            min_partition_size=5,
            max_hierarchy_depth=2,
            partition_by=["sector", "stage"],
            fallback_to_global=True,
        ),
        performance=PerformanceDials(
            enable_caching=True,
            cache_ttl_seconds=3600,
            max_workers=4,
            batch_size=32,
            timeout_seconds=300,
            enable_progress_events=True,
            event_granularity="normal",
        ),
    )


def get_thorough_preset() -> ProjectConfig:
    """
    Thorough preset optimized for maximum accuracy.

    Use when:
    - Final decision-making stage
    - High-stakes investment decisions
    - Regulatory or compliance requirements
    - Accuracy is paramount, latency is acceptable

    Tradeoffs:
    - Extensive hypothesis exploration
    - Deep counterfactual analysis
    - Full hierarchy learning
    - Comprehensive uncertainty decomposition
    """
    return ProjectConfig(
        config_version=CONFIG_VERSION,
        preset=PresetName.THOROUGH,
        description="Maximum accuracy for high-stakes decisions",
        context=ContextDials(
            max_claims=100,
            min_confidence=0.4,
            per_bucket_cap=20,
            conflict_resolution="weighted_average",
            include_low_confidence=True,
            low_confidence_weight=0.3,
        ),
        search=SearchDials(
            max_hypotheses=10,
            beam_width=10,
            max_rules_per_policy=20,
            max_search_iterations=500,
            early_stopping_threshold=0.99,
            mdl_weight=1.0,
            coverage_weight=1.2,
            diversity_bonus=0.2,
            use_neural_guidance=True,
            neural_candidate_limit=100,
        ),
        numeric=NumericDials(
            max_thresholds_per_field=10,
            threshold_discovery_method="information_gain",
            percentile_candidates=[5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95],
            min_samples_per_split=2,
            numeric_precision=3,
            use_trend_predicates=True,
            trend_window_size=6,
        ),
        robustness=RobustnessDials(
            num_counterfactuals=100,
            perturbation_magnitude=0.3,
            confidence_perturbation=0.15,
            polarity_flip_probability=0.2,
            enable_uncertainty_decomposition=True,
            bootstrap_samples=500,
            stability_margin_threshold=0.05,
            min_confidence_for_decision=0.65,
            sensitivity_analysis_depth=3,
        ),
        hierarchy=HierarchyDials(
            enable_hierarchy=True,
            min_partition_size=3,
            max_hierarchy_depth=3,
            partition_by=["sector", "stage", "geography"],
            fallback_to_global=True,
        ),
        performance=PerformanceDials(
            enable_caching=True,
            cache_ttl_seconds=7200,
            max_workers=8,
            batch_size=16,
            timeout_seconds=900,
            enable_progress_events=True,
            event_granularity="verbose",
        ),
    )


# Preset registry
PRESETS: dict[PresetName, callable] = {
    PresetName.FAST: get_fast_preset,
    PresetName.BALANCED: get_balanced_preset,
    PresetName.THOROUGH: get_thorough_preset,
}


def get_preset(name: PresetName | str) -> ProjectConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name (fast, balanced, thorough) or PresetName enum

    Returns:
        ProjectConfig with preset values

    Raises:
        ValueError: If preset name is not recognized
    """
    if isinstance(name, str):
        try:
            name = PresetName(name.lower())
        except ValueError:
            valid = [p.value for p in PresetName if p != PresetName.CUSTOM]
            raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")

    if name == PresetName.CUSTOM:
        # Custom means start from balanced and expect overrides
        return get_balanced_preset()

    if name not in PRESETS:
        raise ValueError(f"No preset factory for {name}")

    return PRESETS[name]()


def list_presets() -> list[dict[str, Any]]:
    """
    List all available presets with their descriptions.

    Returns:
        List of preset info dictionaries
    """
    return [
        {
            "name": PresetName.FAST.value,
            "description": "Optimized for quick turnaround with acceptable accuracy",
            "use_cases": [
                "Rapid initial screening",
                "Batch processing",
                "Low-latency requirements",
            ],
            "tradeoffs": "Fewer hypotheses, less robustness analysis",
        },
        {
            "name": PresetName.BALANCED.value,
            "description": "Balanced tradeoff between speed and accuracy",
            "use_cases": [
                "Standard deal analysis",
                "Day-to-day operations",
                "Default choice",
            ],
            "tradeoffs": "Moderate exploration, standard analysis depth",
        },
        {
            "name": PresetName.THOROUGH.value,
            "description": "Maximum accuracy for high-stakes decisions",
            "use_cases": [
                "Final decision stage",
                "High-stakes investments",
                "Compliance requirements",
            ],
            "tradeoffs": "Longer execution time, more resources",
        },
    ]


def get_preset_comparison() -> dict[str, dict[str, Any]]:
    """
    Get a comparison table of key parameters across presets.

    Returns:
        Dictionary mapping preset names to key parameter values
    """
    comparison = {}

    for preset_name in [PresetName.FAST, PresetName.BALANCED, PresetName.THOROUGH]:
        config = get_preset(preset_name)
        comparison[preset_name.value] = {
            "context": {
                "max_claims": config.context.max_claims,
                "min_confidence": config.context.min_confidence,
            },
            "search": {
                "max_hypotheses": config.search.max_hypotheses,
                "beam_width": config.search.beam_width,
                "max_iterations": config.search.max_search_iterations,
            },
            "robustness": {
                "num_counterfactuals": config.robustness.num_counterfactuals,
                "uncertainty_decomposition": config.robustness.enable_uncertainty_decomposition,
            },
            "hierarchy": {
                "enabled": config.hierarchy.enable_hierarchy,
                "max_depth": config.hierarchy.max_hierarchy_depth,
            },
            "performance": {
                "timeout_seconds": config.performance.timeout_seconds,
                "event_granularity": config.performance.event_granularity,
            },
        }

    return comparison
