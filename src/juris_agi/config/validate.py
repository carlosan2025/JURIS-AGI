"""
Configuration validation and normalization for JURIS-AGI.

Provides utilities for validating configuration ranges, merging overrides,
and normalizing configurations.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from pydantic import ValidationError

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
from .presets import get_preset


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, errors: Optional[list[dict]] = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigVersionError(Exception):
    """Raised when configuration version is incompatible."""

    def __init__(self, found_version: str, expected_version: str):
        super().__init__(
            f"Configuration version mismatch: found {found_version}, "
            f"expected {expected_version}"
        )
        self.found_version = found_version
        self.expected_version = expected_version


def validate_config(config: ProjectConfig | dict[str, Any]) -> ProjectConfig:
    """
    Validate a configuration and return a normalized ProjectConfig.

    Args:
        config: ProjectConfig instance or dictionary

    Returns:
        Validated ProjectConfig

    Raises:
        ConfigValidationError: If validation fails
        ConfigVersionError: If version is incompatible
    """
    # Convert dict to ProjectConfig if needed
    if isinstance(config, dict):
        try:
            config = ProjectConfig.model_validate(config)
        except ValidationError as e:
            errors = [
                {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
                for err in e.errors()
            ]
            raise ConfigValidationError(
                f"Configuration validation failed with {len(errors)} error(s)",
                errors=errors,
            )

    # Check version compatibility
    if not is_version_compatible(config.config_version):
        raise ConfigVersionError(config.config_version, CONFIG_VERSION)

    # Run additional semantic validations
    semantic_errors = _validate_semantic_constraints(config)
    if semantic_errors:
        raise ConfigValidationError(
            f"Semantic validation failed with {len(semantic_errors)} error(s)",
            errors=semantic_errors,
        )

    return config


def is_version_compatible(version: str) -> bool:
    """
    Check if a configuration version is compatible with current schema.

    Uses semantic versioning: major versions must match, minor can differ.

    Args:
        version: Version string to check

    Returns:
        True if compatible
    """
    try:
        current_parts = CONFIG_VERSION.split(".")
        check_parts = version.split(".")

        # Must have exactly 3 parts (major.minor.patch)
        if len(check_parts) != 3:
            return False

        # Major version must match
        if current_parts[0] != check_parts[0]:
            return False

        # Minor version of config can be less than or equal to current
        if int(check_parts[1]) > int(current_parts[1]):
            return False

        return True
    except (ValueError, IndexError):
        return False


def _validate_semantic_constraints(config: ProjectConfig) -> list[dict]:
    """
    Validate semantic constraints that can't be expressed in pydantic schema.

    Args:
        config: Configuration to validate

    Returns:
        List of error dictionaries (empty if valid)
    """
    errors = []

    # Context constraints
    if config.context.include_low_confidence:
        if config.context.low_confidence_weight <= 0:
            errors.append({
                "loc": ["context", "low_confidence_weight"],
                "msg": "low_confidence_weight must be > 0 when include_low_confidence is True",
                "type": "semantic_error",
            })

    # Search constraints
    if config.search.beam_width > config.search.max_search_iterations:
        errors.append({
            "loc": ["search"],
            "msg": "beam_width should not exceed max_search_iterations",
            "type": "semantic_warning",
        })

    total_weight = config.search.mdl_weight + config.search.coverage_weight
    if total_weight == 0:
        errors.append({
            "loc": ["search"],
            "msg": "mdl_weight and coverage_weight cannot both be zero",
            "type": "semantic_error",
        })

    # Numeric constraints
    if config.numeric.use_trend_predicates:
        if config.numeric.trend_window_size < 2:
            errors.append({
                "loc": ["numeric", "trend_window_size"],
                "msg": "trend_window_size must be >= 2 when trend predicates enabled",
                "type": "semantic_error",
            })

    # Robustness constraints
    if config.robustness.sensitivity_analysis_depth > 0:
        if config.robustness.num_counterfactuals < 5:
            errors.append({
                "loc": ["robustness"],
                "msg": "num_counterfactuals should be >= 5 for sensitivity analysis",
                "type": "semantic_warning",
            })

    # Hierarchy constraints
    if config.hierarchy.enable_hierarchy:
        if not config.hierarchy.partition_by:
            errors.append({
                "loc": ["hierarchy", "partition_by"],
                "msg": "partition_by must not be empty when hierarchy is enabled",
                "type": "semantic_error",
            })

    # Performance constraints
    if config.performance.timeout_seconds < 30:
        if config.search.max_search_iterations > 100:
            errors.append({
                "loc": ["performance", "timeout_seconds"],
                "msg": "timeout_seconds may be too low for max_search_iterations > 100",
                "type": "semantic_warning",
            })

    return errors


def merge_configs(
    base: ProjectConfig,
    overrides: dict[str, Any],
    validate: bool = True,
) -> ProjectConfig:
    """
    Merge overrides onto a base configuration.

    Args:
        base: Base configuration to start from
        overrides: Dictionary of values to override
        validate: Whether to validate the merged result

    Returns:
        New ProjectConfig with merged values

    Raises:
        ConfigValidationError: If validation fails (when validate=True)
    """
    # Convert base to dict
    base_dict = base.model_dump()

    # Deep merge overrides
    merged = _deep_merge(base_dict, overrides)

    # Mark as custom if overrides were applied
    if overrides:
        merged["preset"] = PresetName.CUSTOM.value

    # Create new config
    if validate:
        return validate_config(merged)
    else:
        return ProjectConfig.model_validate(merged)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        overrides: Dictionary of overrides

    Returns:
        New merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def normalize_config(config: ProjectConfig) -> ProjectConfig:
    """
    Normalize a configuration to ensure consistency.

    Applies any transformations needed to make the config internally consistent.

    Args:
        config: Configuration to normalize

    Returns:
        Normalized configuration
    """
    # Create mutable copy
    data = config.model_dump()

    # Ensure version is current
    data["config_version"] = CONFIG_VERSION

    # Normalize context
    if data["context"]["min_confidence"] > 0.9:
        # Very high min_confidence might filter too aggressively
        data["context"]["per_bucket_cap"] = max(
            data["context"]["per_bucket_cap"],
            15  # Ensure at least some claims per bucket
        )

    # Normalize search
    if data["search"]["max_hypotheses"] == 1:
        # Single hypothesis means no diversity needed
        data["search"]["diversity_bonus"] = 0.0

    # Normalize robustness
    if not data["robustness"]["enable_uncertainty_decomposition"]:
        # Bootstrap samples not needed without uncertainty decomposition
        data["robustness"]["bootstrap_samples"] = min(
            data["robustness"]["bootstrap_samples"],
            20
        )

    # Normalize hierarchy
    if not data["hierarchy"]["enable_hierarchy"]:
        data["hierarchy"]["partition_by"] = []

    return ProjectConfig.model_validate(data)


def create_config(
    preset: PresetName | str = PresetName.BALANCED,
    overrides: Optional[dict[str, Any]] = None,
    validate: bool = True,
    normalize: bool = True,
) -> ProjectConfig:
    """
    Create a configuration from a preset with optional overrides.

    This is the recommended way to create configurations.

    Args:
        preset: Base preset to start from
        overrides: Optional dictionary of overrides
        validate: Whether to validate the config
        normalize: Whether to normalize the config

    Returns:
        ProjectConfig instance

    Example:
        ```python
        # Create a fast preset with more hypotheses
        config = create_config(
            preset="fast",
            overrides={"search": {"max_hypotheses": 5}}
        )
        ```
    """
    # Get base preset
    base = get_preset(preset)

    # Apply overrides if provided
    if overrides:
        config = merge_configs(base, overrides, validate=validate)
    else:
        config = base
        if validate:
            config = validate_config(config)

    # Normalize if requested
    if normalize:
        config = normalize_config(config)

    return config


def diff_configs(
    config1: ProjectConfig,
    config2: ProjectConfig,
) -> dict[str, Any]:
    """
    Compute the difference between two configurations.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        Dictionary showing differences (values from config2 that differ from config1)
    """
    dict1 = config1.model_dump()
    dict2 = config2.model_dump()

    return _diff_dicts(dict1, dict2)


def _diff_dicts(dict1: dict, dict2: dict, path: str = "") -> dict:
    """
    Recursively compute difference between two dictionaries.
    """
    diff = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        if key not in dict1:
            diff[key] = {"added": dict2[key]}
        elif key not in dict2:
            diff[key] = {"removed": dict1[key]}
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_diff = _diff_dicts(dict1[key], dict2[key], current_path)
            if nested_diff:
                diff[key] = nested_diff
        elif dict1[key] != dict2[key]:
            diff[key] = {"from": dict1[key], "to": dict2[key]}

    return diff


def get_range_info(field_path: str) -> Optional[dict[str, Any]]:
    """
    Get range information for a configuration field.

    Args:
        field_path: Dot-separated path to field (e.g., "context.max_claims")

    Returns:
        Dictionary with min, max, default values, or None if field not found
    """
    # Map of field paths to their constraints
    FIELD_RANGES = {
        "context.max_claims": {"min": 10, "max": 500, "default": 50, "type": "int"},
        "context.min_confidence": {"min": 0.0, "max": 1.0, "default": 0.5, "type": "float"},
        "context.per_bucket_cap": {"min": 1, "max": 50, "default": 10, "type": "int"},
        "context.low_confidence_weight": {"min": 0.0, "max": 1.0, "default": 0.3, "type": "float"},
        "search.max_hypotheses": {"min": 1, "max": 20, "default": 3, "type": "int"},
        "search.beam_width": {"min": 1, "max": 50, "default": 5, "type": "int"},
        "search.max_rules_per_policy": {"min": 1, "max": 50, "default": 10, "type": "int"},
        "search.max_search_iterations": {"min": 10, "max": 1000, "default": 100, "type": "int"},
        "search.early_stopping_threshold": {"min": 0.5, "max": 1.0, "default": 0.95, "type": "float"},
        "search.mdl_weight": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
        "search.coverage_weight": {"min": 0.0, "max": 10.0, "default": 1.0, "type": "float"},
        "search.diversity_bonus": {"min": 0.0, "max": 1.0, "default": 0.1, "type": "float"},
        "search.neural_candidate_limit": {"min": 10, "max": 500, "default": 50, "type": "int"},
        "numeric.max_thresholds_per_field": {"min": 1, "max": 20, "default": 5, "type": "int"},
        "numeric.min_samples_per_split": {"min": 1, "max": 50, "default": 3, "type": "int"},
        "numeric.numeric_precision": {"min": 0, "max": 6, "default": 2, "type": "int"},
        "numeric.trend_window_size": {"min": 2, "max": 12, "default": 3, "type": "int"},
        "robustness.num_counterfactuals": {"min": 0, "max": 200, "default": 20, "type": "int"},
        "robustness.perturbation_magnitude": {"min": 0.0, "max": 1.0, "default": 0.2, "type": "float"},
        "robustness.confidence_perturbation": {"min": 0.0, "max": 0.5, "default": 0.1, "type": "float"},
        "robustness.polarity_flip_probability": {"min": 0.0, "max": 0.5, "default": 0.1, "type": "float"},
        "robustness.bootstrap_samples": {"min": 10, "max": 1000, "default": 100, "type": "int"},
        "robustness.stability_margin_threshold": {"min": 0.0, "max": 0.5, "default": 0.1, "type": "float"},
        "robustness.min_confidence_for_decision": {"min": 0.0, "max": 1.0, "default": 0.6, "type": "float"},
        "robustness.sensitivity_analysis_depth": {"min": 0, "max": 5, "default": 2, "type": "int"},
        "hierarchy.min_partition_size": {"min": 2, "max": 50, "default": 5, "type": "int"},
        "hierarchy.max_hierarchy_depth": {"min": 1, "max": 5, "default": 2, "type": "int"},
        "performance.cache_ttl_seconds": {"min": 60, "max": 86400, "default": 3600, "type": "int"},
        "performance.max_workers": {"min": 1, "max": 32, "default": 4, "type": "int"},
        "performance.batch_size": {"min": 1, "max": 256, "default": 32, "type": "int"},
        "performance.timeout_seconds": {"min": 30, "max": 3600, "default": 300, "type": "int"},
    }

    return FIELD_RANGES.get(field_path)


def clamp_to_range(field_path: str, value: Any) -> Any:
    """
    Clamp a value to its valid range.

    Args:
        field_path: Dot-separated path to field
        value: Value to clamp

    Returns:
        Clamped value, or original if field not found
    """
    range_info = get_range_info(field_path)
    if range_info is None:
        return value

    if range_info["type"] == "int":
        return max(range_info["min"], min(range_info["max"], int(value)))
    elif range_info["type"] == "float":
        return max(range_info["min"], min(range_info["max"], float(value)))

    return value
