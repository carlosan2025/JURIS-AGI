"""
JURIS-AGI Configuration System.

Provides project-level configuration with scaling vs accuracy tradeoffs.

Quick Start:
    ```python
    from juris_agi.config import create_config, PresetName

    # Use a preset
    config = create_config(preset="fast")

    # Customize a preset
    config = create_config(
        preset="balanced",
        overrides={"search": {"max_hypotheses": 5}}
    )

    # Access settings
    print(config.context.max_claims)
    print(config.search.beam_width)
    ```

Available Presets:
    - fast: Quick turnaround, acceptable accuracy
    - balanced: Standard tradeoff for typical use
    - thorough: Maximum accuracy for high-stakes decisions
"""

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
    ConfigDict,
)

from .presets import (
    get_preset,
    get_fast_preset,
    get_balanced_preset,
    get_thorough_preset,
    list_presets,
    get_preset_comparison,
    PRESETS,
)

from .validate import (
    ConfigValidationError,
    ConfigVersionError,
    validate_config,
    merge_configs,
    normalize_config,
    create_config,
    diff_configs,
    get_range_info,
    clamp_to_range,
    is_version_compatible,
)

__all__ = [
    # Version
    "CONFIG_VERSION",
    # Schema
    "PresetName",
    "ProjectConfig",
    "ContextDials",
    "SearchDials",
    "NumericDials",
    "RobustnessDials",
    "HierarchyDials",
    "PerformanceDials",
    "ConfigDict",
    # Presets
    "get_preset",
    "get_fast_preset",
    "get_balanced_preset",
    "get_thorough_preset",
    "list_presets",
    "get_preset_comparison",
    "PRESETS",
    # Validation
    "ConfigValidationError",
    "ConfigVersionError",
    "validate_config",
    "merge_configs",
    "normalize_config",
    "create_config",
    "diff_configs",
    "get_range_info",
    "clamp_to_range",
    "is_version_compatible",
]
