"""Tests for configuration validation and merging."""

import pytest

from juris_agi.config.schema import (
    CONFIG_VERSION,
    PresetName,
    ProjectConfig,
    ContextDials,
    SearchDials,
)
from juris_agi.config.presets import get_preset, get_balanced_preset
from juris_agi.config.validate import (
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


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_valid_config(self):
        """Test validating a valid config."""
        config = ProjectConfig()
        validated = validate_config(config)
        assert isinstance(validated, ProjectConfig)

    def test_validate_dict_config(self):
        """Test validating a dictionary config."""
        data = {
            "config_version": CONFIG_VERSION,
            "preset": "balanced",
            "context": {"max_claims": 60},
        }
        config = validate_config(data)
        assert isinstance(config, ProjectConfig)
        assert config.context.max_claims == 60

    def test_validate_invalid_range(self):
        """Test that invalid ranges raise ConfigValidationError."""
        data = {
            "config_version": CONFIG_VERSION,
            "context": {"max_claims": 5},  # Below minimum
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(data)

        assert len(exc_info.value.errors) > 0

    def test_validate_semantic_error_low_confidence_weight(self):
        """Test semantic validation for low_confidence_weight."""
        config = ProjectConfig(
            context=ContextDials(
                include_low_confidence=True,
                low_confidence_weight=0.0,  # Invalid: must be > 0 when enabled
            )
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        errors = exc_info.value.errors
        assert any("low_confidence_weight" in str(e.get("loc", [])) for e in errors)

    def test_validate_semantic_error_hierarchy_partition(self):
        """Test semantic validation for hierarchy partition_by."""
        config = ProjectConfig()
        config.hierarchy.enable_hierarchy = True
        config.hierarchy.partition_by = []  # Invalid: must not be empty

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        errors = exc_info.value.errors
        assert any("partition_by" in str(e.get("loc", [])) for e in errors)

    def test_validate_semantic_warning_weights(self):
        """Test semantic validation for zero weights."""
        config = ProjectConfig()
        config.search.mdl_weight = 0.0
        config.search.coverage_weight = 0.0

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        errors = exc_info.value.errors
        assert any("cannot both be zero" in str(e.get("msg", "")) for e in errors)


class TestVersionCompatibility:
    """Tests for version compatibility checking."""

    def test_current_version_compatible(self):
        """Test that current version is compatible."""
        assert is_version_compatible(CONFIG_VERSION) is True

    def test_same_major_minor_compatible(self):
        """Test that same major.minor is compatible."""
        assert is_version_compatible("1.0.0") is True
        assert is_version_compatible("1.0.1") is True
        assert is_version_compatible("1.0.99") is True

    def test_lower_minor_compatible(self):
        """Test that lower minor version is compatible."""
        # If current is 1.0.x, then 1.0.0 should be compatible
        assert is_version_compatible("1.0.0") is True

    def test_different_major_incompatible(self):
        """Test that different major version is incompatible."""
        assert is_version_compatible("2.0.0") is False
        assert is_version_compatible("0.9.0") is False

    def test_invalid_version_format(self):
        """Test that invalid version format returns False."""
        assert is_version_compatible("invalid") is False
        assert is_version_compatible("1.0") is False
        assert is_version_compatible("") is False

    def test_version_error_raised(self):
        """Test that version mismatch raises ConfigVersionError."""
        data = {"config_version": "2.0.0"}
        with pytest.raises(ConfigVersionError) as exc_info:
            validate_config(data)

        assert exc_info.value.found_version == "2.0.0"
        assert exc_info.value.expected_version == CONFIG_VERSION


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_empty_overrides(self):
        """Test merging with empty overrides."""
        base = get_balanced_preset()
        merged = merge_configs(base, {})

        assert merged.context.max_claims == base.context.max_claims

    def test_merge_simple_override(self):
        """Test merging a simple override."""
        base = get_balanced_preset()
        overrides = {"context": {"max_claims": 75}}
        merged = merge_configs(base, overrides)

        assert merged.context.max_claims == 75
        # Other values should be preserved
        assert merged.context.min_confidence == base.context.min_confidence

    def test_merge_nested_override(self):
        """Test merging nested overrides."""
        base = get_balanced_preset()
        overrides = {
            "context": {"max_claims": 80},
            "search": {"max_hypotheses": 7, "beam_width": 8},
        }
        merged = merge_configs(base, overrides)

        assert merged.context.max_claims == 80
        assert merged.search.max_hypotheses == 7
        assert merged.search.beam_width == 8

    def test_merge_marks_as_custom(self):
        """Test that merging marks config as custom."""
        base = get_balanced_preset()
        overrides = {"context": {"max_claims": 75}}
        merged = merge_configs(base, overrides)

        assert merged.preset == PresetName.CUSTOM

    def test_merge_preserves_base(self):
        """Test that merging doesn't modify the base config."""
        base = get_balanced_preset()
        original_max_claims = base.context.max_claims

        overrides = {"context": {"max_claims": 200}}
        merge_configs(base, overrides)

        # Base should be unchanged
        assert base.context.max_claims == original_max_claims

    def test_merge_without_validation(self):
        """Test merging without validation."""
        base = ProjectConfig()
        # This would fail validation (weights both zero)
        overrides = {
            "search": {"mdl_weight": 0.0, "coverage_weight": 0.0}
        }

        # Should succeed without validation
        merged = merge_configs(base, overrides, validate=False)
        assert merged.search.mdl_weight == 0.0


class TestNormalizeConfig:
    """Tests for normalize_config function."""

    def test_normalize_updates_version(self):
        """Test that normalization updates version."""
        config = ProjectConfig(config_version="0.9.0")
        normalized = normalize_config(config)
        assert normalized.config_version == CONFIG_VERSION

    def test_normalize_single_hypothesis_diversity(self):
        """Test normalization for single hypothesis."""
        config = ProjectConfig()
        config.search.max_hypotheses = 1
        config.search.diversity_bonus = 0.5  # Will be reset

        normalized = normalize_config(config)
        assert normalized.search.diversity_bonus == 0.0

    def test_normalize_disabled_hierarchy(self):
        """Test normalization for disabled hierarchy."""
        config = ProjectConfig()
        config.hierarchy.enable_hierarchy = False
        config.hierarchy.partition_by = ["sector", "stage"]

        normalized = normalize_config(config)
        assert normalized.hierarchy.partition_by == []

    def test_normalize_disabled_uncertainty(self):
        """Test normalization for disabled uncertainty decomposition."""
        config = ProjectConfig()
        config.robustness.enable_uncertainty_decomposition = False
        config.robustness.bootstrap_samples = 500

        normalized = normalize_config(config)
        assert normalized.robustness.bootstrap_samples <= 20


class TestCreateConfig:
    """Tests for create_config function."""

    def test_create_from_preset_name(self):
        """Test creating config from preset name."""
        config = create_config(preset="fast")
        assert config.preset == PresetName.FAST

    def test_create_from_preset_enum(self):
        """Test creating config from PresetName enum."""
        config = create_config(preset=PresetName.THOROUGH)
        assert config.preset == PresetName.THOROUGH

    def test_create_with_overrides(self):
        """Test creating config with overrides."""
        config = create_config(
            preset="balanced",
            overrides={"context": {"max_claims": 100}}
        )
        assert config.context.max_claims == 100
        assert config.preset == PresetName.CUSTOM

    def test_create_with_validation(self):
        """Test that create_config validates by default."""
        with pytest.raises(ConfigValidationError):
            create_config(
                preset="balanced",
                overrides={"search": {"mdl_weight": 0.0, "coverage_weight": 0.0}}
            )

    def test_create_without_validation(self):
        """Test creating config without validation."""
        config = create_config(
            preset="balanced",
            overrides={"search": {"mdl_weight": 0.0, "coverage_weight": 0.0}},
            validate=False,
        )
        assert config.search.mdl_weight == 0.0

    def test_create_with_normalization(self):
        """Test that create_config normalizes by default."""
        config = create_config(
            preset="balanced",
            overrides={"search": {"max_hypotheses": 1}}
        )
        # Diversity bonus should be normalized to 0
        assert config.search.diversity_bonus == 0.0

    def test_create_without_normalization(self):
        """Test creating config without normalization."""
        config = create_config(
            preset="balanced",
            overrides={"search": {"max_hypotheses": 1, "diversity_bonus": 0.5}},
            normalize=False,
        )
        # Should preserve the diversity bonus
        assert config.search.diversity_bonus == 0.5


class TestDiffConfigs:
    """Tests for diff_configs function."""

    def test_diff_identical_configs(self):
        """Test diffing identical configs."""
        config1 = get_balanced_preset()
        config2 = get_balanced_preset()
        diff = diff_configs(config1, config2)

        # Should be empty (no differences)
        assert diff == {}

    def test_diff_simple_change(self):
        """Test diffing configs with simple change."""
        config1 = get_balanced_preset()
        config2 = get_balanced_preset()
        config2.context.max_claims = 999

        diff = diff_configs(config1, config2)

        assert "context" in diff
        assert "max_claims" in diff["context"]
        assert diff["context"]["max_claims"]["to"] == 999

    def test_diff_multiple_changes(self):
        """Test diffing configs with multiple changes."""
        config1 = get_balanced_preset()
        config2 = get_balanced_preset()
        config2.context.max_claims = 100
        config2.search.beam_width = 10

        diff = diff_configs(config1, config2)

        assert "context" in diff
        assert "search" in diff

    def test_diff_presets(self):
        """Test diffing different presets."""
        fast = get_preset("fast")
        thorough = get_preset("thorough")

        diff = diff_configs(fast, thorough)

        # Should have many differences
        assert len(diff) > 0
        assert "context" in diff
        assert "search" in diff


class TestGetRangeInfo:
    """Tests for get_range_info function."""

    def test_get_known_field_range(self):
        """Test getting range info for known field."""
        info = get_range_info("context.max_claims")
        assert info is not None
        assert info["min"] == 10
        assert info["max"] == 500
        assert info["default"] == 50
        assert info["type"] == "int"

    def test_get_float_field_range(self):
        """Test getting range info for float field."""
        info = get_range_info("context.min_confidence")
        assert info is not None
        assert info["min"] == 0.0
        assert info["max"] == 1.0
        assert info["type"] == "float"

    def test_get_unknown_field_range(self):
        """Test getting range info for unknown field."""
        info = get_range_info("unknown.field.path")
        assert info is None


class TestClampToRange:
    """Tests for clamp_to_range function."""

    def test_clamp_below_min(self):
        """Test clamping value below minimum."""
        result = clamp_to_range("context.max_claims", 5)
        assert result == 10  # Clamped to minimum

    def test_clamp_above_max(self):
        """Test clamping value above maximum."""
        result = clamp_to_range("context.max_claims", 1000)
        assert result == 500  # Clamped to maximum

    def test_clamp_within_range(self):
        """Test clamping value within range."""
        result = clamp_to_range("context.max_claims", 100)
        assert result == 100  # Unchanged

    def test_clamp_float_value(self):
        """Test clamping float value."""
        result = clamp_to_range("context.min_confidence", 1.5)
        assert result == 1.0  # Clamped to maximum

    def test_clamp_unknown_field(self):
        """Test clamping unknown field returns original."""
        result = clamp_to_range("unknown.field", 999)
        assert result == 999  # Unchanged
