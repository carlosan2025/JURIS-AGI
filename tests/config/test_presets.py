"""Tests for configuration presets."""

import pytest

from juris_agi.config.schema import CONFIG_VERSION, PresetName, ProjectConfig
from juris_agi.config.presets import (
    get_preset,
    get_fast_preset,
    get_balanced_preset,
    get_thorough_preset,
    list_presets,
    get_preset_comparison,
    PRESETS,
)


class TestPresetValidity:
    """Tests that all presets are valid and internally consistent."""

    @pytest.mark.parametrize("preset_name", [PresetName.FAST, PresetName.BALANCED, PresetName.THOROUGH])
    def test_preset_is_valid_config(self, preset_name):
        """Test that each preset produces a valid ProjectConfig."""
        config = get_preset(preset_name)
        assert isinstance(config, ProjectConfig)
        assert config.config_version == CONFIG_VERSION

    def test_fast_preset_validity(self):
        """Test fast preset is valid and has expected characteristics."""
        config = get_fast_preset()

        # Should have fast preset marker
        assert config.preset == PresetName.FAST

        # Should have lower limits for speed
        assert config.context.max_claims <= 50
        assert config.search.max_hypotheses <= 3
        assert config.robustness.num_counterfactuals <= 10
        assert config.hierarchy.enable_hierarchy is False
        assert config.performance.timeout_seconds <= 120

    def test_balanced_preset_validity(self):
        """Test balanced preset is valid and has expected characteristics."""
        config = get_balanced_preset()

        assert config.preset == PresetName.BALANCED

        # Should have moderate settings
        assert 30 <= config.context.max_claims <= 100
        assert 1 <= config.search.max_hypotheses <= 10
        assert config.robustness.enable_uncertainty_decomposition is True
        assert config.hierarchy.enable_hierarchy is True

    def test_thorough_preset_validity(self):
        """Test thorough preset is valid and has expected characteristics."""
        config = get_thorough_preset()

        assert config.preset == PresetName.THOROUGH

        # Should have higher limits for accuracy
        assert config.context.max_claims >= 50
        assert config.search.max_hypotheses >= 5
        assert config.robustness.num_counterfactuals >= 50
        assert config.robustness.bootstrap_samples >= 200
        assert config.hierarchy.enable_hierarchy is True

    def test_preset_ordering(self):
        """Test that presets have sensible ordering of key parameters."""
        fast = get_fast_preset()
        balanced = get_balanced_preset()
        thorough = get_thorough_preset()

        # Max claims should increase with thoroughness
        assert fast.context.max_claims <= balanced.context.max_claims <= thorough.context.max_claims

        # Max hypotheses should increase
        assert fast.search.max_hypotheses <= balanced.search.max_hypotheses <= thorough.search.max_hypotheses

        # Counterfactuals should increase
        assert fast.robustness.num_counterfactuals <= balanced.robustness.num_counterfactuals <= thorough.robustness.num_counterfactuals

        # Timeout should increase
        assert fast.performance.timeout_seconds <= balanced.performance.timeout_seconds <= thorough.performance.timeout_seconds


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_preset_by_enum(self):
        """Test getting preset by PresetName enum."""
        config = get_preset(PresetName.FAST)
        assert config.preset == PresetName.FAST

        config = get_preset(PresetName.BALANCED)
        assert config.preset == PresetName.BALANCED

        config = get_preset(PresetName.THOROUGH)
        assert config.preset == PresetName.THOROUGH

    def test_get_preset_by_string(self):
        """Test getting preset by string name."""
        config = get_preset("fast")
        assert config.preset == PresetName.FAST

        config = get_preset("balanced")
        assert config.preset == PresetName.BALANCED

        config = get_preset("thorough")
        assert config.preset == PresetName.THOROUGH

    def test_get_preset_case_insensitive(self):
        """Test that preset names are case-insensitive."""
        config1 = get_preset("FAST")
        config2 = get_preset("Fast")
        config3 = get_preset("fast")

        # All should produce same preset type
        assert config1.preset == config2.preset == config3.preset == PresetName.FAST

    def test_get_preset_custom(self):
        """Test getting CUSTOM preset returns balanced as base."""
        config = get_preset(PresetName.CUSTOM)
        # Custom starts from balanced
        assert config.context.max_claims == get_balanced_preset().context.max_claims

    def test_get_preset_invalid_name(self):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("invalid_preset")

        assert "Unknown preset" in str(exc_info.value)

    def test_all_presets_in_registry(self):
        """Test that all presets are registered."""
        assert PresetName.FAST in PRESETS
        assert PresetName.BALANCED in PRESETS
        assert PresetName.THOROUGH in PRESETS


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_presets_returns_list(self):
        """Test that list_presets returns a list."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) == 3  # fast, balanced, thorough

    def test_list_presets_structure(self):
        """Test that each preset info has required fields."""
        presets = list_presets()

        for preset_info in presets:
            assert "name" in preset_info
            assert "description" in preset_info
            assert "use_cases" in preset_info
            assert "tradeoffs" in preset_info

    def test_list_presets_names(self):
        """Test that preset names are correct."""
        presets = list_presets()
        names = [p["name"] for p in presets]

        assert "fast" in names
        assert "balanced" in names
        assert "thorough" in names


class TestGetPresetComparison:
    """Tests for get_preset_comparison function."""

    def test_comparison_returns_dict(self):
        """Test that comparison returns a dictionary."""
        comparison = get_preset_comparison()
        assert isinstance(comparison, dict)
        assert "fast" in comparison
        assert "balanced" in comparison
        assert "thorough" in comparison

    def test_comparison_structure(self):
        """Test comparison structure."""
        comparison = get_preset_comparison()

        for preset_name, values in comparison.items():
            assert "context" in values
            assert "search" in values
            assert "robustness" in values
            assert "hierarchy" in values
            assert "performance" in values

    def test_comparison_values(self):
        """Test that comparison values match actual presets."""
        comparison = get_preset_comparison()
        fast = get_fast_preset()
        balanced = get_balanced_preset()

        assert comparison["fast"]["context"]["max_claims"] == fast.context.max_claims
        assert comparison["balanced"]["search"]["max_hypotheses"] == balanced.search.max_hypotheses


class TestPresetConsistency:
    """Tests for preset internal consistency."""

    @pytest.mark.parametrize("preset_name", [PresetName.FAST, PresetName.BALANCED, PresetName.THOROUGH])
    def test_preset_all_dials_initialized(self, preset_name):
        """Test that all dials are initialized in each preset."""
        config = get_preset(preset_name)

        # All dial groups should exist and have values
        assert config.context is not None
        assert config.search is not None
        assert config.numeric is not None
        assert config.robustness is not None
        assert config.hierarchy is not None
        assert config.performance is not None

    @pytest.mark.parametrize("preset_name", [PresetName.FAST, PresetName.BALANCED, PresetName.THOROUGH])
    def test_preset_json_serializable(self, preset_name):
        """Test that presets can be serialized to JSON."""
        import json

        config = get_preset(preset_name)
        json_str = json.dumps(config.to_dict())

        # Should be valid JSON
        loaded = json.loads(json_str)
        assert isinstance(loaded, dict)
        assert loaded["preset"] == preset_name.value

    def test_presets_are_independent(self):
        """Test that presets don't share mutable state."""
        fast1 = get_fast_preset()
        fast2 = get_fast_preset()

        # Modify one
        fast1.context.max_claims = 999

        # Other should be unchanged
        assert fast2.context.max_claims != 999

    def test_preset_descriptions_not_empty(self):
        """Test that all presets have descriptions."""
        for preset_name in [PresetName.FAST, PresetName.BALANCED, PresetName.THOROUGH]:
            config = get_preset(preset_name)
            assert config.description is not None
            assert len(config.description) > 0
