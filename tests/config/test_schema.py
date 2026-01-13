"""Tests for configuration schema models."""

import pytest
import json
from pydantic import ValidationError

from juris_agi.config.schema import (
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


class TestContextDials:
    """Tests for ContextDials model."""

    def test_default_values(self):
        """Test default values are sensible."""
        dials = ContextDials()
        assert dials.max_claims == 50
        assert dials.min_confidence == 0.5
        assert dials.per_bucket_cap == 10
        assert dials.conflict_resolution == "highest_confidence"
        assert dials.include_low_confidence is False

    def test_range_validation_max_claims(self):
        """Test max_claims range validation."""
        # Valid values
        assert ContextDials(max_claims=10).max_claims == 10
        assert ContextDials(max_claims=500).max_claims == 500

        # Invalid values
        with pytest.raises(ValidationError):
            ContextDials(max_claims=5)  # Below minimum
        with pytest.raises(ValidationError):
            ContextDials(max_claims=1000)  # Above maximum

    def test_range_validation_min_confidence(self):
        """Test min_confidence range validation."""
        assert ContextDials(min_confidence=0.0).min_confidence == 0.0
        assert ContextDials(min_confidence=1.0).min_confidence == 1.0

        with pytest.raises(ValidationError):
            ContextDials(min_confidence=-0.1)
        with pytest.raises(ValidationError):
            ContextDials(min_confidence=1.5)

    def test_conflict_resolution_validation(self):
        """Test conflict_resolution enum validation."""
        valid_strategies = ["highest_confidence", "most_recent", "manual", "weighted_average"]
        for strategy in valid_strategies:
            dials = ContextDials(conflict_resolution=strategy)
            assert dials.conflict_resolution == strategy

        with pytest.raises(ValidationError):
            ContextDials(conflict_resolution="invalid_strategy")


class TestSearchDials:
    """Tests for SearchDials model."""

    def test_default_values(self):
        """Test default values are sensible."""
        dials = SearchDials()
        assert dials.max_hypotheses == 3
        assert dials.beam_width == 5
        assert dials.max_rules_per_policy == 10
        assert dials.use_neural_guidance is True

    def test_range_validation(self):
        """Test range validation for search parameters."""
        # Valid values
        assert SearchDials(max_hypotheses=1).max_hypotheses == 1
        assert SearchDials(max_hypotheses=20).max_hypotheses == 20
        assert SearchDials(beam_width=50).beam_width == 50

        # Invalid values
        with pytest.raises(ValidationError):
            SearchDials(max_hypotheses=0)
        with pytest.raises(ValidationError):
            SearchDials(beam_width=100)

    def test_weight_ranges(self):
        """Test weight parameter ranges."""
        dials = SearchDials(mdl_weight=0.0, coverage_weight=0.0)
        assert dials.mdl_weight == 0.0
        assert dials.coverage_weight == 0.0

        dials = SearchDials(mdl_weight=10.0, coverage_weight=10.0)
        assert dials.mdl_weight == 10.0

        with pytest.raises(ValidationError):
            SearchDials(mdl_weight=-1.0)


class TestNumericDials:
    """Tests for NumericDials model."""

    def test_default_values(self):
        """Test default values."""
        dials = NumericDials()
        assert dials.max_thresholds_per_field == 5
        assert dials.threshold_discovery_method == "information_gain"
        assert dials.use_trend_predicates is True

    def test_discovery_method_validation(self):
        """Test threshold discovery method validation."""
        valid_methods = ["information_gain", "percentile", "clustering", "entropy"]
        for method in valid_methods:
            dials = NumericDials(threshold_discovery_method=method)
            assert dials.threshold_discovery_method == method

        with pytest.raises(ValidationError):
            NumericDials(threshold_discovery_method="invalid_method")

    def test_percentile_candidates_default(self):
        """Test default percentile candidates."""
        dials = NumericDials()
        assert dials.percentile_candidates == [10, 25, 50, 75, 90]


class TestRobustnessDials:
    """Tests for RobustnessDials model."""

    def test_default_values(self):
        """Test default values."""
        dials = RobustnessDials()
        assert dials.num_counterfactuals == 20
        assert dials.enable_uncertainty_decomposition is True
        assert dials.bootstrap_samples == 100

    def test_range_validation(self):
        """Test range validation."""
        # Can have zero counterfactuals
        assert RobustnessDials(num_counterfactuals=0).num_counterfactuals == 0
        assert RobustnessDials(num_counterfactuals=200).num_counterfactuals == 200

        with pytest.raises(ValidationError):
            RobustnessDials(num_counterfactuals=-1)
        with pytest.raises(ValidationError):
            RobustnessDials(num_counterfactuals=500)


class TestHierarchyDials:
    """Tests for HierarchyDials model."""

    def test_default_values(self):
        """Test default values."""
        dials = HierarchyDials()
        assert dials.enable_hierarchy is True
        assert dials.min_partition_size == 5
        assert dials.partition_by == ["sector", "stage"]

    def test_range_validation(self):
        """Test range validation."""
        assert HierarchyDials(max_hierarchy_depth=1).max_hierarchy_depth == 1
        assert HierarchyDials(max_hierarchy_depth=5).max_hierarchy_depth == 5

        with pytest.raises(ValidationError):
            HierarchyDials(max_hierarchy_depth=0)


class TestPerformanceDials:
    """Tests for PerformanceDials model."""

    def test_default_values(self):
        """Test default values."""
        dials = PerformanceDials()
        assert dials.enable_caching is True
        assert dials.max_workers == 4
        assert dials.event_granularity == "normal"

    def test_event_granularity_validation(self):
        """Test event granularity validation."""
        for granularity in ["minimal", "normal", "verbose"]:
            dials = PerformanceDials(event_granularity=granularity)
            assert dials.event_granularity == granularity

        with pytest.raises(ValidationError):
            PerformanceDials(event_granularity="invalid")


class TestProjectConfig:
    """Tests for ProjectConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = ProjectConfig()
        assert config.config_version == CONFIG_VERSION
        assert config.preset == PresetName.BALANCED
        assert isinstance(config.context, ContextDials)
        assert isinstance(config.search, SearchDials)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ProjectConfig()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["config_version"] == CONFIG_VERSION
        assert "context" in data
        assert "search" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "config_version": CONFIG_VERSION,
            "preset": "fast",
            "context": {"max_claims": 30},
        }
        config = ProjectConfig.from_dict(data)

        assert config.preset == PresetName.FAST
        assert config.context.max_claims == 30

    def test_json_serialization(self):
        """Test JSON serialization roundtrip."""
        config = ProjectConfig(
            project_name="test_project",
            context=ContextDials(max_claims=75),
        )

        # Serialize to JSON
        json_str = json.dumps(config.to_dict())

        # Deserialize from JSON
        loaded = ProjectConfig.from_dict(json.loads(json_str))

        assert loaded.project_name == "test_project"
        assert loaded.context.max_claims == 75

    def test_custom_overrides(self):
        """Test custom overrides field."""
        config = ProjectConfig(
            custom_overrides={"experimental_feature": True, "debug_level": 3}
        )
        assert config.custom_overrides["experimental_feature"] is True
        assert config.custom_overrides["debug_level"] == 3

    def test_version_field(self):
        """Test config_version field."""
        config = ProjectConfig()
        assert config.config_version == CONFIG_VERSION

        # Can set explicit version
        config = ProjectConfig(config_version="1.0.0")
        assert config.config_version == "1.0.0"


class TestPresetName:
    """Tests for PresetName enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert PresetName.FAST.value == "fast"
        assert PresetName.BALANCED.value == "balanced"
        assert PresetName.THOROUGH.value == "thorough"
        assert PresetName.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert PresetName("fast") == PresetName.FAST
        assert PresetName("balanced") == PresetName.BALANCED
