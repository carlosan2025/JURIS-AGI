"""Tests for Threshold Proposer module."""

import pytest

from juris_agi.vc_dsl.thresholds import (
    ThresholdReason,
    ThresholdCandidate,
    ThresholdTrace,
    ThresholdPrior,
    ThresholdProposerConfig,
    propose_thresholds,
    propose_thresholds_for_template,
    propose_all_thresholds,
    get_thresholds_only,
    default_thresholds_for_field,
    FieldObservations,
    merge_threshold_traces,
    DOMAIN_HEURISTICS,
    _round_to_nice_number,
    _compute_quantiles,
    _get_domain_heuristics,
)


class TestThresholdCandidate:
    """Tests for ThresholdCandidate dataclass."""

    def test_candidate_creation(self):
        candidate = ThresholdCandidate(
            value=100.0,
            reason=ThresholdReason.QUANTILE,
            description="P50",
            confidence=0.9,
        )
        assert candidate.value == 100.0
        assert candidate.reason == ThresholdReason.QUANTILE
        assert candidate.description == "P50"
        assert candidate.confidence == 0.9

    def test_candidate_equality(self):
        c1 = ThresholdCandidate(100.0, ThresholdReason.QUANTILE)
        c2 = ThresholdCandidate(100.0, ThresholdReason.DOMAIN_HEURISTIC)
        c3 = ThresholdCandidate(200.0, ThresholdReason.QUANTILE)

        assert c1 == c2  # Same value, different reason
        assert c1 != c3  # Different value

    def test_candidate_hash(self):
        c1 = ThresholdCandidate(100.0, ThresholdReason.QUANTILE)
        c2 = ThresholdCandidate(100.0, ThresholdReason.DOMAIN_HEURISTIC)

        assert hash(c1) == hash(c2)
        assert {c1, c2} == {c1}  # Set deduplication


class TestRoundToNiceNumber:
    """Tests for rounding to nice numbers."""

    def test_round_zero(self):
        assert _round_to_nice_number(0) == 0.0

    def test_round_to_1(self):
        assert _round_to_nice_number(0.8) == 1.0
        assert _round_to_nice_number(1.2) == 1.0

    def test_round_to_2(self):
        assert _round_to_nice_number(1.8) == 2.0
        assert _round_to_nice_number(2.2) == 2.0

    def test_round_to_5(self):
        assert _round_to_nice_number(4.0) == 5.0
        assert _round_to_nice_number(5.5) == 5.0

    def test_round_to_10(self):
        assert _round_to_nice_number(8.0) == 10.0
        assert _round_to_nice_number(9.5) == 10.0

    def test_round_large_numbers(self):
        assert _round_to_nice_number(850000) == 1000000.0
        assert _round_to_nice_number(450000) == 500000.0
        assert _round_to_nice_number(2200000) == 2000000.0

    def test_round_small_numbers(self):
        assert _round_to_nice_number(0.045) == 0.05
        assert _round_to_nice_number(0.008) == 0.01

    def test_round_negative(self):
        assert _round_to_nice_number(-5.0) == -5.0
        assert _round_to_nice_number(-850) == -1000.0


class TestComputeQuantiles:
    """Tests for quantile computation."""

    def test_empty_values(self):
        assert _compute_quantiles([]) == []

    def test_single_value(self):
        result = _compute_quantiles([100.0], (0.5,))
        assert result == [100.0]

    def test_median(self):
        result = _compute_quantiles([1, 2, 3, 4, 5], (0.5,))
        assert result == [3.0]

    def test_quartiles(self):
        result = _compute_quantiles([1, 2, 3, 4, 5], (0.25, 0.5, 0.75))
        assert result[0] == 2.0  # Q1
        assert result[1] == 3.0  # Median
        assert result[2] == 4.0  # Q3

    def test_percentiles(self):
        values = list(range(1, 101))  # 1 to 100
        result = _compute_quantiles(values, (0.1, 0.5, 0.9))
        assert abs(result[0] - 10.9) < 0.1  # P10 ≈ 10
        assert abs(result[1] - 50.5) < 0.1  # P50 ≈ 50
        assert abs(result[2] - 90.1) < 0.1  # P90 ≈ 90


class TestGetDomainHeuristics:
    """Tests for domain heuristic lookup."""

    def test_direct_match(self):
        result = _get_domain_heuristics("gross_margin")
        assert len(result) > 0
        assert any(0.40 in [t[0] for t in result] for _ in [1])

    def test_suffix_match(self):
        result = _get_domain_heuristics("business_model.gross_margin")
        assert len(result) > 0

    def test_contains_match(self):
        result = _get_domain_heuristics("traction.arr")
        assert len(result) > 0
        assert any(t[0] == 1_000_000 for t in result)

    def test_no_match(self):
        result = _get_domain_heuristics("unknown_field_xyz")
        assert result == []


class TestProposeThresholds:
    """Tests for the main propose_thresholds function."""

    def test_propose_with_observed_values(self):
        observed = [100, 200, 300, 400, 500]
        thresholds, trace = propose_thresholds("some_field", observed)

        assert len(thresholds) > 0
        assert trace is not None
        assert trace.observed_count == 5
        assert trace.observed_min == 100
        assert trace.observed_max == 500

    def test_propose_with_domain_heuristics(self):
        # ARR field should get domain heuristics
        thresholds, trace = propose_thresholds("traction.arr", [])

        assert len(thresholds) > 0
        assert trace is not None
        # Should include standard ARR thresholds
        assert any(t >= 1_000_000 for t in thresholds)

    def test_propose_with_priors(self):
        prior = ThresholdPrior(
            field="custom_field",
            suggested_thresholds=[25, 50, 75],
            description="Custom prior",
        )
        thresholds, trace = propose_thresholds(
            "custom_field",
            [],
            priors=[prior],
        )

        assert len(thresholds) > 0
        assert trace is not None
        assert len(trace.priors_used) > 0

    def test_max_thresholds_cap(self):
        """Ensure thresholds are capped at max."""
        # Create many observed values
        observed = list(range(1, 101))

        config = ThresholdProposerConfig(max_thresholds=12)
        thresholds, trace = propose_thresholds("some_field", observed, config=config)

        assert len(thresholds) <= 12
        assert trace.candidates_after_cap <= 12

    def test_threshold_deduplication(self):
        """Ensure duplicate values are deduplicated."""
        observed = [100, 100, 100, 200, 200]
        thresholds, trace = propose_thresholds("some_field", observed)

        # Check no duplicate values
        assert len(thresholds) == len(set(thresholds))
        assert trace.candidates_after_dedup <= trace.candidates_before_dedup

    def test_thresholds_sorted(self):
        """Ensure thresholds are sorted."""
        observed = [500, 100, 300, 200, 400]
        thresholds, _ = propose_thresholds("some_field", observed)

        assert thresholds == sorted(thresholds)

    def test_trace_contains_final_thresholds(self):
        observed = [100, 200, 300]
        thresholds, trace = propose_thresholds("some_field", observed)

        assert trace.final_thresholds == thresholds

    def test_disable_domain_heuristics(self):
        config = ThresholdProposerConfig(use_domain_heuristics=False)
        thresholds, trace = propose_thresholds(
            "traction.arr",
            [],
            config=config,
        )

        # Without observed data or heuristics, should be empty
        assert len(thresholds) == 0
        assert len(trace.heuristics_applied) == 0

    def test_disable_trace(self):
        config = ThresholdProposerConfig(enable_trace=False)
        thresholds, trace = propose_thresholds(
            "some_field",
            [100, 200, 300],
            config=config,
        )

        assert trace is None
        assert len(thresholds) > 0


class TestStableCandidateSets:
    """Tests to ensure candidate sets are stable and deterministic."""

    def test_same_input_same_output(self):
        """Same input should produce same output."""
        observed = [100, 200, 300, 400, 500]

        t1, _ = propose_thresholds("field", observed)
        t2, _ = propose_thresholds("field", observed)

        assert t1 == t2

    def test_order_independent(self):
        """Order of observed values shouldn't matter."""
        observed1 = [100, 200, 300, 400, 500]
        observed2 = [500, 300, 100, 400, 200]

        t1, _ = propose_thresholds("field", observed1)
        t2, _ = propose_thresholds("field", observed2)

        assert t1 == t2

    def test_consistent_across_calls(self):
        """Multiple calls with same data produce same results."""
        observed = [1000000, 2000000, 5000000]

        results = []
        for _ in range(5):
            thresholds, _ = propose_thresholds("traction.arr", observed)
            results.append(tuple(thresholds))

        assert len(set(results)) == 1  # All identical


class TestNoCandidateExplosion:
    """Tests to ensure candidate count stays bounded."""

    def test_cap_at_12_default(self):
        """Default cap is 12 thresholds."""
        # Many observed values plus domain heuristics
        observed = list(range(100, 10000, 100))  # 99 values

        thresholds, trace = propose_thresholds("traction.arr", observed)

        assert len(thresholds) <= 12

    def test_custom_cap(self):
        """Custom cap is respected."""
        observed = list(range(100, 10000, 100))

        for max_t in [5, 8, 15]:
            config = ThresholdProposerConfig(max_thresholds=max_t)
            thresholds, trace = propose_thresholds("field", observed, config=config)

            assert len(thresholds) <= max_t

    def test_many_priors_still_capped(self):
        """Many priors are still capped."""
        priors = [
            ThresholdPrior(
                field="field",
                suggested_thresholds=list(range(1, 51)),  # 50 thresholds
            )
        ]

        config = ThresholdProposerConfig(max_thresholds=10)
        thresholds, _ = propose_thresholds("field", [], priors=priors, config=config)

        assert len(thresholds) <= 10


class TestProposeThresholdsForTemplate:
    """Tests for template-based threshold proposal."""

    def test_known_template(self):
        observed = [500000, 1000000, 2000000]
        thresholds, trace = propose_thresholds_for_template(
            "arr_min",
            observed,
        )

        assert len(thresholds) > 0
        assert trace is not None

    def test_unknown_template(self):
        thresholds, trace = propose_thresholds_for_template(
            "unknown_template_xyz",
            [100, 200],
        )

        assert thresholds == []
        assert trace is None


class TestProposeAllThresholds:
    """Tests for batch threshold proposal."""

    def test_multiple_fields(self):
        observations = [
            FieldObservations("traction.arr", [1000000, 2000000, 5000000]),
            FieldObservations("business_model.gross_margin", [0.5, 0.6, 0.7]),
            FieldObservations("traction.growth_rate", [0.5, 1.0, 1.5]),
        ]

        results = propose_all_thresholds(observations)

        assert len(results) == 3
        for field, (thresholds, trace) in results.items():
            assert len(thresholds) > 0
            assert trace is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_thresholds_only(self):
        thresholds = get_thresholds_only("traction.arr", [1000000])

        assert isinstance(thresholds, list)
        assert len(thresholds) > 0

    def test_default_thresholds_for_field(self):
        # Known field with domain heuristics
        thresholds = default_thresholds_for_field("gross_margin")
        assert len(thresholds) > 0

        # Unknown field
        thresholds = default_thresholds_for_field("unknown_xyz")
        assert len(thresholds) == 0


class TestMergeThresholdTraces:
    """Tests for trace merging."""

    def test_merge_traces(self):
        t1 = ThresholdTrace(
            field="field1",
            observed_count=5,
            observed_min=100,
            observed_max=500,
            observed_mean=300,
            final_thresholds=[100, 200, 300],
            candidates_after_cap=3,
        )
        t2 = ThresholdTrace(
            field="field2",
            observed_count=3,
            observed_min=10,
            observed_max=30,
            observed_mean=20,
            final_thresholds=[10, 20],
            candidates_after_cap=2,
        )

        merged = merge_threshold_traces([t1, t2])

        assert merged["field_count"] == 2
        assert merged["total_thresholds"] == 5
        assert "field1" in merged["fields"]
        assert "field2" in merged["fields"]


class TestDomainHeuristics:
    """Tests for domain heuristics dictionary."""

    def test_gross_margin_heuristics(self):
        assert "gross_margin" in DOMAIN_HEURISTICS
        values = [t[0] for t in DOMAIN_HEURISTICS["gross_margin"]]
        assert 0.40 in values
        assert 0.60 in values

    def test_arr_heuristics(self):
        assert "arr" in DOMAIN_HEURISTICS
        values = [t[0] for t in DOMAIN_HEURISTICS["arr"]]
        assert 1_000_000 in values

    def test_runway_heuristics(self):
        assert "runway_months" in DOMAIN_HEURISTICS
        values = [t[0] for t in DOMAIN_HEURISTICS["runway_months"]]
        assert 12 in values
        assert 18 in values
        assert 24 in values

    def test_ltv_cac_heuristics(self):
        assert "ltv_cac" in DOMAIN_HEURISTICS
        values = [t[0] for t in DOMAIN_HEURISTICS["ltv_cac"]]
        assert 3.0 in values


class TestThresholdTraceToDict:
    """Tests for trace serialization."""

    def test_trace_to_dict(self):
        trace = ThresholdTrace(
            field="test_field",
            observed_count=5,
            observed_min=100.0,
            observed_max=500.0,
            observed_mean=300.0,
            priors_used=["prior1"],
            quantiles_used=[0.5],
            heuristics_applied=["heuristic1"],
            candidates_before_dedup=10,
            candidates_after_dedup=8,
            candidates_after_cap=5,
            final_thresholds=[100.0, 200.0, 300.0],
        )

        d = trace.to_dict()

        assert d["field"] == "test_field"
        assert d["observed"]["count"] == 5
        assert d["observed"]["min"] == 100.0
        assert d["observed"]["max"] == 500.0
        assert d["observed"]["mean"] == 300.0
        assert d["priors_used"] == ["prior1"]
        assert d["quantiles_used"] == [0.5]
        assert d["heuristics_applied"] == ["heuristic1"]
        assert d["candidate_counts"]["before_dedup"] == 10
        assert d["candidate_counts"]["after_dedup"] == 8
        assert d["candidate_counts"]["after_cap"] == 5
        assert d["final_thresholds"] == [100.0, 200.0, 300.0]
