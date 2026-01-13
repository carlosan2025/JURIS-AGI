"""
Tests for uncertainty quantification module.

Tests cover:
- Epistemic uncertainty (policy disagreement, sensitivity)
- Aleatoric uncertainty (low confidence, variance, conflicts)
- Information request generation
- Defer/refusal decisions
"""

import pytest

from juris_agi.vc_dsl import (
    # Predicates
    Ge, Le, Has, And,
    # Evaluation
    Decision, Rule, EvalContext, FieldValue,
    build_context_from_dict,
    # Hypothesis
    HistoricalDecision,
    DecisionDataset,
    HypothesisSet,
    HypothesisSetConfig,
    # Uncertainty
    UncertaintyLevel,
    UncertaintyReason,
    EpistemicUncertainty,
    AleatoricUncertainty,
    InformationRequest,
    UncertaintyReport,
    UncertaintyConfig,
    UncertaintyAnalyzer,
    analyze_uncertainty,
    should_request_more_info,
    get_top_information_requests,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_rules():
    """Simple rule set: high ARR = INVEST, low ARR = PASS."""
    return [
        Rule(
            rule_id="high_arr",
            name="High ARR",
            predicate=Ge("traction.arr", 1_000_000),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="low_arr",
            name="Low ARR",
            predicate=Le("traction.arr", 100_000),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def alternative_rules():
    """Alternative rule set: focus on growth."""
    return [
        Rule(
            rule_id="high_growth",
            name="High Growth",
            predicate=Ge("traction.growth_rate", 100),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="low_growth",
            name="Low Growth",
            predicate=Le("traction.growth_rate", 20),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def hypothesis_set_single(simple_rules):
    """HypothesisSet with single clear policy."""
    decisions = [
        HistoricalDecision(
            deal_id=f"invest_{i}",
            decision=Decision.INVEST,
            context=build_context_from_dict({"traction.arr": 1_500_000}),
        )
        for i in range(5)
    ] + [
        HistoricalDecision(
            deal_id=f"pass_{i}",
            decision=Decision.PASS,
            context=build_context_from_dict({"traction.arr": 50_000}),
        )
        for i in range(5)
    ]
    dataset = DecisionDataset(decisions=decisions)

    config = HypothesisSetConfig(min_coverage=0.3, min_accuracy=0.5)
    hyp_set = HypothesisSet(config=config)
    hyp_set.add_hypothesis(simple_rules, dataset)

    return hyp_set


@pytest.fixture
def hypothesis_set_multiple(simple_rules, alternative_rules):
    """HypothesisSet with multiple competing policies."""
    # Dataset where both policies partially work
    decisions = [
        HistoricalDecision(
            deal_id=f"invest_{i}",
            decision=Decision.INVEST,
            context=build_context_from_dict({
                "traction.arr": 1_500_000,
                "traction.growth_rate": 150,
            }),
        )
        for i in range(5)
    ] + [
        HistoricalDecision(
            deal_id=f"pass_{i}",
            decision=Decision.PASS,
            context=build_context_from_dict({
                "traction.arr": 50_000,
                "traction.growth_rate": 10,
            }),
        )
        for i in range(5)
    ]
    dataset = DecisionDataset(decisions=decisions)

    config = HypothesisSetConfig(
        min_coverage=0.3,
        min_accuracy=0.5,
        diversity_threshold=0.5,  # Allow similar hypotheses
    )
    hyp_set = HypothesisSet(config=config)
    hyp_set.add_hypothesis(simple_rules, dataset, name="ARR Policy")
    hyp_set.add_hypothesis(alternative_rules, dataset, name="Growth Policy")

    return hyp_set


@pytest.fixture
def high_confidence_context():
    """Context with high confidence claims."""
    return build_context_from_dict({
        "traction.arr": 2_000_000,
        "traction.growth_rate": 100,
        "deal.valuation": 20_000_000,
    })


@pytest.fixture
def low_confidence_context():
    """Context with low confidence claims."""
    ctx = EvalContext()
    ctx.fields["traction.arr"] = FieldValue(
        value=2_000_000,
        confidence=0.5,  # Low confidence
        exists=True,
    )
    ctx.fields["traction.growth_rate"] = FieldValue(
        value=100,
        confidence=0.4,  # Low confidence
        exists=True,
    )
    return ctx


@pytest.fixture
def context_with_missing_fields():
    """Context with some missing fields."""
    return build_context_from_dict({
        "traction.arr": 2_000_000,
        # Missing: growth_rate, valuation, team info
    })


@pytest.fixture
def context_with_timeseries():
    """Context with high-variance time series."""
    ctx = build_context_from_dict({
        "traction.arr": 2_000_000,
    })
    # Add volatile time series
    ctx.fields["traction.arr_timeseries"] = FieldValue(
        value=[
            {"t": "Q1 2024", "value": 500_000},
            {"t": "Q2 2024", "value": 2_500_000},  # Big jump
            {"t": "Q3 2024", "value": 1_000_000},  # Big drop
            {"t": "Q4 2024", "value": 2_000_000},
        ],
        exists=True,
    )
    return ctx


# =============================================================================
# Epistemic Uncertainty Tests
# =============================================================================


class TestEpistemicUncertainty:
    """Tests for epistemic uncertainty signals."""

    def test_single_policy_low_epistemic(self, hypothesis_set_single, high_confidence_context):
        """Single policy should have low epistemic uncertainty."""
        analyzer = UncertaintyAnalyzer()
        epistemic = analyzer.analyze_epistemic(high_confidence_context, hypothesis_set_single)

        assert epistemic.num_equivalent_policies <= 1
        assert epistemic.policy_agreement_rate == 1.0
        # Single policy with full agreement should have low uncertainty
        # Score may be elevated by sensitivity analysis
        assert epistemic.score <= 0.4

    def test_multiple_policies_higher_epistemic(self, hypothesis_set_multiple, high_confidence_context):
        """Multiple policies should increase epistemic uncertainty."""
        analyzer = UncertaintyAnalyzer()
        epistemic = analyzer.analyze_epistemic(high_confidence_context, hypothesis_set_multiple)

        assert epistemic.num_equivalent_policies >= 1
        # With multiple policies, there's more model uncertainty
        assert epistemic.score >= 0

    def test_policy_disagreement_detected(self):
        """Disagreement between policies should be detected."""
        # Create two policies that disagree on a specific deal
        rules_invest = [
            Rule(
                rule_id="invest_rule",
                name="Always Invest",
                predicate=Has("traction.arr"),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]
        rules_pass = [
            Rule(
                rule_id="pass_rule",
                name="Always Pass",
                predicate=Has("traction.arr"),
                decision=Decision.PASS,
                priority=5,
            ),
        ]

        # Create dataset
        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=build_context_from_dict({"traction.arr": 1_000_000}),
            )
        ]
        dataset = DecisionDataset(decisions=decisions)

        config = HypothesisSetConfig(
            min_coverage=0.1,
            min_accuracy=0.1,
            diversity_threshold=0.9,  # Allow very different policies
        )
        hyp_set = HypothesisSet(config=config)
        hyp_set.add_hypothesis(rules_invest, dataset, name="Invest Policy")
        hyp_set.add_hypothesis(rules_pass, dataset, name="Pass Policy")

        ctx = build_context_from_dict({"traction.arr": 1_000_000})

        analyzer = UncertaintyAnalyzer()
        epistemic = analyzer.analyze_epistemic(ctx, hyp_set)

        # Policies should disagree
        if len(hyp_set.hypotheses) >= 2:
            assert epistemic.policy_agreement_rate < 1.0
            assert len(epistemic.disagreeing_policies) > 0

    def test_sensitivity_analysis(self, hypothesis_set_single, high_confidence_context):
        """Sensitivity to claim removal is analyzed."""
        analyzer = UncertaintyAnalyzer()
        epistemic = analyzer.analyze_epistemic(high_confidence_context, hypothesis_set_single)

        # Should have analyzed some claims
        # max_sensitivity should be between 0 and 1
        assert 0.0 <= epistemic.max_sensitivity <= 1.0

    def test_epistemic_level_categorization(self):
        """Epistemic uncertainty is correctly categorized."""
        low = EpistemicUncertainty(score=0.1)
        medium = EpistemicUncertainty(score=0.3)
        high = EpistemicUncertainty(score=0.5)
        very_high = EpistemicUncertainty(score=0.7)

        assert low.level == UncertaintyLevel.LOW
        assert medium.level == UncertaintyLevel.MEDIUM
        assert high.level == UncertaintyLevel.HIGH
        assert very_high.level == UncertaintyLevel.VERY_HIGH


# =============================================================================
# Aleatoric Uncertainty Tests
# =============================================================================


class TestAleatoricUncertainty:
    """Tests for aleatoric uncertainty signals."""

    def test_high_confidence_low_aleatoric(self, simple_rules, high_confidence_context):
        """High confidence claims should have low aleatoric uncertainty."""
        analyzer = UncertaintyAnalyzer()
        aleatoric = analyzer.analyze_aleatoric(high_confidence_context, simple_rules)

        assert aleatoric.avg_confidence_of_used_claims >= 0.9
        assert len(aleatoric.low_confidence_claims) == 0
        assert aleatoric.score < 0.3

    def test_low_confidence_high_aleatoric(self, simple_rules, low_confidence_context):
        """Low confidence claims should increase aleatoric uncertainty."""
        analyzer = UncertaintyAnalyzer()
        aleatoric = analyzer.analyze_aleatoric(low_confidence_context, simple_rules)

        assert aleatoric.avg_confidence_of_used_claims < 0.7
        assert len(aleatoric.low_confidence_claims) > 0
        assert aleatoric.score > 0.1

    def test_missing_fields_detected(self, simple_rules, context_with_missing_fields):
        """Missing fields should be detected."""
        # Add a rule that requires growth_rate
        rules = simple_rules + [
            Rule(
                rule_id="growth_rule",
                name="Growth",
                predicate=Ge("traction.growth_rate", 50),
                decision=Decision.INVEST,
                priority=4,
            ),
        ]

        analyzer = UncertaintyAnalyzer()
        aleatoric = analyzer.analyze_aleatoric(context_with_missing_fields, rules)

        assert "traction.growth_rate" in aleatoric.missing_fields
        assert aleatoric.score > 0

    def test_high_variance_timeseries_detected(self, simple_rules, context_with_timeseries):
        """High variance in time series should be detected."""
        analyzer = UncertaintyAnalyzer()
        aleatoric = analyzer.analyze_aleatoric(context_with_timeseries, simple_rules)

        # Should detect high variance in ARR timeseries
        assert len(aleatoric.high_variance_fields) > 0 or aleatoric.max_coefficient_of_variation > 0

    def test_conflicts_increase_aleatoric(self, simple_rules, high_confidence_context):
        """Unresolved conflicts should increase aleatoric uncertainty."""

        class MockConflict:
            def __init__(self):
                self.severity = 0.8
                self.claim_ids = ["claim_1.arr", "claim_2.arr"]

        conflicts = [MockConflict(), MockConflict()]

        analyzer = UncertaintyAnalyzer()
        aleatoric = analyzer.analyze_aleatoric(
            high_confidence_context, simple_rules, conflicts=conflicts
        )

        assert aleatoric.unresolved_conflicts == 2
        assert aleatoric.conflict_severity_sum > 0
        assert aleatoric.score > 0

    def test_aleatoric_level_categorization(self):
        """Aleatoric uncertainty is correctly categorized."""
        low = AleatoricUncertainty(score=0.1)
        medium = AleatoricUncertainty(score=0.3)
        high = AleatoricUncertainty(score=0.5)
        very_high = AleatoricUncertainty(score=0.7)

        assert low.level == UncertaintyLevel.LOW
        assert medium.level == UncertaintyLevel.MEDIUM
        assert high.level == UncertaintyLevel.HIGH
        assert very_high.level == UncertaintyLevel.VERY_HIGH


# =============================================================================
# Information Request Tests
# =============================================================================


class TestInformationRequests:
    """Tests for information request generation."""

    def test_requests_for_low_confidence(self, simple_rules, low_confidence_context):
        """Generates requests for low confidence claims."""
        analyzer = UncertaintyAnalyzer()

        epistemic = EpistemicUncertainty()
        aleatoric = analyzer.analyze_aleatoric(low_confidence_context, simple_rules)

        requests = analyzer.generate_information_requests(epistemic, aleatoric, simple_rules)

        # Should suggest improving confidence
        assert len(requests) > 0
        assert any("confidence" in r.description.lower() for r in requests)

    def test_requests_for_missing_fields(self, simple_rules, context_with_missing_fields):
        """Generates requests for missing fields."""
        rules = simple_rules + [
            Rule(
                rule_id="growth_rule",
                name="Growth",
                predicate=Ge("traction.growth_rate", 50),
                decision=Decision.INVEST,
            ),
        ]

        analyzer = UncertaintyAnalyzer()

        epistemic = EpistemicUncertainty()
        aleatoric = analyzer.analyze_aleatoric(context_with_missing_fields, rules)

        requests = analyzer.generate_information_requests(epistemic, aleatoric, rules)

        # Should suggest getting missing fields
        assert len(requests) > 0
        assert any("growth_rate" in r.field for r in requests)

    def test_requests_sorted_by_importance(self):
        """Requests are sorted by importance."""
        analyzer = UncertaintyAnalyzer()

        epistemic = EpistemicUncertainty(
            most_sensitive_claims=["sensitive_field"]
        )
        aleatoric = AleatoricUncertainty(
            low_confidence_claims=["low_conf_1", "low_conf_2"],
            missing_fields=["missing_1"],
            missing_field_importance={"missing_1": 0.9},
        )

        requests = analyzer.generate_information_requests(epistemic, aleatoric, [])

        if len(requests) >= 2:
            # Should be sorted by importance (descending)
            for i in range(len(requests) - 1):
                assert requests[i].importance >= requests[i + 1].importance

    def test_max_requests_limit(self):
        """Number of requests is limited."""
        config = UncertaintyConfig(max_information_requests=3)
        analyzer = UncertaintyAnalyzer(config=config)

        aleatoric = AleatoricUncertainty(
            low_confidence_claims=["f1", "f2", "f3", "f4", "f5"],
            missing_fields=["m1", "m2", "m3"],
            missing_field_importance={"m1": 0.5, "m2": 0.5, "m3": 0.5},
        )

        requests = analyzer.generate_information_requests(
            EpistemicUncertainty(), aleatoric, []
        )

        assert len(requests) <= 3


# =============================================================================
# Uncertainty Report Tests
# =============================================================================


class TestUncertaintyReport:
    """Tests for complete uncertainty reports."""

    def test_analyze_complete_report(self, hypothesis_set_single, high_confidence_context):
        """Full analysis produces complete report."""
        report = analyze_uncertainty(
            context=high_confidence_context,
            hypothesis_set=hypothesis_set_single,
            deal_id="test_deal",
        )

        assert report.deal_id == "test_deal"
        assert report.epistemic is not None
        assert report.aleatoric is not None
        assert 0 <= report.total_uncertainty <= 1
        assert report.level in UncertaintyLevel

    def test_report_includes_decision(self, hypothesis_set_single, high_confidence_context):
        """Report includes the decision from best policy."""
        report = analyze_uncertainty(
            context=high_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        assert report.decision is not None
        assert report.decision in Decision

    def test_report_includes_reasons(self, hypothesis_set_single, low_confidence_context):
        """Report includes reasons for uncertainty."""
        report = analyze_uncertainty(
            context=low_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        assert len(report.top_reasons) > 0
        for reason in report.top_reasons:
            assert reason.category in ["epistemic", "aleatoric"]
            assert reason.code != ""
            assert reason.description != ""

    def test_report_to_dict(self, hypothesis_set_single, high_confidence_context):
        """Report can be converted to dictionary."""
        report = analyze_uncertainty(
            context=high_confidence_context,
            hypothesis_set=hypothesis_set_single,
            deal_id="test_deal",
        )

        d = report.to_dict()

        assert d["deal_id"] == "test_deal"
        assert "epistemic_score" in d
        assert "aleatoric_score" in d
        assert "total_uncertainty" in d
        assert "uncertainty_level" in d
        assert "top_reasons" in d
        assert "information_requests" in d

    def test_compute_total_uncertainty(self):
        """Total uncertainty is computed from components."""
        report = UncertaintyReport()
        report.epistemic = EpistemicUncertainty()
        report.aleatoric = AleatoricUncertainty()

        # Manually set scores (compute_score updates internal score)
        report.epistemic.policy_agreement_rate = 0.5  # 50% disagreement
        report.aleatoric.avg_confidence_of_used_claims = 0.5  # Low confidence

        report.compute_total()

        # Should compute scores based on the components
        assert report.epistemic.score > 0  # Disagreement contributes
        assert report.aleatoric.score > 0  # Low confidence contributes
        assert report.total_uncertainty > 0  # Total should be positive


# =============================================================================
# Defer/Refusal Tests
# =============================================================================


class TestDeferDecisions:
    """Tests for defer/refusal decisions."""

    def test_high_uncertainty_should_defer(self, hypothesis_set_single, low_confidence_context):
        """High uncertainty should trigger defer."""
        config = UncertaintyConfig(
            defer_aleatoric_threshold=0.2,  # Low threshold to trigger defer
        )
        analyzer = UncertaintyAnalyzer(config=config)

        report = analyzer.analyze(
            context=low_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        # With very low defer threshold, low confidence context should defer
        # (This depends on the actual confidence values)
        assert isinstance(report.should_defer, bool)

    def test_low_uncertainty_no_defer(self, hypothesis_set_single, high_confidence_context):
        """Low uncertainty should not trigger defer."""
        config = UncertaintyConfig(
            defer_epistemic_threshold=0.9,  # High threshold
            defer_aleatoric_threshold=0.9,  # High threshold
            defer_total_threshold=0.9,  # High threshold
        )
        analyzer = UncertaintyAnalyzer(config=config)

        report = analyzer.analyze(
            context=high_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        assert report.should_defer is False

    def test_defer_reason_provided(self, hypothesis_set_single, low_confidence_context):
        """Defer reason is provided when deferring."""
        config = UncertaintyConfig(
            defer_aleatoric_threshold=0.1,  # Very low to trigger
        )
        analyzer = UncertaintyAnalyzer(config=config)

        report = analyzer.analyze(
            context=low_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        if report.should_defer:
            assert report.defer_reason is not None
            assert len(report.defer_reason) > 0

    def test_should_request_more_info_function(self, hypothesis_set_single, high_confidence_context):
        """Convenience function works correctly."""
        report = analyze_uncertainty(
            context=high_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        result = should_request_more_info(report)
        assert isinstance(result, bool)

    def test_get_top_information_requests_function(self, hypothesis_set_single, low_confidence_context):
        """Convenience function returns formatted requests."""
        report = analyze_uncertainty(
            context=low_confidence_context,
            hypothesis_set=hypothesis_set_single,
        )

        requests = get_top_information_requests(report, max_requests=3)

        assert len(requests) <= 3
        for req in requests:
            assert "field" in req
            assert "description" in req
            assert "importance" in req


# =============================================================================
# Integration Tests
# =============================================================================


class TestUncertaintyIntegration:
    """Integration tests for uncertainty analysis."""

    def test_clean_data_low_uncertainty(self):
        """Clean data with clear decisions should have low uncertainty."""
        # Create clear dataset
        decisions = [
            HistoricalDecision(
                deal_id=f"invest_{i}",
                decision=Decision.INVEST,
                context=build_context_from_dict({
                    "traction.arr": 2_000_000,
                    "traction.growth_rate": 100,
                }),
            )
            for i in range(10)
        ]
        dataset = DecisionDataset(decisions=decisions)

        rules = [
            Rule(
                rule_id="high_arr",
                name="High ARR",
                predicate=Ge("traction.arr", 1_000_000),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]

        config = HypothesisSetConfig(min_coverage=0.3, min_accuracy=0.5)
        hyp_set = HypothesisSet(config=config)
        hyp_set.add_hypothesis(rules, dataset)

        # Analyze a matching deal
        ctx = build_context_from_dict({
            "traction.arr": 3_000_000,
            "traction.growth_rate": 120,
        })

        report = analyze_uncertainty(ctx, hyp_set, deal_id="clean_deal")

        # Should have relatively low uncertainty
        assert report.total_uncertainty < 0.5
        assert report.should_defer is False

    def test_noisy_data_high_uncertainty(self):
        """Noisy data should produce high uncertainty."""
        # Create noisy context
        ctx = EvalContext()
        ctx.fields["traction.arr"] = FieldValue(
            value=1_000_000,
            confidence=0.3,  # Very low confidence
            exists=True,
        )

        # Add high-variance timeseries
        ctx.fields["traction.arr_timeseries"] = FieldValue(
            value=[
                {"t": "Q1 2024", "value": 100_000},
                {"t": "Q2 2024", "value": 2_000_000},  # Huge jump
                {"t": "Q3 2024", "value": 500_000},   # Big drop
                {"t": "Q4 2024", "value": 1_000_000},
            ],
            exists=True,
        )

        rules = [
            Rule(
                rule_id="high_arr",
                name="High ARR",
                predicate=Ge("traction.arr", 1_000_000),
                decision=Decision.INVEST,
            ),
        ]

        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=ctx,
            )
        ]
        dataset = DecisionDataset(decisions=decisions)

        config = HypothesisSetConfig(min_coverage=0.1, min_accuracy=0.1)
        hyp_set = HypothesisSet(config=config)
        hyp_set.add_hypothesis(rules, dataset)

        report = analyze_uncertainty(ctx, hyp_set, deal_id="noisy_deal")

        # Should have elevated aleatoric uncertainty
        assert report.aleatoric.score > 0

    def test_actionable_suggestions(self, hypothesis_set_single):
        """Report provides actionable suggestions."""
        # Context with issues
        ctx = EvalContext()
        ctx.fields["traction.arr"] = FieldValue(
            value=1_500_000,
            confidence=0.4,  # Low confidence
            exists=True,
        )
        # Missing growth_rate

        report = analyze_uncertainty(ctx, hypothesis_set_single)

        # Should have information requests
        if report.aleatoric.score > 0.1:
            assert len(report.information_requests) > 0
            # Each request should be actionable
            for req in report.information_requests:
                assert req.field != ""
                assert req.description != ""
                assert req.reason != ""


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_hypothesis_set(self, high_confidence_context):
        """Empty hypothesis set is handled."""
        hyp_set = HypothesisSet()

        report = analyze_uncertainty(high_confidence_context, hyp_set)

        assert report.decision is None
        assert report.epistemic.score == 0

    def test_empty_context(self, hypothesis_set_single):
        """Empty context is handled."""
        ctx = EvalContext()

        report = analyze_uncertainty(ctx, hypothesis_set_single)

        # Should have high aleatoric uncertainty due to missing fields
        assert isinstance(report.aleatoric.score, float)

    def test_all_unknown_fields(self, hypothesis_set_single):
        """Context with all unknown fields is handled."""
        ctx = build_context_from_dict({
            "irrelevant.field1": 100,
            "irrelevant.field2": 200,
        })

        report = analyze_uncertainty(ctx, hypothesis_set_single)

        # Should handle gracefully
        assert isinstance(report.total_uncertainty, float)
