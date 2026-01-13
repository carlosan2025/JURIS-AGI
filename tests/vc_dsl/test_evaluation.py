"""
Unit tests for JURIS VC DSL evaluation engine.

Tests:
- Rule evaluation
- Missing field behavior
- Low confidence behavior
- Rule engine with multiple rules
"""

import pytest

from juris_agi.vc_dsl import (
    # Predicates
    Has,
    Ge,
    Le,
    And,
    ConfGe,
    # Evaluation
    Rule,
    RuleEngine,
    Decision,
    EvaluationTrace,
    build_context_from_dict,
    build_context_from_claims,
    create_threshold_rule,
    # Context
    EvalContext,
    FieldValue,
    EvalResult,
)
from juris_agi.evidence_client import Claim, ClaimPolarity


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strong_deal_context():
    """Context for a strong deal."""
    return build_context_from_dict({
        "traction.arr": 2000000,
        "traction.growth_rate": 0.20,
        "business_model.gross_margin": 0.80,
        "team_quality.founder_background": "Ex-Google, 10 years",
        "capital_intensity.runway_months": 24,
    })


@pytest.fixture
def weak_deal_context():
    """Context for a weak deal."""
    return build_context_from_dict({
        "traction.arr": 100000,
        "traction.growth_rate": 0.05,
        "business_model.gross_margin": 0.40,
        "capital_intensity.runway_months": 6,
    })


@pytest.fixture
def incomplete_context():
    """Context with missing key fields."""
    return build_context_from_dict({
        "traction.arr": 500000,
        # Missing: growth_rate, gross_margin, runway
    })


@pytest.fixture
def low_confidence_context():
    """Context with low confidence values."""
    return EvalContext(
        fields={
            "traction.arr": FieldValue(value=2000000, confidence=0.4),
            "traction.growth_rate": FieldValue(value=0.20, confidence=0.3),
        }
    )


# =============================================================================
# Rule Evaluation Tests
# =============================================================================


class TestRuleEvaluation:
    """Tests for individual rule evaluation."""

    def test_rule_fires_when_predicate_true(self, strong_deal_context):
        """Rule should fire when predicate evaluates to TRUE."""
        rule = Rule(
            rule_id="r1",
            name="High ARR",
            predicate=Ge("traction.arr", 1000000),
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(strong_deal_context)

        assert outcome.result == EvalResult.TRUE
        assert outcome.decision == Decision.INVEST

    def test_rule_does_not_fire_when_predicate_false(self, weak_deal_context):
        """Rule should not fire when predicate evaluates to FALSE."""
        rule = Rule(
            rule_id="r1",
            name="High ARR",
            predicate=Ge("traction.arr", 1000000),
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(weak_deal_context)

        assert outcome.result == EvalResult.FALSE
        assert outcome.decision is None

    def test_rule_unknown_when_missing_field(self, incomplete_context):
        """Rule should be UNKNOWN when required field is missing."""
        rule = Rule(
            rule_id="r1",
            name="Growth Check",
            predicate=Ge("traction.growth_rate", 0.15),
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(incomplete_context)

        assert outcome.result == EvalResult.UNKNOWN
        assert "traction.growth_rate" in outcome.fields_missing

    def test_rule_tracks_used_fields(self, strong_deal_context):
        """Rule should track which fields were used."""
        rule = Rule(
            rule_id="r1",
            name="ARR and Margin",
            predicate=And([
                Ge("traction.arr", 500000),
                Ge("business_model.gross_margin", 0.6),
            ]),
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(strong_deal_context)

        assert "traction.arr" in outcome.fields_used
        assert "business_model.gross_margin" in outcome.fields_used


class TestConfidenceGating:
    """Tests for confidence gate behavior."""

    def test_rule_with_confidence_gate(self, low_confidence_context):
        """Rule with confidence gate should be UNKNOWN for low confidence."""
        rule = Rule(
            rule_id="r1",
            name="High ARR (gated)",
            predicate=And([
                ConfGe("traction.arr", 0.7),
                Ge("traction.arr", 1000000),
            ]),
            decision=Decision.INVEST,
            min_confidence=0.7,
        )

        outcome = rule.evaluate(low_confidence_context)

        # Should be UNKNOWN because confidence (0.4) < min (0.7)
        assert outcome.result == EvalResult.UNKNOWN
        assert "traction.arr" in outcome.fields_low_confidence

    def test_rule_fires_with_sufficient_confidence(self, strong_deal_context):
        """Rule should fire when confidence is sufficient."""
        rule = create_threshold_rule(
            rule_id="r1",
            name="High ARR",
            field="traction.arr",
            operator="ge",
            threshold=1000000,
            decision=Decision.INVEST,
            min_confidence=0.7,
        )

        outcome = rule.evaluate(strong_deal_context)

        # Default confidence from build_context_from_dict is 1.0
        assert outcome.result == EvalResult.TRUE


# =============================================================================
# Rule Engine Tests
# =============================================================================


class TestRuleEngine:
    """Tests for rule engine."""

    def test_first_matching_rule_wins(self, strong_deal_context):
        """Engine should use first matching rule's decision."""
        rules = [
            Rule(
                rule_id="r1",
                name="Unicorn",
                predicate=Ge("traction.arr", 10000000),
                decision=Decision.INVEST,
                priority=10,
            ),
            Rule(
                rule_id="r2",
                name="Strong ARR",
                predicate=Ge("traction.arr", 1000000),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]

        engine = RuleEngine(rules)
        trace = engine.evaluate(strong_deal_context)

        # r2 should match (ARR = 2M)
        assert trace.final_decision == Decision.INVEST
        assert any(o.rule_id == "r2" and o.result == EvalResult.TRUE for o in trace.rule_outcomes)

    def test_pass_has_veto_power(self, weak_deal_context):
        """PASS rules should override INVEST rules."""
        rules = [
            Rule(
                rule_id="r1",
                name="Has ARR",
                predicate=Has("traction.arr"),
                decision=Decision.INVEST,
                priority=5,
            ),
            Rule(
                rule_id="r2",
                name="Low Growth",
                predicate=Le("traction.growth_rate", 0.10),
                decision=Decision.PASS,
                priority=7,
            ),
        ]

        engine = RuleEngine(rules)
        trace = engine.evaluate(weak_deal_context)

        # PASS should win due to veto power
        assert trace.final_decision == Decision.PASS

    def test_default_decision_when_no_rules_match(self, incomplete_context):
        """Engine should return default decision when no rules match."""
        rules = [
            Rule(
                rule_id="r1",
                name="Strong Growth",
                predicate=Ge("traction.growth_rate", 0.20),
                decision=Decision.INVEST,
            ),
        ]

        engine = RuleEngine(rules, default_decision=Decision.DEFER)
        trace = engine.evaluate(incomplete_context)

        # Missing field -> UNKNOWN, no match -> default
        assert trace.final_decision == Decision.DEFER

    def test_trace_includes_all_outcomes(self, strong_deal_context):
        """Trace should include outcomes for all rules."""
        rules = [
            Rule(
                rule_id="r1",
                name="Rule 1",
                predicate=Has("traction.arr"),
                decision=Decision.INVEST,
            ),
            Rule(
                rule_id="r2",
                name="Rule 2",
                predicate=Has("missing.field"),
                decision=Decision.PASS,
            ),
        ]

        engine = RuleEngine(rules)
        trace = engine.evaluate(strong_deal_context)

        assert len(trace.rule_outcomes) == 2
        assert trace.rules_fired[0].rule_id == "r1"


class TestUnknownHandling:
    """Tests for UNKNOWN handling modes."""

    def test_defer_mode_unknown_causes_defer(self, incomplete_context):
        """In defer mode, high-priority UNKNOWN should cause DEFER."""
        rules = [
            Rule(
                rule_id="r1",
                name="Growth Check",
                predicate=Ge("traction.growth_rate", 0.15),
                decision=Decision.INVEST,
                priority=7,
            ),
        ]

        engine = RuleEngine(rules, unknown_handling="defer")
        trace = engine.evaluate(incomplete_context)

        assert trace.final_decision == Decision.DEFER

    def test_continue_mode_skips_unknown(self, incomplete_context):
        """In continue mode, UNKNOWN rules should be skipped."""
        rules = [
            Rule(
                rule_id="r1",
                name="Growth Check",
                predicate=Ge("traction.growth_rate", 0.15),
                decision=Decision.INVEST,
                priority=7,
            ),
            Rule(
                rule_id="r2",
                name="ARR Check",
                predicate=Ge("traction.arr", 400000),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]

        engine = RuleEngine(rules, unknown_handling="continue")
        trace = engine.evaluate(incomplete_context)

        # r1 is UNKNOWN, r2 matches
        assert trace.final_decision == Decision.INVEST


# =============================================================================
# Context Building Tests
# =============================================================================


class TestContextBuilding:
    """Tests for context building utilities."""

    def test_build_from_dict(self):
        """Should build context from simple dict."""
        ctx = build_context_from_dict({
            "field1": 100,
            "field2": "value",
        })

        assert ctx.has_field("field1")
        assert ctx.get_field("field1").value == 100
        assert ctx.get_field("field1").confidence == 1.0

    def test_build_from_claims(self):
        """Should build context from Claim objects."""
        claims = [
            Claim(
                claim_id="c1",
                claim_type="traction",
                field="arr",
                value=1000000,
                confidence=0.9,
                polarity=ClaimPolarity.SUPPORTIVE,
            ),
            Claim(
                claim_id="c2",
                claim_type="team_quality",
                field="size",
                value=12,
                confidence=0.85,
                polarity=ClaimPolarity.NEUTRAL,
            ),
        ]

        ctx = build_context_from_claims(claims)

        assert ctx.has_field("traction.arr")
        assert ctx.get_field("traction.arr").value == 1000000
        assert ctx.get_field("traction.arr").confidence == 0.9

    def test_build_from_claims_keeps_highest_confidence(self):
        """Should keep highest confidence when multiple claims for same field."""
        claims = [
            Claim(
                claim_id="c1",
                claim_type="traction",
                field="arr",
                value=1000000,
                confidence=0.6,
                polarity=ClaimPolarity.SUPPORTIVE,
            ),
            Claim(
                claim_id="c2",
                claim_type="traction",
                field="arr",
                value=1200000,
                confidence=0.9,
                polarity=ClaimPolarity.SUPPORTIVE,
            ),
        ]

        ctx = build_context_from_claims(claims)

        # Should keep the higher confidence claim
        assert ctx.get_field("traction.arr").value == 1200000
        assert ctx.get_field("traction.arr").confidence == 0.9


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for rule creation helpers."""

    def test_create_threshold_rule_ge(self, strong_deal_context):
        """Should create working threshold rule with ge operator."""
        rule = create_threshold_rule(
            rule_id="test",
            name="ARR Threshold",
            field="traction.arr",
            operator="ge",
            threshold=1000000,
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(strong_deal_context)
        assert outcome.result == EvalResult.TRUE

    def test_create_threshold_rule_le(self, weak_deal_context):
        """Should create working threshold rule with le operator."""
        rule = create_threshold_rule(
            rule_id="test",
            name="Low Burn",
            field="capital_intensity.runway_months",
            operator="le",
            threshold=12,
            decision=Decision.PASS,
        )

        outcome = rule.evaluate(weak_deal_context)
        assert outcome.result == EvalResult.TRUE

    def test_create_threshold_rule_between(self, strong_deal_context):
        """Should create working threshold rule with between operator."""
        rule = create_threshold_rule(
            rule_id="test",
            name="Runway Range",
            field="capital_intensity.runway_months",
            operator="between",
            threshold=12,
            threshold_hi=36,
            decision=Decision.INVEST,
        )

        outcome = rule.evaluate(strong_deal_context)
        assert outcome.result == EvalResult.TRUE
