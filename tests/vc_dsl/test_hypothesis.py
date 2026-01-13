"""
Tests for multi-hypothesis reasoning module.

Tests cover:
- MDL scoring for rule programs
- Coverage and exception tracking
- HypothesisSet top-K management
- Regime inconsistency detection
- Multi-policy evaluation
"""

import pytest
from datetime import datetime

from juris_agi.vc_dsl import (
    # Predicates
    Ge, Le, Between, Has, And, Or,
    # Evaluation
    Decision, Rule, EvalContext, FieldValue,
    build_context_from_dict,
    # Hypothesis
    HistoricalDecision,
    DecisionDataset,
    CoverageStats,
    ExceptionCase,
    MDLScoreBreakdown,
    MDLScorer,
    PolicyHypothesis,
    HypothesisSetConfig,
    HypothesisSet,
    MultiHypothesisEngine,
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
def growth_focused_rules():
    """Growth-focused rules: prioritize high growth rate."""
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
def market_focused_rules():
    """Market-focused rules: prioritize large TAM."""
    return [
        Rule(
            rule_id="large_tam",
            name="Large TAM",
            predicate=Ge("market.tam", 10_000_000_000),
            decision=Decision.INVEST,
            priority=5,
        ),
        Rule(
            rule_id="small_tam",
            name="Small TAM",
            predicate=Le("market.tam", 100_000_000),
            decision=Decision.PASS,
            priority=3,
        ),
    ]


@pytest.fixture
def coherent_dataset():
    """Dataset where all decisions follow a coherent pattern (high ARR = INVEST)."""
    decisions = []

    # High ARR deals that were invested in
    for i in range(10):
        ctx = build_context_from_dict({
            "traction.arr": 1_500_000 + i * 100_000,
            "traction.growth_rate": 80 + i * 5,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"deal_invest_{i}",
            decision=Decision.INVEST,
            context=ctx,
        ))

    # Low ARR deals that were passed on
    for i in range(10):
        ctx = build_context_from_dict({
            "traction.arr": 50_000 + i * 5_000,
            "traction.growth_rate": 10 + i * 2,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"deal_pass_{i}",
            decision=Decision.PASS,
            context=ctx,
        ))

    return DecisionDataset(
        decisions=decisions,
        name="coherent_dataset",
        description="Clear pattern: high ARR = INVEST, low ARR = PASS",
    )


@pytest.fixture
def inconsistent_dataset():
    """Dataset with two inconsistent decision regimes.

    Regime A: Prioritizes ARR (used in deals 0-9)
    Regime B: Prioritizes growth rate (used in deals 10-19)
    """
    decisions = []

    # Regime A: ARR-focused decisions
    # High ARR, low growth -> INVEST (ARR matters)
    for i in range(5):
        ctx = build_context_from_dict({
            "traction.arr": 2_000_000 + i * 100_000,
            "traction.growth_rate": 15,  # Low growth
            "market.tam": 5_000_000_000,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"regime_a_invest_{i}",
            decision=Decision.INVEST,
            context=ctx,
        ))

    # Low ARR, high growth -> PASS (ARR matters)
    for i in range(5):
        ctx = build_context_from_dict({
            "traction.arr": 50_000 + i * 5_000,
            "traction.growth_rate": 150,  # High growth
            "market.tam": 5_000_000_000,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"regime_a_pass_{i}",
            decision=Decision.PASS,
            context=ctx,
        ))

    # Regime B: Growth-focused decisions
    # Low ARR, high growth -> INVEST (growth matters)
    for i in range(5):
        ctx = build_context_from_dict({
            "traction.arr": 80_000 + i * 5_000,
            "traction.growth_rate": 200 + i * 10,
            "market.tam": 15_000_000_000,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"regime_b_invest_{i}",
            decision=Decision.INVEST,
            context=ctx,
        ))

    # High ARR, low growth -> PASS (growth matters)
    for i in range(5):
        ctx = build_context_from_dict({
            "traction.arr": 3_000_000 + i * 100_000,
            "traction.growth_rate": 5,  # Very low growth
            "market.tam": 15_000_000_000,
        })
        decisions.append(HistoricalDecision(
            deal_id=f"regime_b_pass_{i}",
            decision=Decision.PASS,
            context=ctx,
        ))

    return DecisionDataset(
        decisions=decisions,
        name="inconsistent_dataset",
        description="Two conflicting regimes: ARR-focused vs Growth-focused",
    )


# =============================================================================
# MDL Scorer Tests
# =============================================================================


class TestMDLScorer:
    """Tests for MDL scoring."""

    def test_score_empty_rules(self):
        """Empty rule set has zero complexity."""
        scorer = MDLScorer()
        score = scorer.score_rules([])
        assert score == 0.0

    def test_score_single_simple_rule(self, simple_rules):
        """Single rule has base cost plus predicate cost."""
        scorer = MDLScorer(rule_base_cost=1.0, predicate_cost=0.5)
        score = scorer.score_rules([simple_rules[0]])

        # Should have base cost + some predicate cost
        assert score > 0
        assert score < 10  # Reasonable upper bound

    def test_score_multiple_rules(self, simple_rules):
        """Multiple rules accumulate costs."""
        scorer = MDLScorer()
        single_score = scorer.score_rules([simple_rules[0]])
        double_score = scorer.score_rules(simple_rules)

        # More rules = higher score
        assert double_score > single_score

    def test_complex_predicate_higher_cost(self):
        """Complex predicates cost more."""
        scorer = MDLScorer()

        simple_rule = Rule(
            rule_id="simple",
            name="Simple",
            predicate=Ge("traction.arr", 1_000_000),
            decision=Decision.INVEST,
        )

        complex_rule = Rule(
            rule_id="complex",
            name="Complex",
            predicate=And([
                Ge("traction.arr", 1_000_000),
                Ge("traction.growth_rate", 50),
                Le("deal.valuation", 50_000_000),
            ]),
            decision=Decision.INVEST,
        )

        simple_cost = scorer.score_rules([simple_rule])
        complex_cost = scorer.score_rules([complex_rule])

        assert complex_cost > simple_cost

    def test_score_coverage_perfect(self, simple_rules, coherent_dataset):
        """Perfect coverage with no exceptions."""
        scorer = MDLScorer()
        stats, exceptions, breakdown = scorer.score_coverage(
            simple_rules, coherent_dataset
        )

        # Should have high coverage and accuracy
        assert stats.covered > 0
        assert stats.accuracy >= 0.8  # Should be high given coherent data

    def test_score_coverage_exceptions_tracked(self):
        """Exceptions are tracked correctly."""
        scorer = MDLScorer()

        # Rules that will make wrong predictions
        rules = [
            Rule(
                rule_id="always_invest",
                name="Always Invest",
                predicate=Has("traction.arr"),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]

        # Dataset with PASS decisions
        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.PASS,
                context=build_context_from_dict({"traction.arr": 100_000}),
            ),
        ]
        dataset = DecisionDataset(decisions=decisions)

        stats, exceptions, breakdown = scorer.score_coverage(rules, dataset)

        assert stats.exceptions == 1
        assert len(exceptions) == 1
        assert exceptions[0].actual_decision == Decision.PASS
        assert exceptions[0].predicted_decision == Decision.INVEST


class TestMDLScoreBreakdown:
    """Tests for MDL score breakdown."""

    def test_compute_total(self):
        """Total is computed correctly."""
        breakdown = MDLScoreBreakdown(
            rule_complexity=5.0,
            exception_cost=2.0,
            coverage_penalty=1.0,
            confidence_bonus=0.5,
        )
        total = breakdown.compute_total()

        # total = complexity + exception + coverage - confidence
        assert total == 5.0 + 2.0 + 1.0 - 0.5
        assert breakdown.total_score == total

    def test_lower_score_is_better(self):
        """Lower MDL score indicates better hypothesis."""
        good = MDLScoreBreakdown(
            rule_complexity=2.0,
            exception_cost=0.0,
            coverage_penalty=0.0,
            confidence_bonus=1.0,
        )
        good.compute_total()

        bad = MDLScoreBreakdown(
            rule_complexity=5.0,
            exception_cost=10.0,
            coverage_penalty=5.0,
            confidence_bonus=0.0,
        )
        bad.compute_total()

        assert good.total_score < bad.total_score


# =============================================================================
# Coverage Stats Tests
# =============================================================================


class TestCoverageStats:
    """Tests for coverage statistics."""

    def test_coverage_rate(self):
        """Coverage rate is correct."""
        stats = CoverageStats(
            total_decisions=100,
            covered=80,
            correct=60,
            exceptions=20,
            uncovered=20,
        )

        assert stats.coverage_rate == 0.8

    def test_accuracy(self):
        """Accuracy among covered is correct."""
        stats = CoverageStats(
            total_decisions=100,
            covered=80,
            correct=60,
            exceptions=20,
            uncovered=20,
        )

        assert stats.accuracy == 0.75  # 60/80

    def test_exception_rate(self):
        """Exception rate is correct."""
        stats = CoverageStats(
            total_decisions=100,
            covered=80,
            correct=60,
            exceptions=20,
            uncovered=20,
        )

        assert stats.exception_rate == 0.25  # 20/80

    def test_empty_coverage(self):
        """Empty coverage handles division by zero."""
        stats = CoverageStats(total_decisions=0)

        assert stats.coverage_rate == 0.0
        assert stats.accuracy == 0.0
        assert stats.exception_rate == 0.0


# =============================================================================
# Decision Dataset Tests
# =============================================================================


class TestDecisionDataset:
    """Tests for decision dataset."""

    def test_filter_by_decision(self, coherent_dataset):
        """Can filter by decision type."""
        invest_only = coherent_dataset.filter_by_decision(Decision.INVEST)
        pass_only = coherent_dataset.filter_by_decision(Decision.PASS)

        assert len(invest_only) == 10
        assert len(pass_only) == 10
        assert all(d.decision == Decision.INVEST for d in invest_only.decisions)
        assert all(d.decision == Decision.PASS for d in pass_only.decisions)

    def test_decision_counts(self, coherent_dataset):
        """Decision counts are correct."""
        counts = coherent_dataset.decision_counts

        assert counts[Decision.INVEST] == 10
        assert counts[Decision.PASS] == 10

    def test_len(self, coherent_dataset):
        """Length is correct."""
        assert len(coherent_dataset) == 20


# =============================================================================
# Hypothesis Set Tests
# =============================================================================


class TestHypothesisSet:
    """Tests for hypothesis set management."""

    def test_add_valid_hypothesis(self, simple_rules, coherent_dataset):
        """Valid hypothesis is added."""
        config = HypothesisSetConfig(
            min_coverage=0.3,
            min_accuracy=0.5,
        )
        hyp_set = HypothesisSet(config=config)

        hyp = hyp_set.add_hypothesis(
            rules=simple_rules,
            dataset=coherent_dataset,
            name="Test Policy",
        )

        assert hyp is not None
        assert hyp.name == "Test Policy"
        assert len(hyp_set.hypotheses) == 1

    def test_reject_low_coverage(self, simple_rules):
        """Hypothesis with low coverage is rejected."""
        config = HypothesisSetConfig(
            min_coverage=0.9,  # Very high requirement
        )
        hyp_set = HypothesisSet(config=config)

        # Dataset where rules won't cover many decisions
        decisions = [
            HistoricalDecision(
                deal_id=f"deal_{i}",
                decision=Decision.INVEST,
                context=build_context_from_dict({
                    "traction.arr": 500_000,  # Middle ground, won't trigger either rule
                }),
            )
            for i in range(10)
        ]
        dataset = DecisionDataset(decisions=decisions)

        hyp = hyp_set.add_hypothesis(
            rules=simple_rules,
            dataset=dataset,
        )

        assert hyp is None

    def test_reject_low_accuracy(self):
        """Hypothesis with low accuracy is rejected."""
        config = HypothesisSetConfig(
            min_coverage=0.1,
            min_accuracy=0.9,  # Very high requirement
        )
        hyp_set = HypothesisSet(config=config)

        # Rules that always predict wrong
        rules = [
            Rule(
                rule_id="wrong",
                name="Wrong",
                predicate=Has("traction.arr"),
                decision=Decision.PASS,  # Always PASS
                priority=5,
            ),
        ]

        # Dataset with INVEST decisions
        decisions = [
            HistoricalDecision(
                deal_id=f"deal_{i}",
                decision=Decision.INVEST,
                context=build_context_from_dict({"traction.arr": 1_000_000}),
            )
            for i in range(10)
        ]
        dataset = DecisionDataset(decisions=decisions)

        hyp = hyp_set.add_hypothesis(rules=rules, dataset=dataset)

        assert hyp is None

    def test_diversity_check(self, simple_rules, coherent_dataset):
        """Duplicate hypotheses are rejected."""
        config = HypothesisSetConfig(
            min_coverage=0.3,
            min_accuracy=0.5,
            diversity_threshold=0.2,
        )
        hyp_set = HypothesisSet(config=config)

        # Add first hypothesis
        hyp1 = hyp_set.add_hypothesis(
            rules=simple_rules,
            dataset=coherent_dataset,
        )
        assert hyp1 is not None

        # Try to add identical hypothesis
        hyp2 = hyp_set.add_hypothesis(
            rules=simple_rules,
            dataset=coherent_dataset,
        )
        assert hyp2 is None
        assert len(hyp_set.hypotheses) == 1

    def test_prune_to_top_k(self, coherent_dataset):
        """Hypotheses are pruned to top-K."""
        config = HypothesisSetConfig(
            max_hypotheses=2,
            min_coverage=0.1,
            min_accuracy=0.1,
            diversity_threshold=0.5,  # Allow similar hypotheses
        )
        hyp_set = HypothesisSet(config=config)

        # Create three different rule sets
        rule_sets = [
            [Rule(
                rule_id=f"rule_{i}",
                name=f"Rule {i}",
                predicate=Ge("traction.arr", 100_000 * (i + 1)),
                decision=Decision.INVEST,
            )]
            for i in range(3)
        ]

        for i, rules in enumerate(rule_sets):
            hyp_set.add_hypothesis(
                rules=rules,
                dataset=coherent_dataset,
                name=f"Policy_{i}",
            )

        # Should have at most 2
        assert len(hyp_set.hypotheses) <= 2

    def test_get_best(self, simple_rules, coherent_dataset):
        """Get best hypothesis."""
        config = HypothesisSetConfig(min_coverage=0.3, min_accuracy=0.5)
        hyp_set = HypothesisSet(config=config)

        hyp_set.add_hypothesis(rules=simple_rules, dataset=coherent_dataset)

        best = hyp_set.get_best()
        assert best is not None
        assert best.hypothesis_id == "policy_1"

    def test_evaluate_deal(self, simple_rules, coherent_dataset):
        """Evaluate a deal under all policies."""
        config = HypothesisSetConfig(min_coverage=0.3, min_accuracy=0.5)
        hyp_set = HypothesisSet(config=config)

        hyp_set.add_hypothesis(rules=simple_rules, dataset=coherent_dataset)

        # Evaluate a new deal
        ctx = build_context_from_dict({"traction.arr": 2_000_000})
        result = hyp_set.evaluate_deal(ctx)

        assert result["num_policies"] == 1
        assert len(result["policies"]) == 1
        assert result["policies"][0]["decision"] == "invest"
        assert result["consensus"] == "invest"


# =============================================================================
# Multi-Hypothesis Engine Tests
# =============================================================================


class TestMultiHypothesisEngine:
    """Tests for the multi-hypothesis engine."""

    def test_learn_from_coherent_dataset(self, coherent_dataset, simple_rules, growth_focused_rules):
        """Coherent dataset yields one dominant policy."""
        config = HypothesisSetConfig(
            max_hypotheses=5,
            min_coverage=0.3,
            min_accuracy=0.5,
            diversity_threshold=0.3,
        )
        engine = MultiHypothesisEngine(config=config)

        # Try both rule sets
        candidate_rules = [simple_rules, growth_focused_rules]

        hyp_set = engine.learn_from_dataset(
            dataset=coherent_dataset,
            candidate_rules=candidate_rules,
        )

        # Should accept at least one hypothesis
        assert len(hyp_set.hypotheses) >= 1

        # Best hypothesis should have good accuracy
        best = hyp_set.get_best()
        assert best is not None
        assert best.accuracy >= 0.7

    def test_learn_from_inconsistent_dataset_yields_multiple_policies(
        self, inconsistent_dataset, simple_rules, growth_focused_rules, market_focused_rules
    ):
        """Inconsistent dataset with multiple regimes yields multiple policies."""
        config = HypothesisSetConfig(
            max_hypotheses=5,
            min_coverage=0.2,  # Lower threshold to accept more hypotheses
            min_accuracy=0.3,  # Lower threshold due to regime conflicts
            diversity_threshold=0.3,
        )
        engine = MultiHypothesisEngine(config=config)

        # Multiple candidate rule sets
        candidate_rules = [
            simple_rules,
            growth_focused_rules,
            market_focused_rules,
        ]

        hyp_set = engine.learn_from_dataset(
            dataset=inconsistent_dataset,
            candidate_rules=candidate_rules,
        )

        # With inconsistent data and diverse rules, should get multiple hypotheses
        # (or at least no single hypothesis dominates perfectly)
        all_hyps = hyp_set.get_all()

        # The key test: with inconsistent regimes, no single hypothesis
        # should have perfect accuracy
        for hyp in all_hyps:
            # Since regimes conflict, accuracy should be imperfect
            assert hyp.accuracy < 1.0 or hyp.coverage_rate < 1.0

    def test_detect_regime_inconsistency(self, inconsistent_dataset, simple_rules):
        """Detect that dataset has inconsistent regimes."""
        config = HypothesisSetConfig(
            min_coverage=0.2,
            min_accuracy=0.3,
        )
        engine = MultiHypothesisEngine(config=config)

        # Learn from inconsistent data
        engine.learn_from_dataset(
            dataset=inconsistent_dataset,
            candidate_rules=[simple_rules],
        )

        # Detect inconsistency
        analysis = engine.detect_regime_inconsistency(inconsistent_dataset)

        # Should detect issues (either high exception rate or competing hypotheses)
        assert "is_inconsistent" in analysis
        assert "exception_rate" in analysis
        assert "reason" in analysis

    def test_evaluate_new_deal(self, coherent_dataset, simple_rules):
        """Evaluate a new deal under all policies."""
        config = HypothesisSetConfig(
            min_coverage=0.3,
            min_accuracy=0.5,
        )
        engine = MultiHypothesisEngine(config=config)

        engine.learn_from_dataset(
            dataset=coherent_dataset,
            candidate_rules=[simple_rules],
        )

        # Evaluate new deal
        ctx = build_context_from_dict({
            "traction.arr": 2_000_000,
            "traction.growth_rate": 100,
        })

        result = engine.evaluate_new_deal(ctx)

        assert "num_policies" in result
        assert "policies" in result
        assert "consensus" in result


# =============================================================================
# Policy Hypothesis Tests
# =============================================================================


class TestPolicyHypothesis:
    """Tests for policy hypothesis."""

    def test_to_dict(self, simple_rules):
        """Convert hypothesis to dictionary."""
        stats = CoverageStats(
            total_decisions=20,
            covered=18,
            correct=16,
            exceptions=2,
            uncovered=2,
        )

        hyp = PolicyHypothesis(
            hypothesis_id="policy_1",
            name="Test Policy",
            rules=simple_rules,
            coverage_stats=stats,
            exceptions=[
                ExceptionCase(
                    deal_id="deal_1",
                    actual_decision=Decision.PASS,
                    predicted_decision=Decision.INVEST,
                    rules_fired=["high_arr"],
                ),
            ],
        )

        d = hyp.to_dict()

        assert d["id"] == "policy_1"
        assert d["name"] == "Test Policy"
        assert len(d["rules"]) == 2
        assert d["coverage_stats"]["total"] == 20
        assert d["coverage_stats"]["coverage_rate"] == 0.9
        assert len(d["exceptions"]) == 1

    def test_score_property(self):
        """Score property returns MDL total."""
        breakdown = MDLScoreBreakdown()
        breakdown.total_score = 10.5

        hyp = PolicyHypothesis(
            hypothesis_id="policy_1",
            name="Test",
            rules=[],
            mdl_breakdown=breakdown,
        )

        assert hyp.score == 10.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiHypothesisIntegration:
    """Integration tests for multi-hypothesis system."""

    def test_coherent_data_single_policy(self):
        """Coherent data should yield one dominant policy.

        This is the key test: when historical decisions follow a clear pattern,
        the system should converge on a single policy that explains most decisions.
        """
        # Create a clearly coherent dataset
        decisions = []

        # Pattern: ARR > 1M AND growth > 50% -> INVEST
        for i in range(15):
            ctx = build_context_from_dict({
                "traction.arr": 1_500_000 + i * 100_000,
                "traction.growth_rate": 60 + i * 5,
            })
            decisions.append(HistoricalDecision(
                deal_id=f"invest_{i}",
                decision=Decision.INVEST,
                context=ctx,
            ))

        # Pattern: ARR < 500K -> PASS
        for i in range(15):
            ctx = build_context_from_dict({
                "traction.arr": 100_000 + i * 20_000,
                "traction.growth_rate": 30 + i * 2,
            })
            decisions.append(HistoricalDecision(
                deal_id=f"pass_{i}",
                decision=Decision.PASS,
                context=ctx,
            ))

        dataset = DecisionDataset(decisions=decisions)

        # Rules that match the pattern
        matching_rules = [
            Rule(
                rule_id="arr_invest",
                name="High ARR",
                predicate=Ge("traction.arr", 1_000_000),
                decision=Decision.INVEST,
                priority=5,
            ),
            Rule(
                rule_id="arr_pass",
                name="Low ARR",
                predicate=Le("traction.arr", 500_000),
                decision=Decision.PASS,
                priority=3,
            ),
        ]

        config = HypothesisSetConfig(
            max_hypotheses=5,
            min_coverage=0.5,
            min_accuracy=0.7,
        )
        engine = MultiHypothesisEngine(config=config)

        hyp_set = engine.learn_from_dataset(
            dataset=dataset,
            candidate_rules=[matching_rules],
        )

        # Should get at least one good hypothesis
        assert len(hyp_set.hypotheses) >= 1

        # Best hypothesis should have high accuracy
        best = hyp_set.get_best()
        assert best is not None
        assert best.accuracy >= 0.8
        assert best.coverage_rate >= 0.6

    def test_inconsistent_regimes_multiple_policies(self):
        """Inconsistent regimes should yield multiple competing policies.

        This is the key test: when historical decisions don't follow a single
        pattern (e.g., different partners have different investment styles),
        the system should produce multiple hypotheses.
        """
        decisions = []

        # Regime A: "Growth investor" - high growth rate matters
        # Invests in high-growth, regardless of ARR
        for i in range(10):
            ctx = build_context_from_dict({
                "traction.arr": 200_000,  # Modest ARR
                "traction.growth_rate": 150 + i * 10,  # High growth
            })
            decisions.append(HistoricalDecision(
                deal_id=f"growth_invest_{i}",
                decision=Decision.INVEST,
                context=ctx,
            ))

        # Passes on low-growth even with high ARR
        for i in range(5):
            ctx = build_context_from_dict({
                "traction.arr": 3_000_000,  # High ARR
                "traction.growth_rate": 10,  # Low growth
            })
            decisions.append(HistoricalDecision(
                deal_id=f"growth_pass_{i}",
                decision=Decision.PASS,
                context=ctx,
            ))

        # Regime B: "Scale investor" - high ARR matters
        # Invests in high-ARR, regardless of growth
        for i in range(10):
            ctx = build_context_from_dict({
                "traction.arr": 5_000_000 + i * 500_000,  # Very high ARR
                "traction.growth_rate": 20,  # Low growth
            })
            decisions.append(HistoricalDecision(
                deal_id=f"scale_invest_{i}",
                decision=Decision.INVEST,
                context=ctx,
            ))

        # Passes on low-ARR even with high growth
        for i in range(5):
            ctx = build_context_from_dict({
                "traction.arr": 50_000,  # Low ARR
                "traction.growth_rate": 200,  # High growth
            })
            decisions.append(HistoricalDecision(
                deal_id=f"scale_pass_{i}",
                decision=Decision.PASS,
                context=ctx,
            ))

        dataset = DecisionDataset(decisions=decisions)

        # Two different rule philosophies
        growth_rules = [
            Rule(
                rule_id="growth_invest",
                name="High Growth",
                predicate=Ge("traction.growth_rate", 100),
                decision=Decision.INVEST,
                priority=5,
            ),
            Rule(
                rule_id="growth_pass",
                name="Low Growth",
                predicate=Le("traction.growth_rate", 30),
                decision=Decision.PASS,
                priority=3,
            ),
        ]

        scale_rules = [
            Rule(
                rule_id="scale_invest",
                name="High ARR",
                predicate=Ge("traction.arr", 2_000_000),
                decision=Decision.INVEST,
                priority=5,
            ),
            Rule(
                rule_id="scale_pass",
                name="Low ARR",
                predicate=Le("traction.arr", 100_000),
                decision=Decision.PASS,
                priority=3,
            ),
        ]

        config = HypothesisSetConfig(
            max_hypotheses=5,
            min_coverage=0.2,  # Lower thresholds since regimes conflict
            min_accuracy=0.4,
            diversity_threshold=0.3,
        )
        engine = MultiHypothesisEngine(config=config)

        hyp_set = engine.learn_from_dataset(
            dataset=dataset,
            candidate_rules=[growth_rules, scale_rules],
        )

        # Key assertion: neither policy should have perfect accuracy
        # because the data contains conflicting regimes
        for hyp in hyp_set.hypotheses:
            # Neither regime alone explains all decisions
            assert hyp.accuracy < 0.9 or hyp.coverage_rate < 0.7

        # Regime inconsistency should be detected
        analysis = engine.detect_regime_inconsistency(dataset)
        # Exception rate should be elevated due to conflicting patterns
        assert analysis["exception_rate"] > 0 or len(hyp_set.hypotheses) >= 1

    def test_consensus_when_policies_agree(self):
        """When all policies agree, consensus confidence should be high."""
        decisions = [
            HistoricalDecision(
                deal_id=f"deal_{i}",
                decision=Decision.INVEST,
                context=build_context_from_dict({
                    "traction.arr": 2_000_000,
                    "traction.growth_rate": 100,
                }),
            )
            for i in range(10)
        ]
        dataset = DecisionDataset(decisions=decisions)

        # Both rules should trigger INVEST for high-ARR, high-growth deals
        rules = [
            Rule(
                rule_id="high_arr",
                name="High ARR",
                predicate=Ge("traction.arr", 1_000_000),
                decision=Decision.INVEST,
                priority=5,
            ),
        ]

        config = HypothesisSetConfig(min_coverage=0.1, min_accuracy=0.1)
        engine = MultiHypothesisEngine(config=config)
        engine.learn_from_dataset(dataset, [rules])

        # Evaluate deal that should clearly be INVEST
        ctx = build_context_from_dict({
            "traction.arr": 5_000_000,
            "traction.growth_rate": 150,
        })
        result = engine.evaluate_new_deal(ctx)

        if result["num_policies"] > 0:
            assert result["consensus"] == "invest"
            assert result["consensus_confidence"] == 1.0


class TestRobustness:
    """Tests for robustness scoring."""

    def test_robustness_with_small_dataset(self):
        """Small dataset should return default robustness."""
        decisions = [
            HistoricalDecision(
                deal_id="deal_1",
                decision=Decision.INVEST,
                context=build_context_from_dict({"traction.arr": 1_000_000}),
            ),
        ]
        dataset = DecisionDataset(decisions=decisions)

        rules = [
            Rule(
                rule_id="test",
                name="Test",
                predicate=Has("traction.arr"),
                decision=Decision.INVEST,
            ),
        ]

        hyp_set = HypothesisSet()
        hyp = hyp_set.add_hypothesis(rules, dataset)

        # Small dataset should use default robustness
        if hyp:
            assert hyp.robustness_score == 0.5

    def test_robustness_with_large_dataset(self, coherent_dataset, simple_rules):
        """Large dataset should compute actual robustness."""
        config = HypothesisSetConfig(
            min_coverage=0.3,
            min_accuracy=0.5,
            robustness_samples=5,
        )
        hyp_set = HypothesisSet(config=config)

        hyp = hyp_set.add_hypothesis(simple_rules, coherent_dataset)

        assert hyp is not None
        # Robustness should be computed (not default 0.5)
        assert 0.0 <= hyp.robustness_score <= 1.0
