"""
Multi-hypothesis reasoning for JURIS VC DSL.

When decisions are inconsistent or underdetermined, produces K competing
rule-sets ("policy variants") instead of forcing a single one.

Uses Minimum Description Length (MDL) principle to balance:
- Rule simplicity (prefer shorter rules)
- Coverage (explain historical decisions)
- Exceptions (minimize unexplained cases)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence

from .evaluation import Decision, Rule, RuleOutcome, EvaluationTrace, EvalContext
from .predicates_v2 import EvalResult, Predicate

logger = logging.getLogger(__name__)


# =============================================================================
# Historical Decision Data
# =============================================================================


@dataclass
class HistoricalDecision:
    """A historical investment decision for learning."""

    deal_id: str
    decision: Decision
    context: EvalContext
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionDataset:
    """A dataset of historical decisions for policy learning."""

    decisions: list[HistoricalDecision] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def __len__(self) -> int:
        return len(self.decisions)

    def filter_by_decision(self, decision: Decision) -> "DecisionDataset":
        """Filter to decisions of a specific type."""
        return DecisionDataset(
            decisions=[d for d in self.decisions if d.decision == decision],
            name=f"{self.name}_{decision.value}",
        )

    @property
    def decision_counts(self) -> dict[Decision, int]:
        """Count of each decision type."""
        counts = {d: 0 for d in Decision}
        for hist in self.decisions:
            counts[hist.decision] += 1
        return counts


# =============================================================================
# Coverage and Exception Tracking
# =============================================================================


@dataclass
class CoverageResult:
    """Result of evaluating rule coverage on a decision."""

    deal_id: str
    actual_decision: Decision
    predicted_decision: Optional[Decision]
    is_covered: bool  # Did rules make a prediction?
    is_correct: bool  # Does prediction match actual?
    rules_fired: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CoverageStats:
    """Statistics about rule coverage over a dataset."""

    total_decisions: int = 0
    covered: int = 0  # Made a prediction
    correct: int = 0  # Prediction matched actual
    exceptions: int = 0  # Covered but wrong
    uncovered: int = 0  # No prediction made

    # By decision type
    invest_covered: int = 0
    invest_correct: int = 0
    pass_covered: int = 0
    pass_correct: int = 0

    @property
    def coverage_rate(self) -> float:
        """Fraction of decisions covered."""
        if self.total_decisions == 0:
            return 0.0
        return self.covered / self.total_decisions

    @property
    def accuracy(self) -> float:
        """Accuracy among covered decisions."""
        if self.covered == 0:
            return 0.0
        return self.correct / self.covered

    @property
    def exception_rate(self) -> float:
        """Rate of exceptions (wrong predictions)."""
        if self.covered == 0:
            return 0.0
        return self.exceptions / self.covered


@dataclass
class ExceptionCase:
    """A case where the policy made a wrong prediction."""

    deal_id: str
    actual_decision: Decision
    predicted_decision: Decision
    rules_fired: list[str]
    context_summary: dict[str, Any] = field(default_factory=dict)
    severity: float = 1.0  # Higher = more important to explain


# =============================================================================
# MDL Scoring
# =============================================================================


@dataclass
class MDLScoreBreakdown:
    """Breakdown of MDL score components."""

    # Model complexity (description length of rules)
    rule_complexity: float = 0.0
    num_rules: int = 0
    avg_rule_length: float = 0.0

    # Data fit (description length of exceptions)
    exception_cost: float = 0.0
    num_exceptions: int = 0

    # Coverage penalty (uncovered decisions)
    coverage_penalty: float = 0.0
    num_uncovered: int = 0

    # Confidence bonus (high-confidence claims used)
    confidence_bonus: float = 0.0

    # Total MDL score (lower is better)
    total_score: float = 0.0

    def compute_total(self) -> float:
        """Compute total MDL score."""
        self.total_score = (
            self.rule_complexity
            + self.exception_cost
            + self.coverage_penalty
            - self.confidence_bonus
        )
        return self.total_score


class MDLScorer:
    """
    Computes MDL (Minimum Description Length) scores for rule sets.

    MDL balances model complexity with data fit:
    - Shorter rules are preferred (lower description length)
    - Fewer exceptions are preferred
    - Higher coverage is preferred
    """

    def __init__(
        self,
        rule_base_cost: float = 1.0,
        predicate_cost: float = 0.5,
        exception_cost: float = 2.0,
        uncovered_cost: float = 1.0,
        confidence_weight: float = 0.1,
    ):
        """
        Initialize the scorer.

        Args:
            rule_base_cost: Base cost per rule
            predicate_cost: Cost per predicate in a rule
            exception_cost: Cost per exception (wrong prediction)
            uncovered_cost: Cost per uncovered decision
            confidence_weight: Bonus weight for high-confidence claims
        """
        self.rule_base_cost = rule_base_cost
        self.predicate_cost = predicate_cost
        self.exception_cost = exception_cost
        self.uncovered_cost = uncovered_cost
        self.confidence_weight = confidence_weight

    def score_rules(self, rules: list[Rule]) -> float:
        """Score the complexity of a rule set."""
        total = 0.0
        for rule in rules:
            # Base cost
            total += self.rule_base_cost

            # Predicate complexity
            total += self._predicate_complexity(rule.predicate)

        return total

    def _predicate_complexity(self, pred: Predicate) -> float:
        """Compute complexity of a predicate."""
        # Get DSL representation length as proxy for complexity
        dsl = pred.to_dsl()
        # Count operators and terms
        num_ops = dsl.count("(") + dsl.count(",")
        return self.predicate_cost * num_ops

    def score_coverage(
        self,
        rules: list[Rule],
        dataset: DecisionDataset,
        default_decision: Decision = Decision.DEFER,
    ) -> tuple[CoverageStats, list[ExceptionCase], MDLScoreBreakdown]:
        """
        Score rule coverage and compute MDL breakdown.

        Returns:
            Tuple of (coverage_stats, exceptions, mdl_breakdown)
        """
        from .evaluation import RuleEngine

        stats = CoverageStats(total_decisions=len(dataset))
        exceptions: list[ExceptionCase] = []

        # Total confidence of claims used
        total_confidence = 0.0
        confidence_count = 0

        engine = RuleEngine(rules, default_decision=default_decision)

        for hist in dataset.decisions:
            trace = engine.evaluate(hist.context)
            predicted = trace.final_decision

            # Track confidence
            for outcome in trace.rule_outcomes:
                if outcome.result == EvalResult.TRUE:
                    for f in outcome.fields_used:
                        fv = hist.context.get_field(f)
                        if fv.exists:
                            total_confidence += fv.confidence
                            confidence_count += 1

            # Track coverage
            is_covered = predicted != Decision.DEFER
            is_correct = predicted == hist.decision

            if is_covered:
                stats.covered += 1
                if is_correct:
                    stats.correct += 1
                else:
                    stats.exceptions += 1
                    exceptions.append(ExceptionCase(
                        deal_id=hist.deal_id,
                        actual_decision=hist.decision,
                        predicted_decision=predicted,
                        rules_fired=[o.rule_id for o in trace.rules_fired],
                        severity=1.0 if hist.decision == Decision.PASS else 0.8,
                    ))

                # Track by decision type
                if hist.decision == Decision.INVEST:
                    stats.invest_covered += 1
                    if is_correct:
                        stats.invest_correct += 1
                elif hist.decision == Decision.PASS:
                    stats.pass_covered += 1
                    if is_correct:
                        stats.pass_correct += 1
            else:
                stats.uncovered += 1

        # Compute MDL breakdown
        breakdown = MDLScoreBreakdown(
            rule_complexity=self.score_rules(rules),
            num_rules=len(rules),
            avg_rule_length=sum(len(r.predicate.to_dsl()) for r in rules) / max(1, len(rules)),
            exception_cost=self.exception_cost * len(exceptions),
            num_exceptions=len(exceptions),
            coverage_penalty=self.uncovered_cost * stats.uncovered,
            num_uncovered=stats.uncovered,
            confidence_bonus=self.confidence_weight * (
                total_confidence / max(1, confidence_count)
            ) if confidence_count > 0 else 0.0,
        )
        breakdown.compute_total()

        return stats, exceptions, breakdown


# =============================================================================
# Policy Hypothesis
# =============================================================================


@dataclass
class PolicyHypothesis:
    """
    A hypothesis about the investment policy.

    Represents one possible rule-set that could explain historical decisions.
    """

    hypothesis_id: str
    name: str
    rules: list[Rule]

    # Coverage information
    coverage_stats: CoverageStats = field(default_factory=CoverageStats)
    exceptions: list[ExceptionCase] = field(default_factory=list)

    # Scoring
    mdl_breakdown: MDLScoreBreakdown = field(default_factory=MDLScoreBreakdown)
    robustness_score: float = 0.0  # How stable under perturbation

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

    @property
    def score(self) -> float:
        """Overall score (lower MDL = better)."""
        return self.mdl_breakdown.total_score

    @property
    def coverage_rate(self) -> float:
        """Coverage rate."""
        return self.coverage_stats.coverage_rate

    @property
    def accuracy(self) -> float:
        """Accuracy among covered decisions."""
        return self.coverage_stats.accuracy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.hypothesis_id,
            "name": self.name,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "predicate": r.predicate.to_dsl(),
                    "decision": r.decision.value,
                    "priority": r.priority,
                }
                for r in self.rules
            ],
            "coverage_stats": {
                "total": self.coverage_stats.total_decisions,
                "covered": self.coverage_stats.covered,
                "correct": self.coverage_stats.correct,
                "exceptions": self.coverage_stats.exceptions,
                "uncovered": self.coverage_stats.uncovered,
                "coverage_rate": self.coverage_stats.coverage_rate,
                "accuracy": self.coverage_stats.accuracy,
            },
            "exceptions": [
                {
                    "deal_id": e.deal_id,
                    "actual": e.actual_decision.value,
                    "predicted": e.predicted_decision.value,
                    "rules_fired": e.rules_fired,
                }
                for e in self.exceptions
            ],
            "score_breakdown": {
                "rule_complexity": self.mdl_breakdown.rule_complexity,
                "exception_cost": self.mdl_breakdown.exception_cost,
                "coverage_penalty": self.mdl_breakdown.coverage_penalty,
                "confidence_bonus": self.mdl_breakdown.confidence_bonus,
                "total_score": self.mdl_breakdown.total_score,
            },
            "robustness_score": self.robustness_score,
        }


# =============================================================================
# Hypothesis Set (Top-K Management)
# =============================================================================


@dataclass
class HypothesisSetConfig:
    """Configuration for hypothesis set management."""

    # Maximum hypotheses to maintain
    max_hypotheses: int = 5

    # Minimum coverage required
    min_coverage: float = 0.5

    # Minimum accuracy required
    min_accuracy: float = 0.6

    # Diversity threshold (reject if too similar)
    diversity_threshold: float = 0.2

    # Robustness evaluation
    robustness_samples: int = 10
    robustness_noise: float = 0.1


class HypothesisSet:
    """
    Manages a set of top-K policy hypotheses.

    Maintains diverse, high-quality hypotheses under MDL objective.
    """

    def __init__(
        self,
        config: Optional[HypothesisSetConfig] = None,
        scorer: Optional[MDLScorer] = None,
    ):
        """
        Initialize hypothesis set.

        Args:
            config: Configuration options
            scorer: MDL scorer to use
        """
        self.config = config or HypothesisSetConfig()
        self.scorer = scorer or MDLScorer()
        self.hypotheses: list[PolicyHypothesis] = []
        self._next_id = 1

    def add_hypothesis(
        self,
        rules: list[Rule],
        dataset: DecisionDataset,
        name: Optional[str] = None,
        description: str = "",
    ) -> Optional[PolicyHypothesis]:
        """
        Add a new hypothesis if it meets quality criteria.

        Args:
            rules: Rules for the hypothesis
            dataset: Dataset to evaluate against
            name: Optional name
            description: Description

        Returns:
            Added hypothesis if accepted, None if rejected
        """
        # Score the hypothesis
        stats, exceptions, breakdown = self.scorer.score_coverage(rules, dataset)

        # Check minimum coverage
        if stats.coverage_rate < self.config.min_coverage:
            logger.debug(
                f"Hypothesis rejected: coverage {stats.coverage_rate:.2%} "
                f"< min {self.config.min_coverage:.2%}"
            )
            return None

        # Check minimum accuracy
        if stats.accuracy < self.config.min_accuracy:
            logger.debug(
                f"Hypothesis rejected: accuracy {stats.accuracy:.2%} "
                f"< min {self.config.min_accuracy:.2%}"
            )
            return None

        # Check diversity against existing hypotheses
        if not self._is_diverse_enough(rules):
            logger.debug("Hypothesis rejected: too similar to existing")
            return None

        # Create hypothesis
        hypothesis = PolicyHypothesis(
            hypothesis_id=f"policy_{self._next_id}",
            name=name or f"Policy {self._next_id}",
            rules=rules,
            coverage_stats=stats,
            exceptions=exceptions,
            mdl_breakdown=breakdown,
            description=description,
        )
        self._next_id += 1

        # Compute robustness
        hypothesis.robustness_score = self._compute_robustness(hypothesis, dataset)

        # Add to set
        self.hypotheses.append(hypothesis)

        # Prune to top-K
        self._prune_to_top_k()

        return hypothesis

    def _is_diverse_enough(self, new_rules: list[Rule]) -> bool:
        """Check if new rules are diverse enough from existing."""
        if not self.hypotheses:
            return True

        new_rule_dsls = {r.predicate.to_dsl() for r in new_rules}

        for hyp in self.hypotheses:
            existing_dsls = {r.predicate.to_dsl() for r in hyp.rules}

            # Compute Jaccard similarity
            intersection = len(new_rule_dsls & existing_dsls)
            union = len(new_rule_dsls | existing_dsls)

            if union > 0:
                similarity = intersection / union
                if similarity > (1 - self.config.diversity_threshold):
                    return False

        return True

    def _compute_robustness(
        self,
        hypothesis: PolicyHypothesis,
        dataset: DecisionDataset,
    ) -> float:
        """
        Compute robustness score via perturbation analysis.

        Tests how stable the hypothesis is when data is slightly perturbed.
        """
        if len(dataset) < 5:
            return 0.5  # Not enough data for robustness

        import random

        original_accuracy = hypothesis.accuracy
        accuracy_deltas = []

        for _ in range(self.config.robustness_samples):
            # Create perturbed dataset (drop random 10% of decisions)
            sample_size = max(1, int(len(dataset) * (1 - self.config.robustness_noise)))
            sampled = random.sample(dataset.decisions, sample_size)
            perturbed = DecisionDataset(decisions=sampled)

            # Evaluate on perturbed dataset
            stats, _, _ = self.scorer.score_coverage(hypothesis.rules, perturbed)

            accuracy_deltas.append(abs(stats.accuracy - original_accuracy))

        # Robustness = 1 - average accuracy change
        avg_delta = sum(accuracy_deltas) / len(accuracy_deltas) if accuracy_deltas else 0
        return max(0.0, 1.0 - avg_delta * 5)  # Scale so 20% change = 0 robustness

    def _prune_to_top_k(self) -> None:
        """Prune hypotheses to keep only top-K by score."""
        if len(self.hypotheses) <= self.config.max_hypotheses:
            return

        # Sort by MDL score (lower is better)
        self.hypotheses.sort(key=lambda h: h.score)

        # Keep top-K
        self.hypotheses = self.hypotheses[:self.config.max_hypotheses]

    def get_best(self) -> Optional[PolicyHypothesis]:
        """Get the best hypothesis by MDL score."""
        if not self.hypotheses:
            return None
        return min(self.hypotheses, key=lambda h: h.score)

    def get_all(self) -> list[PolicyHypothesis]:
        """Get all hypotheses sorted by score."""
        return sorted(self.hypotheses, key=lambda h: h.score)

    def evaluate_deal(
        self,
        context: EvalContext,
        default_decision: Decision = Decision.DEFER,
    ) -> dict[str, Any]:
        """
        Evaluate a new deal under all policies.

        Returns:
            Dict with evaluation results from each policy
        """
        from .evaluation import RuleEngine

        results = {
            "num_policies": len(self.hypotheses),
            "policies": [],
            "consensus": None,
            "consensus_confidence": 0.0,
        }

        decisions = []

        for hyp in self.hypotheses:
            engine = RuleEngine(hyp.rules, default_decision=default_decision)
            trace = engine.evaluate(context)

            policy_result = {
                "policy_id": hyp.hypothesis_id,
                "policy_name": hyp.name,
                "decision": trace.final_decision.value,
                "confidence": trace.decision_confidence,
                "rules_fired": [o.rule_id for o in trace.rules_fired],
                "policy_accuracy": hyp.accuracy,
                "policy_coverage": hyp.coverage_rate,
            }
            results["policies"].append(policy_result)
            decisions.append(trace.final_decision)

        # Compute consensus
        if decisions:
            decision_counts = {}
            for d in decisions:
                decision_counts[d] = decision_counts.get(d, 0) + 1

            most_common = max(decision_counts, key=decision_counts.get)
            results["consensus"] = most_common.value
            results["consensus_confidence"] = decision_counts[most_common] / len(decisions)

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_hypotheses": len(self.hypotheses),
            "config": {
                "max_hypotheses": self.config.max_hypotheses,
                "min_coverage": self.config.min_coverage,
                "min_accuracy": self.config.min_accuracy,
            },
            "hypotheses": [h.to_dict() for h in self.get_all()],
        }


# =============================================================================
# Multi-Hypothesis Engine
# =============================================================================


class MultiHypothesisEngine:
    """
    Engine for learning and maintaining multiple policy hypotheses.

    Used when historical decisions are inconsistent or underdetermined.
    """

    def __init__(
        self,
        config: Optional[HypothesisSetConfig] = None,
        scorer: Optional[MDLScorer] = None,
    ):
        """
        Initialize the engine.

        Args:
            config: Hypothesis set configuration
            scorer: MDL scorer
        """
        self.config = config or HypothesisSetConfig()
        self.scorer = scorer or MDLScorer()
        self.hypothesis_set = HypothesisSet(config=self.config, scorer=self.scorer)

    def learn_from_dataset(
        self,
        dataset: DecisionDataset,
        candidate_rules: list[list[Rule]],
    ) -> HypothesisSet:
        """
        Learn hypotheses from a dataset using candidate rule sets.

        Args:
            dataset: Historical decisions
            candidate_rules: List of candidate rule sets to evaluate

        Returns:
            HypothesisSet with top-K hypotheses
        """
        logger.info(
            f"Learning from {len(dataset)} decisions with "
            f"{len(candidate_rules)} candidate rule sets"
        )

        for i, rules in enumerate(candidate_rules):
            self.hypothesis_set.add_hypothesis(
                rules=rules,
                dataset=dataset,
                name=f"Candidate_{i+1}",
            )

        logger.info(f"Accepted {len(self.hypothesis_set.hypotheses)} hypotheses")

        return self.hypothesis_set

    def detect_regime_inconsistency(
        self,
        dataset: DecisionDataset,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Detect if dataset contains inconsistent decision regimes.

        Returns:
            Analysis of potential regime inconsistency
        """
        # Check if best hypothesis has high exception rate
        best = self.hypothesis_set.get_best()
        if not best:
            return {
                "is_inconsistent": False,
                "reason": "No hypotheses learned yet",
            }

        exception_rate = best.coverage_stats.exception_rate

        # Check if multiple hypotheses have similar scores
        all_hyps = self.hypothesis_set.get_all()
        if len(all_hyps) >= 2:
            score_variance = _variance([h.score for h in all_hyps])
        else:
            score_variance = 0.0

        is_inconsistent = exception_rate > threshold or (
            len(all_hyps) >= 2 and score_variance < 0.1
        )

        return {
            "is_inconsistent": is_inconsistent,
            "exception_rate": exception_rate,
            "num_hypotheses": len(all_hyps),
            "score_variance": score_variance,
            "reason": (
                "High exception rate suggests multiple decision regimes"
                if exception_rate > threshold
                else "Multiple hypotheses with similar scores"
                if is_inconsistent
                else "Dataset appears consistent"
            ),
        }

    def evaluate_new_deal(
        self,
        context: EvalContext,
    ) -> dict[str, Any]:
        """
        Evaluate a new deal under all policies.

        Returns comprehensive evaluation including consensus.
        """
        return self.hypothesis_set.evaluate_deal(context)


def _variance(values: list[float]) -> float:
    """Compute variance of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
