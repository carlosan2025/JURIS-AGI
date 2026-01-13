"""
Uncertainty quantification for JURIS VC DSL.

Provides two types of uncertainty signals:

1. Epistemic Uncertainty (model uncertainty / uncertainty about the policy)
   - Number of near-equivalent policies
   - Disagreement between policies on new deals
   - Sensitivity to claim removal

2. Aleatoric Uncertainty (data uncertainty / uncertainty in the evidence)
   - Low confidence claims used in decision
   - High variance in time-series data
   - Large unresolved conflict clusters

Also provides actionable suggestions for reducing uncertainty by
identifying which missing claims would help most.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from .evaluation import Decision, Rule, RuleEngine, EvalContext
from .hypothesis import (
    DecisionDataset,
    HypothesisSet,
    PolicyHypothesis,
    MDLScorer,
)
from .predicates_v2 import EvalResult, FieldValue

logger = logging.getLogger(__name__)


# =============================================================================
# Uncertainty Types
# =============================================================================


class UncertaintyLevel(Enum):
    """Categorical uncertainty levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class UncertaintyReason:
    """A reason contributing to uncertainty."""

    category: str  # "epistemic" or "aleatoric"
    code: str  # Machine-readable code
    description: str  # Human-readable description
    severity: float  # 0-1 contribution to uncertainty
    affected_fields: list[str] = field(default_factory=list)
    suggestion: Optional[str] = None


# =============================================================================
# Epistemic Uncertainty (Model/Policy Uncertainty)
# =============================================================================


@dataclass
class EpistemicUncertainty:
    """
    Epistemic uncertainty signals - uncertainty about the policy.

    These can be reduced by gathering more training examples.
    """

    # Number of near-equivalent policies
    num_equivalent_policies: int = 0
    policy_score_variance: float = 0.0

    # Policy disagreement on this deal
    policy_agreement_rate: float = 1.0  # 1.0 = all agree, 0.5 = split
    disagreeing_policies: list[str] = field(default_factory=list)
    majority_decision: Optional[Decision] = None
    minority_decision: Optional[Decision] = None

    # Sensitivity to claim removal
    most_sensitive_claims: list[str] = field(default_factory=list)
    max_sensitivity: float = 0.0  # Decision flip probability

    # Overall epistemic score (0=certain, 1=uncertain)
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute overall epistemic uncertainty score."""
        components = []

        # Policy equivalence (more equivalent policies = more uncertainty)
        if self.num_equivalent_policies > 1:
            equiv_score = min(1.0, (self.num_equivalent_policies - 1) / 4)
            components.append(equiv_score * 0.3)

        # Policy disagreement (lower agreement = more uncertainty)
        disagree_score = 1.0 - self.policy_agreement_rate
        components.append(disagree_score * 0.4)

        # Sensitivity (higher sensitivity = more uncertainty)
        components.append(self.max_sensitivity * 0.3)

        self.score = sum(components)
        return self.score

    @property
    def level(self) -> UncertaintyLevel:
        """Categorical uncertainty level."""
        if self.score < 0.2:
            return UncertaintyLevel.LOW
        elif self.score < 0.4:
            return UncertaintyLevel.MEDIUM
        elif self.score < 0.6:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.VERY_HIGH

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "level": self.level.value,
            "num_equivalent_policies": self.num_equivalent_policies,
            "policy_agreement_rate": self.policy_agreement_rate,
            "disagreeing_policies": self.disagreeing_policies,
            "majority_decision": self.majority_decision.value if self.majority_decision else None,
            "minority_decision": self.minority_decision.value if self.minority_decision else None,
            "most_sensitive_claims": self.most_sensitive_claims,
            "max_sensitivity": self.max_sensitivity,
        }


# =============================================================================
# Aleatoric Uncertainty (Data/Evidence Uncertainty)
# =============================================================================


@dataclass
class AleatoricUncertainty:
    """
    Aleatoric uncertainty signals - uncertainty in the evidence.

    These cannot be reduced by more training data, only by better evidence.
    """

    # Low confidence claims
    low_confidence_claims: list[str] = field(default_factory=list)
    avg_confidence_of_used_claims: float = 1.0
    min_confidence_of_used_claims: float = 1.0

    # Time-series variance
    high_variance_fields: list[str] = field(default_factory=list)
    max_coefficient_of_variation: float = 0.0

    # Conflict clusters
    unresolved_conflicts: int = 0
    conflict_severity_sum: float = 0.0
    conflicting_fields: list[str] = field(default_factory=list)

    # Missing critical fields
    missing_fields: list[str] = field(default_factory=list)
    missing_field_importance: dict[str, float] = field(default_factory=dict)

    # Overall aleatoric score (0=certain, 1=uncertain)
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute overall aleatoric uncertainty score."""
        components = []

        # Low confidence (lower avg confidence = more uncertainty)
        if self.avg_confidence_of_used_claims < 1.0:
            conf_score = 1.0 - self.avg_confidence_of_used_claims
            components.append(conf_score * 0.3)

        # Time-series variance
        if self.max_coefficient_of_variation > 0:
            var_score = min(1.0, self.max_coefficient_of_variation / 0.5)
            components.append(var_score * 0.2)

        # Conflicts
        if self.unresolved_conflicts > 0:
            conflict_score = min(1.0, self.conflict_severity_sum / 2.0)
            components.append(conflict_score * 0.25)

        # Missing fields
        if self.missing_fields:
            missing_score = min(1.0, len(self.missing_fields) / 5)
            components.append(missing_score * 0.25)

        self.score = sum(components)
        return self.score

    @property
    def level(self) -> UncertaintyLevel:
        """Categorical uncertainty level."""
        if self.score < 0.2:
            return UncertaintyLevel.LOW
        elif self.score < 0.4:
            return UncertaintyLevel.MEDIUM
        elif self.score < 0.6:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.VERY_HIGH

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "level": self.level.value,
            "low_confidence_claims": self.low_confidence_claims,
            "avg_confidence": self.avg_confidence_of_used_claims,
            "min_confidence": self.min_confidence_of_used_claims,
            "high_variance_fields": self.high_variance_fields,
            "max_coefficient_of_variation": self.max_coefficient_of_variation,
            "unresolved_conflicts": self.unresolved_conflicts,
            "conflicting_fields": self.conflicting_fields,
            "missing_fields": self.missing_fields,
        }


# =============================================================================
# Information Request (Suggestions to Reduce Uncertainty)
# =============================================================================


@dataclass
class InformationRequest:
    """A suggested piece of information that would reduce uncertainty."""

    field: str  # Field path e.g. "traction.arr"
    description: str  # Human-readable description
    importance: float  # 0-1 how much it would reduce uncertainty
    uncertainty_type: str  # "epistemic" or "aleatoric"
    reason: str  # Why this information would help


@dataclass
class UncertaintyReport:
    """
    Complete uncertainty report for a decision.

    Includes both uncertainty types, reasons, and actionable suggestions.
    """

    # Decision info
    deal_id: str = ""
    decision: Optional[Decision] = None
    decision_confidence: float = 0.0

    # Uncertainty scores
    epistemic: EpistemicUncertainty = field(default_factory=EpistemicUncertainty)
    aleatoric: AleatoricUncertainty = field(default_factory=AleatoricUncertainty)

    # Combined score
    total_uncertainty: float = 0.0

    # Top reasons for uncertainty
    top_reasons: list[UncertaintyReason] = field(default_factory=list)

    # Information requests (suggestions to reduce uncertainty)
    information_requests: list[InformationRequest] = field(default_factory=list)

    # Should we refuse / request more info?
    should_defer: bool = False
    defer_reason: Optional[str] = None

    def compute_total(self) -> float:
        """Compute total uncertainty from components."""
        self.epistemic.compute_score()
        self.aleatoric.compute_score()

        # Weighted combination (epistemic slightly more important)
        self.total_uncertainty = (
            0.55 * self.epistemic.score +
            0.45 * self.aleatoric.score
        )
        return self.total_uncertainty

    @property
    def level(self) -> UncertaintyLevel:
        """Overall uncertainty level."""
        if self.total_uncertainty < 0.2:
            return UncertaintyLevel.LOW
        elif self.total_uncertainty < 0.4:
            return UncertaintyLevel.MEDIUM
        elif self.total_uncertainty < 0.6:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.VERY_HIGH

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deal_id": self.deal_id,
            "decision": self.decision.value if self.decision else None,
            "decision_confidence": self.decision_confidence,
            "epistemic_score": self.epistemic.score,
            "aleatoric_score": self.aleatoric.score,
            "total_uncertainty": self.total_uncertainty,
            "uncertainty_level": self.level.value,
            "epistemic": self.epistemic.to_dict(),
            "aleatoric": self.aleatoric.to_dict(),
            "top_reasons": [
                {
                    "category": r.category,
                    "code": r.code,
                    "description": r.description,
                    "severity": r.severity,
                }
                for r in self.top_reasons
            ],
            "information_requests": [
                {
                    "field": r.field,
                    "description": r.description,
                    "importance": r.importance,
                    "reason": r.reason,
                }
                for r in self.information_requests
            ],
            "should_defer": self.should_defer,
            "defer_reason": self.defer_reason,
        }


# =============================================================================
# Uncertainty Analyzer
# =============================================================================


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty analysis."""

    # Thresholds
    low_confidence_threshold: float = 0.7
    high_variance_threshold: float = 0.3  # Coefficient of variation
    sensitivity_sample_count: int = 5

    # Defer thresholds
    defer_epistemic_threshold: float = 0.6
    defer_aleatoric_threshold: float = 0.7
    defer_total_threshold: float = 0.5

    # Policy equivalence threshold (MDL score difference)
    policy_equivalence_threshold: float = 0.5

    # Maximum information requests to suggest
    max_information_requests: int = 5

    # Field importance for missing field analysis
    critical_fields: list[str] = field(default_factory=lambda: [
        "traction.arr",
        "traction.growth_rate",
        "deal.valuation",
        "team.founders_experience",
        "market.tam",
        "financial.runway_months",
    ])


class UncertaintyAnalyzer:
    """
    Analyzes uncertainty in investment decisions.

    Computes both epistemic and aleatoric uncertainty and provides
    actionable suggestions for reducing uncertainty.
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """Initialize the analyzer."""
        self.config = config or UncertaintyConfig()
        self.scorer = MDLScorer()

    def analyze_epistemic(
        self,
        context: EvalContext,
        hypothesis_set: HypothesisSet,
    ) -> EpistemicUncertainty:
        """
        Analyze epistemic uncertainty from policy disagreement.

        Args:
            context: Deal context
            hypothesis_set: Set of learned policies

        Returns:
            EpistemicUncertainty analysis
        """
        result = EpistemicUncertainty()

        hypotheses = hypothesis_set.get_all()
        if not hypotheses:
            return result

        # Count near-equivalent policies
        if len(hypotheses) >= 2:
            best_score = hypotheses[0].score
            equivalent = [
                h for h in hypotheses
                if abs(h.score - best_score) < self.config.policy_equivalence_threshold
            ]
            result.num_equivalent_policies = len(equivalent)
            result.policy_score_variance = _variance([h.score for h in hypotheses])

        # Evaluate under each policy
        decisions = []
        for hyp in hypotheses:
            engine = RuleEngine(hyp.rules)
            trace = engine.evaluate(context)
            decisions.append((hyp.hypothesis_id, trace.final_decision))

        # Calculate agreement rate
        if decisions:
            decision_counts: dict[Decision, int] = {}
            for _, dec in decisions:
                decision_counts[dec] = decision_counts.get(dec, 0) + 1

            most_common = max(decision_counts, key=decision_counts.get)
            result.majority_decision = most_common
            result.policy_agreement_rate = decision_counts[most_common] / len(decisions)

            # Find disagreeing policies
            result.disagreeing_policies = [
                hyp_id for hyp_id, dec in decisions
                if dec != most_common
            ]

            # Find minority decision
            if len(decision_counts) > 1:
                for dec, count in decision_counts.items():
                    if dec != most_common:
                        result.minority_decision = dec
                        break

        # Sensitivity analysis (sample claim removal)
        result.most_sensitive_claims, result.max_sensitivity = self._analyze_sensitivity(
            context, hypotheses[0] if hypotheses else None
        )

        result.compute_score()
        return result

    def _analyze_sensitivity(
        self,
        context: EvalContext,
        hypothesis: Optional[PolicyHypothesis],
    ) -> tuple[list[str], float]:
        """Analyze sensitivity to claim removal."""
        if not hypothesis:
            return [], 0.0

        engine = RuleEngine(hypothesis.rules)
        original_trace = engine.evaluate(context)
        original_decision = original_trace.final_decision

        sensitive_claims = []
        max_sensitivity = 0.0

        # Get fields that were used in the decision
        used_fields = set()
        for outcome in original_trace.rule_outcomes:
            if outcome.result == EvalResult.TRUE:
                used_fields.update(outcome.fields_used)

        # Test removing each field
        for field_name in list(used_fields)[:self.config.sensitivity_sample_count]:
            # Create modified context without this field
            modified_fields = {
                k: v for k, v in context.fields.items()
                if k != field_name
            }
            modified_context = EvalContext(fields=modified_fields)

            # Evaluate
            modified_trace = engine.evaluate(modified_context)

            # Check if decision changed
            if modified_trace.final_decision != original_decision:
                sensitive_claims.append(field_name)
                max_sensitivity = max(max_sensitivity, 1.0)
            elif modified_trace.decision_confidence < original_trace.decision_confidence - 0.1:
                max_sensitivity = max(max_sensitivity, 0.5)

        return sensitive_claims, max_sensitivity

    def analyze_aleatoric(
        self,
        context: EvalContext,
        rules: list[Rule],
        conflicts: Optional[list[Any]] = None,
    ) -> AleatoricUncertainty:
        """
        Analyze aleatoric uncertainty from evidence quality.

        Args:
            context: Deal context
            rules: Rules being used
            conflicts: List of Conflict objects

        Returns:
            AleatoricUncertainty analysis
        """
        result = AleatoricUncertainty()

        # Identify fields used by rules
        used_fields = set()
        for rule in rules:
            used_fields.update(rule.predicate.get_fields())

        # Analyze confidence of used claims
        confidences = []
        for field_name in used_fields:
            fv = context.get_field(field_name)
            if fv.exists:
                confidences.append(fv.confidence)
                if fv.confidence < self.config.low_confidence_threshold:
                    result.low_confidence_claims.append(field_name)
            else:
                result.missing_fields.append(field_name)

        if confidences:
            result.avg_confidence_of_used_claims = sum(confidences) / len(confidences)
            result.min_confidence_of_used_claims = min(confidences)

        # Analyze time-series variance
        for field_name, fv in context.fields.items():
            if not fv.exists:
                continue

            # Check for timeseries data
            ts_key = f"{field_name}_timeseries"
            ts_fv = context.get_field(ts_key)
            if ts_fv.exists and isinstance(ts_fv.value, list):
                cv = self._compute_coefficient_of_variation(ts_fv.value)
                if cv > self.config.high_variance_threshold:
                    result.high_variance_fields.append(field_name)
                    result.max_coefficient_of_variation = max(
                        result.max_coefficient_of_variation, cv
                    )

        # Analyze conflicts
        if conflicts:
            result.unresolved_conflicts = len(conflicts)
            for conflict in conflicts:
                if hasattr(conflict, "severity"):
                    result.conflict_severity_sum += conflict.severity
                if hasattr(conflict, "claim_ids"):
                    # Extract field names from claim IDs if possible
                    result.conflicting_fields.extend(
                        [cid.split(".")[-1] for cid in conflict.claim_ids[:2]]
                    )

        # Compute importance of missing fields
        for field_name in result.missing_fields:
            importance = self._estimate_field_importance(field_name, rules)
            result.missing_field_importance[field_name] = importance

        result.compute_score()
        return result

    def _compute_coefficient_of_variation(self, timeseries_data: list) -> float:
        """Compute coefficient of variation for time series."""
        if len(timeseries_data) < 2:
            return 0.0

        values = []
        for point in timeseries_data:
            if isinstance(point, dict) and "value" in point:
                values.append(point["value"])
            elif hasattr(point, "value"):
                values.append(point.value)

        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        return std_dev / abs(mean)

    def _estimate_field_importance(
        self,
        field_name: str,
        rules: list[Rule],
    ) -> float:
        """Estimate importance of a missing field."""
        # Base importance from whether it's in critical fields
        if field_name in self.config.critical_fields:
            base_importance = 0.8
        else:
            base_importance = 0.3

        # Increase importance if field is used in high-priority rules
        for rule in rules:
            if field_name in rule.predicate.get_fields():
                base_importance = max(base_importance, 0.5 + rule.priority * 0.05)

        return min(1.0, base_importance)

    def generate_information_requests(
        self,
        epistemic: EpistemicUncertainty,
        aleatoric: AleatoricUncertainty,
        rules: list[Rule],
    ) -> list[InformationRequest]:
        """
        Generate suggestions for reducing uncertainty.

        Returns list of information requests sorted by importance.
        """
        requests = []

        # Request for low-confidence claims (improve confidence)
        for field_name in aleatoric.low_confidence_claims[:3]:
            requests.append(InformationRequest(
                field=field_name,
                description=f"Improve confidence for {field_name}",
                importance=0.7,
                uncertainty_type="aleatoric",
                reason="Low confidence claim affects decision reliability",
            ))

        # Request for missing fields
        sorted_missing = sorted(
            aleatoric.missing_field_importance.items(),
            key=lambda x: -x[1]
        )
        for field_name, importance in sorted_missing[:3]:
            requests.append(InformationRequest(
                field=field_name,
                description=f"Provide value for {field_name}",
                importance=importance,
                uncertainty_type="aleatoric",
                reason="Missing field required by decision rules",
            ))

        # Request for sensitive claims (epistemic)
        for field_name in epistemic.most_sensitive_claims[:2]:
            requests.append(InformationRequest(
                field=field_name,
                description=f"Verify or corroborate {field_name}",
                importance=0.8,
                uncertainty_type="epistemic",
                reason="Decision is sensitive to this claim - additional validation recommended",
            ))

        # Request for conflicting fields
        for field_name in aleatoric.conflicting_fields[:2]:
            requests.append(InformationRequest(
                field=field_name,
                description=f"Resolve conflict in {field_name}",
                importance=0.75,
                uncertainty_type="aleatoric",
                reason="Conflicting information from multiple sources",
            ))

        # Request for high-variance time series
        for field_name in aleatoric.high_variance_fields[:2]:
            requests.append(InformationRequest(
                field=field_name,
                description=f"Explain variance in {field_name} trend",
                importance=0.6,
                uncertainty_type="aleatoric",
                reason="High volatility in time-series data",
            ))

        # Sort by importance and limit
        requests.sort(key=lambda r: -r.importance)
        return requests[:self.config.max_information_requests]

    def compile_reasons(
        self,
        epistemic: EpistemicUncertainty,
        aleatoric: AleatoricUncertainty,
    ) -> list[UncertaintyReason]:
        """Compile top reasons for uncertainty."""
        reasons = []

        # Epistemic reasons
        if epistemic.num_equivalent_policies > 1:
            reasons.append(UncertaintyReason(
                category="epistemic",
                code="MULTIPLE_POLICIES",
                description=f"{epistemic.num_equivalent_policies} near-equivalent policies found",
                severity=min(0.8, 0.2 * epistemic.num_equivalent_policies),
            ))

        if epistemic.policy_agreement_rate < 0.8:
            reasons.append(UncertaintyReason(
                category="epistemic",
                code="POLICY_DISAGREEMENT",
                description=f"Policies disagree: {len(epistemic.disagreeing_policies)} minority votes",
                severity=1.0 - epistemic.policy_agreement_rate,
            ))

        if epistemic.max_sensitivity > 0.5:
            reasons.append(UncertaintyReason(
                category="epistemic",
                code="HIGH_SENSITIVITY",
                description=f"Decision sensitive to {len(epistemic.most_sensitive_claims)} claims",
                severity=epistemic.max_sensitivity,
                affected_fields=epistemic.most_sensitive_claims,
            ))

        # Aleatoric reasons
        if aleatoric.low_confidence_claims:
            reasons.append(UncertaintyReason(
                category="aleatoric",
                code="LOW_CONFIDENCE",
                description=f"{len(aleatoric.low_confidence_claims)} low-confidence claims used",
                severity=1.0 - aleatoric.avg_confidence_of_used_claims,
                affected_fields=aleatoric.low_confidence_claims,
            ))

        if aleatoric.high_variance_fields:
            reasons.append(UncertaintyReason(
                category="aleatoric",
                code="HIGH_VARIANCE",
                description=f"High variance in {len(aleatoric.high_variance_fields)} time-series",
                severity=min(0.8, aleatoric.max_coefficient_of_variation),
                affected_fields=aleatoric.high_variance_fields,
            ))

        if aleatoric.unresolved_conflicts > 0:
            reasons.append(UncertaintyReason(
                category="aleatoric",
                code="UNRESOLVED_CONFLICTS",
                description=f"{aleatoric.unresolved_conflicts} unresolved claim conflicts",
                severity=min(0.9, aleatoric.conflict_severity_sum),
                affected_fields=aleatoric.conflicting_fields,
            ))

        if aleatoric.missing_fields:
            reasons.append(UncertaintyReason(
                category="aleatoric",
                code="MISSING_FIELDS",
                description=f"{len(aleatoric.missing_fields)} required fields missing",
                severity=min(0.8, len(aleatoric.missing_fields) * 0.15),
                affected_fields=aleatoric.missing_fields,
            ))

        # Sort by severity
        reasons.sort(key=lambda r: -r.severity)
        return reasons[:7]  # Top 7 reasons

    def analyze(
        self,
        context: EvalContext,
        hypothesis_set: HypothesisSet,
        conflicts: Optional[list[Any]] = None,
        deal_id: str = "",
    ) -> UncertaintyReport:
        """
        Perform complete uncertainty analysis.

        Args:
            context: Deal context
            hypothesis_set: Set of learned policies
            conflicts: Optional list of conflicts
            deal_id: Deal identifier

        Returns:
            Complete UncertaintyReport
        """
        report = UncertaintyReport(deal_id=deal_id)

        # Get best policy for rule analysis
        best = hypothesis_set.get_best()
        rules = best.rules if best else []

        # Analyze epistemic uncertainty
        report.epistemic = self.analyze_epistemic(context, hypothesis_set)

        # Analyze aleatoric uncertainty
        report.aleatoric = self.analyze_aleatoric(context, rules, conflicts)

        # Compute total uncertainty
        report.compute_total()

        # Compile reasons
        report.top_reasons = self.compile_reasons(report.epistemic, report.aleatoric)

        # Generate information requests
        report.information_requests = self.generate_information_requests(
            report.epistemic, report.aleatoric, rules
        )

        # Determine if we should defer
        report.should_defer = (
            report.epistemic.score >= self.config.defer_epistemic_threshold or
            report.aleatoric.score >= self.config.defer_aleatoric_threshold or
            report.total_uncertainty >= self.config.defer_total_threshold
        )

        if report.should_defer:
            if report.epistemic.score >= report.aleatoric.score:
                report.defer_reason = (
                    "High policy uncertainty - multiple interpretations possible. "
                    "Consider gathering more historical decisions or clarifying investment thesis."
                )
            else:
                report.defer_reason = (
                    "High evidence uncertainty - key claims have low confidence or conflicts. "
                    f"Recommended: gather information on {report.information_requests[0].field if report.information_requests else 'critical fields'}."
                )

        # Get decision from best policy
        if best:
            engine = RuleEngine(rules)
            trace = engine.evaluate(context)
            report.decision = trace.final_decision
            report.decision_confidence = trace.decision_confidence

        return report


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_uncertainty(
    context: EvalContext,
    hypothesis_set: HypothesisSet,
    conflicts: Optional[list[Any]] = None,
    deal_id: str = "",
    config: Optional[UncertaintyConfig] = None,
) -> UncertaintyReport:
    """
    Convenience function to analyze uncertainty.

    Args:
        context: Deal context
        hypothesis_set: Set of learned policies
        conflicts: Optional conflicts
        deal_id: Deal identifier
        config: Optional configuration

    Returns:
        UncertaintyReport
    """
    analyzer = UncertaintyAnalyzer(config=config)
    return analyzer.analyze(
        context=context,
        hypothesis_set=hypothesis_set,
        conflicts=conflicts,
        deal_id=deal_id,
    )


def should_request_more_info(report: UncertaintyReport) -> bool:
    """Check if decision should be deferred pending more information."""
    return report.should_defer


def get_top_information_requests(
    report: UncertaintyReport,
    max_requests: int = 5,
) -> list[dict[str, Any]]:
    """Get top information requests as dictionaries."""
    return [
        {
            "field": r.field,
            "description": r.description,
            "importance": r.importance,
            "reason": r.reason,
        }
        for r in report.information_requests[:max_requests]
    ]


def _variance(values: list[float]) -> float:
    """Compute variance of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
