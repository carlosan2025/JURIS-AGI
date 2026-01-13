"""
Rule evaluation engine for JURIS VC DSL.

Handles:
- Safe evaluation with missing fields
- Confidence gates
- Three-valued logic (True, False, Unknown)
- Rule application and trace generation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from juris_agi.evidence_client.types import Claim

from .predicates_v2 import (
    EvalContext,
    EvalResult,
    FieldValue,
    Predicate,
)
from .typing import ValueType, infer_value_type, get_field_unit


class Decision(Enum):
    """Investment decision outcomes."""

    INVEST = "invest"
    PASS = "pass"
    DEFER = "defer"


@dataclass
class RuleOutcome:
    """Outcome of evaluating a single rule."""

    rule_id: str
    rule_name: str
    result: EvalResult
    decision: Optional[Decision] = None
    priority: int = 0
    fields_used: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    fields_low_confidence: list[str] = field(default_factory=list)
    explanation: Optional[str] = None


@dataclass
class EvaluationTrace:
    """Trace of rule evaluation for explainability."""

    rule_outcomes: list[RuleOutcome] = field(default_factory=list)
    final_decision: Optional[Decision] = None
    decision_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_stats: dict[str, Any] = field(default_factory=dict)

    def add_outcome(self, outcome: RuleOutcome) -> None:
        self.rule_outcomes.append(outcome)

    @property
    def rules_fired(self) -> list[RuleOutcome]:
        """Get rules that evaluated to TRUE."""
        return [r for r in self.rule_outcomes if r.result == EvalResult.TRUE]

    @property
    def rules_unknown(self) -> list[RuleOutcome]:
        """Get rules that evaluated to UNKNOWN."""
        return [r for r in self.rule_outcomes if r.result == EvalResult.UNKNOWN]


@dataclass
class Rule:
    """A decision rule with predicate and outcome."""

    rule_id: str
    name: str
    predicate: Predicate
    decision: Decision
    priority: int = 0  # Higher priority rules are evaluated first
    requires_all_fields: bool = False  # If True, UNKNOWN -> FALSE
    min_confidence: float = 0.0  # Minimum confidence for fields

    def evaluate(self, ctx: EvalContext) -> RuleOutcome:
        """Evaluate the rule against a context."""
        # Check field availability
        fields = self.predicate.get_fields()
        fields_missing = [f for f in fields if not ctx.has_field(f)]
        fields_low_confidence = [
            f for f in fields
            if ctx.has_field(f) and ctx.get_field(f).confidence < self.min_confidence
        ]

        # Evaluate predicate
        result = self.predicate.evaluate(ctx)

        # Handle requires_all_fields
        if self.requires_all_fields and result == EvalResult.UNKNOWN:
            result = EvalResult.FALSE

        return RuleOutcome(
            rule_id=self.rule_id,
            rule_name=self.name,
            result=result,
            decision=self.decision if result == EvalResult.TRUE else None,
            priority=self.priority,
            fields_used=fields,
            fields_missing=fields_missing,
            fields_low_confidence=fields_low_confidence,
            explanation=self._generate_explanation(result, fields_missing, fields_low_confidence),
        )

    def _generate_explanation(
        self,
        result: EvalResult,
        missing: list[str],
        low_conf: list[str],
    ) -> str:
        """Generate human-readable explanation."""
        if result == EvalResult.TRUE:
            return f"Rule '{self.name}' fired: {self.decision.value.upper()}"
        elif result == EvalResult.UNKNOWN:
            reasons = []
            if missing:
                reasons.append(f"missing fields: {', '.join(missing)}")
            if low_conf:
                reasons.append(f"low confidence: {', '.join(low_conf)}")
            return f"Rule '{self.name}' is UNKNOWN due to {'; '.join(reasons)}"
        else:
            return f"Rule '{self.name}' did not match"


class RuleEngine:
    """
    Engine for evaluating decision rules.

    Evaluates rules in priority order, stopping at first match.
    Supports conflict resolution and trace generation.
    """

    def __init__(
        self,
        rules: list[Rule],
        default_decision: Decision = Decision.DEFER,
        unknown_handling: str = "defer",  # "defer", "pass", or "continue"
    ):
        """
        Initialize the rule engine.

        Args:
            rules: List of rules to evaluate
            default_decision: Decision when no rules match
            unknown_handling: How to handle UNKNOWN results
                - "defer": Return DEFER if any critical rule is UNKNOWN
                - "pass": Treat UNKNOWN as FALSE for matching
                - "continue": Skip UNKNOWN rules, continue evaluation
        """
        self.rules = sorted(rules, key=lambda r: -r.priority)  # High priority first
        self.default_decision = default_decision
        self.unknown_handling = unknown_handling

    def evaluate(self, ctx: EvalContext) -> EvaluationTrace:
        """
        Evaluate all rules against the context.

        Returns trace with all outcomes and final decision.
        """
        trace = EvaluationTrace()
        trace.context_stats = {
            "total_fields": len(ctx.fields),
            "fields_with_values": sum(1 for f in ctx.fields.values() if f.exists),
        }

        invest_outcomes: list[RuleOutcome] = []
        pass_outcomes: list[RuleOutcome] = []
        unknown_outcomes: list[RuleOutcome] = []

        # Evaluate all rules
        for rule in self.rules:
            outcome = rule.evaluate(ctx)
            trace.add_outcome(outcome)

            if outcome.result == EvalResult.TRUE:
                if outcome.decision == Decision.INVEST:
                    invest_outcomes.append(outcome)
                elif outcome.decision == Decision.PASS:
                    pass_outcomes.append(outcome)
            elif outcome.result == EvalResult.UNKNOWN:
                unknown_outcomes.append(outcome)

        # Determine final decision
        trace.final_decision, trace.decision_confidence = self._resolve_decision(
            invest_outcomes, pass_outcomes, unknown_outcomes
        )

        return trace

    def _resolve_decision(
        self,
        invest_outcomes: list[RuleOutcome],
        pass_outcomes: list[RuleOutcome],
        unknown_outcomes: list[RuleOutcome],
    ) -> tuple[Decision, float]:
        """Resolve conflicting outcomes to a final decision."""
        # PASS rules have veto power (any PASS -> PASS)
        if pass_outcomes:
            highest_priority = max(o.priority for o in pass_outcomes)
            return Decision.PASS, 0.8 + 0.1 * (highest_priority / 10)

        # Handle UNKNOWN outcomes
        if unknown_outcomes and self.unknown_handling == "defer":
            # Check if any high-priority rule is unknown
            high_priority_unknown = any(o.priority >= 5 for o in unknown_outcomes)
            if high_priority_unknown:
                return Decision.DEFER, 0.5

        # Check INVEST outcomes
        if invest_outcomes:
            highest_priority = max(o.priority for o in invest_outcomes)
            # Confidence based on number of supporting rules
            confidence = min(0.9, 0.6 + 0.1 * len(invest_outcomes))
            return Decision.INVEST, confidence

        # Default
        return self.default_decision, 0.4


def build_context_from_claims(
    claims: list[Claim],
    default_confidence_threshold: float = 0.0,
) -> EvalContext:
    """
    Build an evaluation context from a list of claims.

    Converts claims to FieldValues keyed by "claim_type.field".

    Args:
        claims: List of claims
        default_confidence_threshold: Default confidence threshold

    Returns:
        EvalContext ready for rule evaluation
    """
    fields: dict[str, FieldValue] = {}

    for claim in claims:
        # Create field key
        key = f"{claim.claim_type}.{claim.field}"

        # Get source type from citations if available
        source_type = None
        if claim.citations:
            source_type = claim.citations[0].document_type

        # Create field value
        fv = FieldValue(
            value=claim.value,
            confidence=claim.confidence,
            source_type=source_type,
            as_of_date=claim.as_of_date.isoformat() if claim.as_of_date else None,
            exists=True,
        )

        # If multiple claims for same field, keep highest confidence
        if key in fields:
            if fv.confidence > fields[key].confidence:
                fields[key] = fv
        else:
            fields[key] = fv

    return EvalContext(
        fields=fields,
        default_confidence_threshold=default_confidence_threshold,
    )


def build_context_from_dict(
    data: dict[str, Any],
    confidence: float = 1.0,
) -> EvalContext:
    """
    Build an evaluation context from a flat dictionary.

    Useful for testing and simple cases.

    Args:
        data: Dictionary of field -> value
        confidence: Default confidence for all fields

    Returns:
        EvalContext
    """
    fields: dict[str, FieldValue] = {}

    for key, value in data.items():
        fields[key] = FieldValue(
            value=value,
            confidence=confidence,
            exists=True,
        )

    return EvalContext(fields=fields)


# =============================================================================
# Standard Rule Templates
# =============================================================================


def create_threshold_rule(
    rule_id: str,
    name: str,
    field: str,
    operator: str,  # "ge", "le", "between"
    threshold: float,
    threshold_hi: Optional[float] = None,
    decision: Decision = Decision.INVEST,
    priority: int = 5,
    min_confidence: float = 0.7,
) -> Rule:
    """
    Create a threshold-based rule.

    Args:
        rule_id: Unique rule identifier
        name: Human-readable rule name
        field: Field to check (e.g., "traction.arr")
        operator: Comparison operator
        threshold: Threshold value (or low bound for between)
        threshold_hi: High bound for between operator
        decision: Decision when rule matches
        priority: Rule priority
        min_confidence: Minimum confidence required

    Returns:
        Configured Rule
    """
    from .predicates_v2 import Ge, Le, Between, ConfGe, And

    # Build predicate
    if operator == "ge":
        value_pred = Ge(field, threshold)
    elif operator == "le":
        value_pred = Le(field, threshold)
    elif operator == "between":
        if threshold_hi is None:
            raise ValueError("between operator requires threshold_hi")
        value_pred = Between(field, threshold, threshold_hi)
    else:
        raise ValueError(f"Unknown operator: {operator}")

    # Add confidence gate
    predicate = And([ConfGe(field, min_confidence), value_pred])

    return Rule(
        rule_id=rule_id,
        name=name,
        predicate=predicate,
        decision=decision,
        priority=priority,
        min_confidence=min_confidence,
    )


def create_enum_rule(
    rule_id: str,
    name: str,
    field: str,
    values: list[str],
    decision: Decision = Decision.INVEST,
    priority: int = 5,
    min_confidence: float = 0.7,
) -> Rule:
    """Create a rule that matches enum field values."""
    from .predicates_v2 import In, ConfGe, And

    predicate = And([ConfGe(field, min_confidence), In(field, values)])

    return Rule(
        rule_id=rule_id,
        name=name,
        predicate=predicate,
        decision=decision,
        priority=priority,
        min_confidence=min_confidence,
    )


def create_existence_rule(
    rule_id: str,
    name: str,
    required_fields: list[str],
    decision: Decision = Decision.DEFER,
    priority: int = 8,
) -> Rule:
    """Create a rule that requires certain fields to exist."""
    from .predicates_v2 import Has, And, Not

    # All fields must exist
    predicates = [Not(Has(f)) for f in required_fields]

    return Rule(
        rule_id=rule_id,
        name=name,
        predicate=And(predicates) if len(predicates) > 1 else predicates[0],
        decision=decision,
        priority=priority,
        requires_all_fields=True,
    )
