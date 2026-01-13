"""
CRE (Candidate Rule Engine) search space for JURIS VC DSL.

Defines predicate templates for rule learning/search.
Thresholds come from a Threshold Proposer (not infinite search).

This module provides:
- Predicate templates with placeholder thresholds
- Template instantiation with proposed thresholds
- Search space configuration
- Integration with Threshold Proposer for bounded search
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from .predicates_v2 import (
    Predicate,
    Has,
    Eq,
    In,
    Ge,
    Le,
    Between,
    Trend,
    ConfGe,
    And,
    Or,
)
from .typing import ValueType, TrendKind
from .evaluation import Decision

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of predicate templates."""

    EXISTENCE = "existence"  # has(field)
    EQUALITY = "equality"  # eq(field, value)
    MEMBERSHIP = "membership"  # in(field, values)
    THRESHOLD_GE = "threshold_ge"  # ge(field, threshold)
    THRESHOLD_LE = "threshold_le"  # le(field, threshold)
    RANGE = "range"  # between(field, lo, hi)
    TREND = "trend"  # trend(field, window, kind)
    COMPOSITE = "composite"  # and/or combinations


@dataclass
class PredicateTemplate:
    """
    A template for generating predicates.

    Templates define the structure of a predicate but leave
    threshold values as placeholders to be filled by the
    Threshold Proposer.
    """

    template_id: str
    template_type: TemplateType
    field: str
    value_type: ValueType = ValueType.NUMERIC
    description: str = ""

    # For threshold templates
    threshold_placeholder: str = "THRESHOLD"

    # For range templates
    lo_placeholder: str = "LO"
    hi_placeholder: str = "HI"

    # For enum templates
    candidate_values: list[Any] = field(default_factory=list)

    # For trend templates
    window_options: list[int] = field(default_factory=lambda: [3, 6, 12])
    trend_kinds: list[TrendKind] = field(
        default_factory=lambda: [TrendKind.UP, TrendKind.DOWN, TrendKind.FLAT]
    )

    # Confidence gate
    min_confidence: float = 0.7

    def instantiate(self, **kwargs) -> Predicate:
        """
        Instantiate the template with specific values.

        Args:
            **kwargs: Values to fill placeholders

        Returns:
            Concrete Predicate instance
        """
        conf_gate = ConfGe(self.field, self.min_confidence)

        if self.template_type == TemplateType.EXISTENCE:
            return Has(self.field)

        elif self.template_type == TemplateType.EQUALITY:
            value = kwargs.get("value", self.candidate_values[0] if self.candidate_values else None)
            return And([conf_gate, Eq(self.field, value, self.value_type)])

        elif self.template_type == TemplateType.MEMBERSHIP:
            values = kwargs.get("values", self.candidate_values)
            return And([conf_gate, In(self.field, values, self.value_type)])

        elif self.template_type == TemplateType.THRESHOLD_GE:
            threshold = kwargs.get("threshold")
            if threshold is None:
                raise ValueError(f"Threshold required for {self.template_id}")
            return And([conf_gate, Ge(self.field, float(threshold), self.value_type)])

        elif self.template_type == TemplateType.THRESHOLD_LE:
            threshold = kwargs.get("threshold")
            if threshold is None:
                raise ValueError(f"Threshold required for {self.template_id}")
            return And([conf_gate, Le(self.field, float(threshold), self.value_type)])

        elif self.template_type == TemplateType.RANGE:
            lo = kwargs.get("lo")
            hi = kwargs.get("hi")
            if lo is None or hi is None:
                raise ValueError(f"lo and hi required for {self.template_id}")
            return And([conf_gate, Between(self.field, float(lo), float(hi), self.value_type)])

        elif self.template_type == TemplateType.TREND:
            window = kwargs.get("window", self.window_options[0])
            kind = kwargs.get("kind", self.trend_kinds[0])
            if isinstance(kind, str):
                kind = TrendKind(kind)
            return And([conf_gate, Trend(self.field, int(window), kind)])

        else:
            raise ValueError(f"Unknown template type: {self.template_type}")

    def get_placeholders(self) -> list[str]:
        """Get list of placeholder names that need values."""
        if self.template_type in [TemplateType.THRESHOLD_GE, TemplateType.THRESHOLD_LE]:
            return ["threshold"]
        elif self.template_type == TemplateType.RANGE:
            return ["lo", "hi"]
        elif self.template_type == TemplateType.EQUALITY:
            return ["value"]
        elif self.template_type == TemplateType.MEMBERSHIP:
            return ["values"]
        elif self.template_type == TemplateType.TREND:
            return ["window", "kind"]
        return []


@dataclass
class RuleTemplate:
    """
    A template for generating rules.

    Combines multiple predicate templates with a decision.
    """

    template_id: str
    name: str
    decision: Decision
    predicate_templates: list[PredicateTemplate]
    combinator: str = "and"  # "and" or "or"
    priority: int = 5
    description: str = ""

    def instantiate(self, threshold_values: dict[str, dict[str, Any]]) -> "Rule":
        """
        Instantiate the rule template with threshold values.

        Args:
            threshold_values: Map from predicate template_id to kwargs

        Returns:
            Concrete Rule instance
        """
        from .evaluation import Rule

        predicates = []
        for pt in self.predicate_templates:
            kwargs = threshold_values.get(pt.template_id, {})
            try:
                pred = pt.instantiate(**kwargs)
                predicates.append(pred)
            except ValueError:
                # Skip if required values not provided
                pass

        if not predicates:
            raise ValueError("No valid predicates after instantiation")

        if len(predicates) == 1:
            combined = predicates[0]
        elif self.combinator == "and":
            combined = And(predicates)
        else:
            combined = Or(predicates)

        return Rule(
            rule_id=f"rule_{self.template_id}",
            name=self.name,
            predicate=combined,
            decision=self.decision,
            priority=self.priority,
        )


# =============================================================================
# Standard VC Predicate Templates
# =============================================================================


# Traction templates
TRACTION_TEMPLATES = [
    PredicateTemplate(
        template_id="arr_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.arr",
        value_type=ValueType.NUMERIC,
        description="ARR >= threshold",
    ),
    PredicateTemplate(
        template_id="mrr_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.mrr",
        value_type=ValueType.NUMERIC,
        description="MRR >= threshold",
    ),
    PredicateTemplate(
        template_id="growth_rate_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.growth_rate",
        value_type=ValueType.NUMERIC,
        description="Growth rate >= threshold (as ratio)",
    ),
    PredicateTemplate(
        template_id="customers_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.customers",
        value_type=ValueType.NUMERIC,
        description="Customer count >= threshold",
    ),
    PredicateTemplate(
        template_id="nps_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.nps",
        value_type=ValueType.NUMERIC,
        description="NPS >= threshold",
    ),
    PredicateTemplate(
        template_id="retention_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="traction.retention_rate",
        value_type=ValueType.NUMERIC,
        description="Retention rate >= threshold",
    ),
]

# Team templates
TEAM_TEMPLATES = [
    PredicateTemplate(
        template_id="team_size_range",
        template_type=TemplateType.RANGE,
        field="team_composition.team_size",
        value_type=ValueType.NUMERIC,
        description="Team size in range",
    ),
    PredicateTemplate(
        template_id="founder_quality",
        template_type=TemplateType.EXISTENCE,
        field="team_quality.founder_background",
        description="Has founder background info",
    ),
    PredicateTemplate(
        template_id="repeat_founder",
        template_type=TemplateType.EQUALITY,
        field="team_quality.repeat_founder",
        value_type=ValueType.BOOLEAN,
        candidate_values=[True, False],
        description="Is repeat founder",
    ),
]

# Market templates
MARKET_TEMPLATES = [
    PredicateTemplate(
        template_id="tam_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="market_scope.tam",
        value_type=ValueType.NUMERIC,
        description="TAM >= threshold",
    ),
    PredicateTemplate(
        template_id="market_timing",
        template_type=TemplateType.EXISTENCE,
        field="market_scope.market_timing",
        description="Has market timing assessment",
    ),
]

# Financial templates
FINANCIAL_TEMPLATES = [
    PredicateTemplate(
        template_id="gross_margin_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="business_model.gross_margin",
        value_type=ValueType.NUMERIC,
        description="Gross margin >= threshold",
    ),
    PredicateTemplate(
        template_id="ltv_cac_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="business_model.ltv_cac_ratio",
        value_type=ValueType.NUMERIC,
        description="LTV/CAC >= threshold",
    ),
    PredicateTemplate(
        template_id="monthly_burn_max",
        template_type=TemplateType.THRESHOLD_LE,
        field="capital_intensity.monthly_burn",
        value_type=ValueType.NUMERIC,
        description="Monthly burn <= threshold",
    ),
    PredicateTemplate(
        template_id="runway_min",
        template_type=TemplateType.THRESHOLD_GE,
        field="capital_intensity.runway_months",
        value_type=ValueType.NUMERIC,
        description="Runway >= threshold months",
    ),
]

# Risk templates
RISK_TEMPLATES = [
    PredicateTemplate(
        template_id="key_person_risk",
        template_type=TemplateType.EXISTENCE,
        field="execution_risk.key_person",
        description="Has key person risk identified",
    ),
    PredicateTemplate(
        template_id="regulatory_status",
        template_type=TemplateType.EXISTENCE,
        field="regulatory_risk.status",
        description="Has regulatory status",
    ),
    PredicateTemplate(
        template_id="competitive_moat",
        template_type=TemplateType.EXISTENCE,
        field="differentiation.moat",
        description="Has competitive moat identified",
    ),
]

# Deal terms templates
DEAL_TEMPLATES = [
    PredicateTemplate(
        template_id="valuation_range",
        template_type=TemplateType.RANGE,
        field="round_terms.pre_money_valuation",
        value_type=ValueType.NUMERIC,
        description="Pre-money valuation in range",
    ),
    PredicateTemplate(
        template_id="stage",
        template_type=TemplateType.MEMBERSHIP,
        field="round_terms.stage",
        value_type=ValueType.ENUM,
        candidate_values=["seed", "series_a", "series_b"],
        description="Funding stage in list",
    ),
]

# Trend templates
TREND_TEMPLATES = [
    PredicateTemplate(
        template_id="arr_trend",
        template_type=TemplateType.TREND,
        field="traction.arr",
        window_options=[3, 6, 12],
        trend_kinds=[TrendKind.UP, TrendKind.ACCELERATING],
        description="ARR trend over time",
    ),
    PredicateTemplate(
        template_id="growth_trend",
        template_type=TemplateType.TREND,
        field="traction.growth_rate",
        window_options=[3, 6],
        trend_kinds=[TrendKind.UP, TrendKind.FLAT, TrendKind.DOWN],
        description="Growth rate trend",
    ),
]


# All templates
ALL_TEMPLATES = (
    TRACTION_TEMPLATES
    + TEAM_TEMPLATES
    + MARKET_TEMPLATES
    + FINANCIAL_TEMPLATES
    + RISK_TEMPLATES
    + DEAL_TEMPLATES
    + TREND_TEMPLATES
)

TEMPLATE_BY_ID = {t.template_id: t for t in ALL_TEMPLATES}


# =============================================================================
# Search Space Configuration
# =============================================================================


@dataclass
class SearchSpaceConfig:
    """Configuration for CRE search space."""

    # Which template types to include
    include_existence: bool = True
    include_thresholds: bool = True
    include_ranges: bool = True
    include_enums: bool = True
    include_trends: bool = False  # Trends require time-series data

    # Constraints
    max_predicates_per_rule: int = 5
    min_confidence: float = 0.7

    # Template filters
    include_templates: Optional[list[str]] = None  # If set, only these
    exclude_templates: Optional[list[str]] = None  # If set, exclude these

    def get_templates(self) -> list[PredicateTemplate]:
        """Get templates matching this configuration."""
        templates = []

        for t in ALL_TEMPLATES:
            # Check type filters
            if t.template_type == TemplateType.EXISTENCE and not self.include_existence:
                continue
            if t.template_type in [TemplateType.THRESHOLD_GE, TemplateType.THRESHOLD_LE] and not self.include_thresholds:
                continue
            if t.template_type == TemplateType.RANGE and not self.include_ranges:
                continue
            if t.template_type in [TemplateType.EQUALITY, TemplateType.MEMBERSHIP] and not self.include_enums:
                continue
            if t.template_type == TemplateType.TREND and not self.include_trends:
                continue

            # Check include/exclude lists
            if self.include_templates and t.template_id not in self.include_templates:
                continue
            if self.exclude_templates and t.template_id in self.exclude_templates:
                continue

            # Apply min confidence
            t_copy = PredicateTemplate(
                template_id=t.template_id,
                template_type=t.template_type,
                field=t.field,
                value_type=t.value_type,
                description=t.description,
                candidate_values=t.candidate_values,
                window_options=t.window_options,
                trend_kinds=t.trend_kinds,
                min_confidence=self.min_confidence,
            )
            templates.append(t_copy)

        return templates


def get_template_by_field(field: str) -> Optional[PredicateTemplate]:
    """Find a template that matches the given field."""
    for t in ALL_TEMPLATES:
        if t.field == field:
            return t
    return None


def get_templates_for_claim_type(claim_type: str) -> list[PredicateTemplate]:
    """Get all templates for a claim type."""
    return [t for t in ALL_TEMPLATES if t.field.startswith(f"{claim_type}.")]


# =============================================================================
# Predicate Synthesizer with Threshold Proposer Integration
# =============================================================================


@dataclass
class SynthesizerConfig:
    """Configuration for predicate synthesis."""

    # Maximum predicates to generate per template
    max_predicates_per_template: int = 12

    # Threshold proposer settings
    threshold_quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    use_domain_heuristics: bool = True
    round_thresholds: bool = True

    # Confidence settings
    min_confidence: float = 0.7

    # Which template types to synthesize
    synthesize_ge: bool = True
    synthesize_le: bool = True
    synthesize_between: bool = True


@dataclass
class SynthesisTrace:
    """Trace of predicate synthesis process."""

    template_id: str
    field: str
    threshold_count: int
    predicates_generated: int
    thresholds_used: list[float] = field(default_factory=list)


class PredicateSynthesizer:
    """
    Synthesizes predicates from templates using Threshold Proposer.

    Ensures that ge/le/between predicates only use thresholds from
    the Threshold Proposer, keeping search tractable.
    """

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        observed_data: Optional[dict[str, list[float]]] = None,
    ):
        """
        Initialize synthesizer.

        Args:
            config: Synthesis configuration
            observed_data: Map from field names to observed values
        """
        self.config = config or SynthesizerConfig()
        self.observed_data = observed_data or {}
        self._threshold_cache: dict[str, list[float]] = {}
        self._traces: list[SynthesisTrace] = []

    def _get_thresholds(self, field: str) -> list[float]:
        """Get thresholds for a field using Threshold Proposer."""
        if field in self._threshold_cache:
            return self._threshold_cache[field]

        # Import here to avoid circular imports
        from .thresholds import (
            propose_thresholds,
            ThresholdProposerConfig,
        )

        observed = self.observed_data.get(field, [])

        proposer_config = ThresholdProposerConfig(
            max_thresholds=self.config.max_predicates_per_template,
            quantiles=self.config.threshold_quantiles,
            use_domain_heuristics=self.config.use_domain_heuristics,
            round_to_nice=self.config.round_thresholds,
            min_confidence=0.5,
            enable_trace=True,
        )

        thresholds, trace = propose_thresholds(
            field=field,
            observed_values=observed,
            config=proposer_config,
        )

        if trace:
            logger.debug(
                f"Proposed {len(thresholds)} thresholds for {field}: "
                f"{thresholds[:5]}{'...' if len(thresholds) > 5 else ''}"
            )

        self._threshold_cache[field] = thresholds
        return thresholds

    def synthesize_from_template(
        self,
        template: PredicateTemplate,
    ) -> list[Predicate]:
        """
        Synthesize predicates from a single template.

        For threshold templates (ge/le/between), uses Threshold Proposer
        to generate a bounded set of candidates.

        Args:
            template: Template to synthesize from

        Returns:
            List of concrete predicates
        """
        predicates = []
        thresholds_used = []

        if template.template_type == TemplateType.EXISTENCE:
            # Existence predicates don't need thresholds
            predicates.append(template.instantiate())

        elif template.template_type == TemplateType.EQUALITY:
            # Generate one predicate per candidate value
            for value in template.candidate_values:
                predicates.append(template.instantiate(value=value))

        elif template.template_type == TemplateType.MEMBERSHIP:
            # Use all candidate values
            predicates.append(template.instantiate(values=template.candidate_values))

        elif template.template_type == TemplateType.THRESHOLD_GE and self.config.synthesize_ge:
            # Get thresholds from Threshold Proposer
            thresholds = self._get_thresholds(template.field)
            thresholds_used = thresholds
            for threshold in thresholds:
                predicates.append(template.instantiate(threshold=threshold))

        elif template.template_type == TemplateType.THRESHOLD_LE and self.config.synthesize_le:
            # Get thresholds from Threshold Proposer
            thresholds = self._get_thresholds(template.field)
            thresholds_used = thresholds
            for threshold in thresholds:
                predicates.append(template.instantiate(threshold=threshold))

        elif template.template_type == TemplateType.RANGE and self.config.synthesize_between:
            # Generate range predicates from threshold pairs
            thresholds = self._get_thresholds(template.field)
            thresholds_used = thresholds

            # Generate adjacent pairs
            for i in range(len(thresholds) - 1):
                lo = thresholds[i]
                hi = thresholds[i + 1]
                predicates.append(template.instantiate(lo=lo, hi=hi))

            # Also generate wider ranges (skip one)
            for i in range(len(thresholds) - 2):
                lo = thresholds[i]
                hi = thresholds[i + 2]
                predicates.append(template.instantiate(lo=lo, hi=hi))

        elif template.template_type == TemplateType.TREND:
            # Generate predicates for each window/kind combination
            for window in template.window_options:
                for kind in template.trend_kinds:
                    predicates.append(template.instantiate(window=window, kind=kind))

        # Record trace
        self._traces.append(SynthesisTrace(
            template_id=template.template_id,
            field=template.field,
            threshold_count=len(thresholds_used),
            predicates_generated=len(predicates),
            thresholds_used=thresholds_used,
        ))

        return predicates

    def synthesize_all(
        self,
        templates: Optional[list[PredicateTemplate]] = None,
        config: Optional[SearchSpaceConfig] = None,
    ) -> list[Predicate]:
        """
        Synthesize predicates from all templates.

        Args:
            templates: Templates to use (defaults to all matching config)
            config: Search space config for filtering templates

        Returns:
            List of all synthesized predicates
        """
        if templates is None:
            if config is None:
                config = SearchSpaceConfig()
            templates = config.get_templates()

        all_predicates = []
        for template in templates:
            predicates = self.synthesize_from_template(template)
            all_predicates.extend(predicates)

        logger.info(
            f"Synthesized {len(all_predicates)} predicates from "
            f"{len(templates)} templates"
        )

        return all_predicates

    def get_traces(self) -> list[SynthesisTrace]:
        """Get synthesis traces for debugging/logging."""
        return self._traces

    def get_trace_summary(self) -> dict:
        """Get a summary of synthesis traces."""
        return {
            "templates_processed": len(self._traces),
            "total_predicates": sum(t.predicates_generated for t in self._traces),
            "total_thresholds": sum(t.threshold_count for t in self._traces),
            "by_template": {
                t.template_id: {
                    "field": t.field,
                    "thresholds": t.threshold_count,
                    "predicates": t.predicates_generated,
                }
                for t in self._traces
            },
        }

    def clear_cache(self):
        """Clear threshold cache."""
        self._threshold_cache.clear()


def synthesize_predicates(
    observed_data: Optional[dict[str, list[float]]] = None,
    config: Optional[SynthesizerConfig] = None,
    search_config: Optional[SearchSpaceConfig] = None,
) -> tuple[list[Predicate], dict]:
    """
    Convenience function to synthesize predicates.

    Args:
        observed_data: Map from field names to observed values
        config: Synthesizer configuration
        search_config: Search space configuration

    Returns:
        Tuple of (predicates, trace_summary)
    """
    synthesizer = PredicateSynthesizer(
        config=config,
        observed_data=observed_data,
    )

    predicates = synthesizer.synthesize_all(config=search_config)
    summary = synthesizer.get_trace_summary()

    return predicates, summary
