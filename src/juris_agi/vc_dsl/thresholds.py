"""
Threshold Proposer for JURIS VC DSL.

Generates finite candidate sets for numeric thresholds to keep
program/rule search tractable. Uses:
- Quantiles over observed values
- Domain heuristics for common VC fields
- Rounding to sensible steps

Caps the number of thresholds per field (default: 12) to prevent
search space explosion.
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


class ThresholdReason(Enum):
    """Reason why a threshold was selected."""

    QUANTILE = "quantile"  # Derived from observed data quantile
    DOMAIN_HEURISTIC = "domain_heuristic"  # From domain-specific knowledge
    ROUND_NUMBER = "round_number"  # Rounded to sensible step
    BOUNDARY = "boundary"  # Min/max boundary
    PRIOR = "prior"  # From prior knowledge


@dataclass
class ThresholdCandidate:
    """A proposed threshold value with provenance."""

    value: float
    reason: ThresholdReason
    description: str = ""
    confidence: float = 1.0  # How confident we are this is a good threshold

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, ThresholdCandidate):
            return self.value == other.value
        return False


@dataclass
class ThresholdTrace:
    """Trace log for threshold proposal process."""

    field: str
    observed_count: int
    observed_min: Optional[float]
    observed_max: Optional[float]
    observed_mean: Optional[float]
    priors_used: list[str] = field(default_factory=list)
    quantiles_used: list[float] = field(default_factory=list)
    heuristics_applied: list[str] = field(default_factory=list)
    candidates_before_dedup: int = 0
    candidates_after_dedup: int = 0
    candidates_after_cap: int = 0
    final_thresholds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "field": self.field,
            "observed": {
                "count": self.observed_count,
                "min": self.observed_min,
                "max": self.observed_max,
                "mean": self.observed_mean,
            },
            "priors_used": self.priors_used,
            "quantiles_used": self.quantiles_used,
            "heuristics_applied": self.heuristics_applied,
            "candidate_counts": {
                "before_dedup": self.candidates_before_dedup,
                "after_dedup": self.candidates_after_dedup,
                "after_cap": self.candidates_after_cap,
            },
            "final_thresholds": self.final_thresholds,
        }


# =============================================================================
# Domain Heuristics
# =============================================================================

# Domain-specific threshold heuristics for common VC fields
# Maps field patterns to lists of (threshold, description) tuples
DOMAIN_HEURISTICS: dict[str, list[tuple[float, str]]] = {
    # Gross margin thresholds (as ratio 0-1 or percentage 0-100)
    "gross_margin": [
        (0.40, "40% - minimum for software"),
        (0.60, "60% - good software margin"),
        (0.70, "70% - strong SaaS margin"),
        (0.80, "80% - excellent margin"),
    ],
    # Growth rate thresholds (as ratio or percentage)
    "growth_rate": [
        (0.50, "50% - moderate growth"),
        (1.00, "100% - doubling YoY"),
        (2.00, "200% - triple-digit growth"),
        (3.00, "300% - hypergrowth"),
    ],
    # Runway in months
    "runway_months": [
        (6, "6 months - danger zone"),
        (12, "12 months - minimum comfortable"),
        (18, "18 months - standard target"),
        (24, "24 months - well-capitalized"),
    ],
    "runway": [
        (6, "6 months - danger zone"),
        (12, "12 months - minimum comfortable"),
        (18, "18 months - standard target"),
        (24, "24 months - well-capitalized"),
    ],
    # LTV/CAC ratio
    "ltv_cac": [
        (1.0, "1:1 - break-even"),
        (3.0, "3:1 - healthy ratio"),
        (5.0, "5:1 - strong unit economics"),
    ],
    # NPS score (-100 to 100)
    "nps": [
        (0, "0 - neutral"),
        (30, "30 - good"),
        (50, "50 - excellent"),
        (70, "70 - world-class"),
    ],
    # Retention rate (as ratio 0-1)
    "retention": [
        (0.80, "80% - concerning"),
        (0.90, "90% - acceptable"),
        (0.95, "95% - good"),
        (0.98, "98% - excellent"),
    ],
    # Net revenue retention (as ratio, often > 1)
    "nrr": [
        (0.90, "90% - net contraction"),
        (1.00, "100% - flat"),
        (1.10, "110% - healthy expansion"),
        (1.30, "130% - strong expansion"),
    ],
    # Customer count
    "customers": [
        (10, "10 - early traction"),
        (50, "50 - product-market fit signal"),
        (100, "100 - proven demand"),
        (500, "500 - scaling"),
        (1000, "1000 - scale achieved"),
    ],
    # ARR thresholds (in dollars)
    "arr": [
        (100_000, "$100K - early revenue"),
        (500_000, "$500K - seed stage"),
        (1_000_000, "$1M - Series A threshold"),
        (5_000_000, "$5M - Series B threshold"),
        (10_000_000, "$10M - growth stage"),
    ],
    # MRR thresholds (in dollars)
    "mrr": [
        (10_000, "$10K - early revenue"),
        (50_000, "$50K - traction"),
        (100_000, "$100K - scaling"),
        (500_000, "$500K - significant"),
    ],
    # TAM/SAM/SOM (in dollars, typically billions)
    "tam": [
        (1_000_000_000, "$1B - minimum for VC"),
        (5_000_000_000, "$5B - attractive"),
        (10_000_000_000, "$10B - large market"),
        (50_000_000_000, "$50B - massive market"),
    ],
    # Monthly burn (in dollars)
    "monthly_burn": [
        (50_000, "$50K - lean"),
        (100_000, "$100K - moderate"),
        (250_000, "$250K - growth mode"),
        (500_000, "$500K - aggressive"),
    ],
    "burn": [
        (50_000, "$50K - lean"),
        (100_000, "$100K - moderate"),
        (250_000, "$250K - growth mode"),
        (500_000, "$500K - aggressive"),
    ],
    # Valuation (pre-money)
    "valuation": [
        (5_000_000, "$5M - early seed"),
        (10_000_000, "$10M - seed"),
        (25_000_000, "$25M - Series A"),
        (50_000_000, "$50M - Series B"),
        (100_000_000, "$100M - growth"),
    ],
    # Team size
    "team_size": [
        (5, "5 - founding team"),
        (10, "10 - early team"),
        (25, "25 - scaling"),
        (50, "50 - significant org"),
        (100, "100 - large team"),
    ],
}


def _get_domain_heuristics(field: str) -> list[tuple[float, str]]:
    """Get domain heuristics for a field by matching patterns."""
    field_lower = field.lower()

    # Direct match
    if field_lower in DOMAIN_HEURISTICS:
        return DOMAIN_HEURISTICS[field_lower]

    # Match field suffix (e.g., "traction.arr" matches "arr")
    for key, values in DOMAIN_HEURISTICS.items():
        if field_lower.endswith(f".{key}") or field_lower.endswith(f"_{key}"):
            return values
        if key in field_lower:
            return values

    return []


# =============================================================================
# Rounding Logic
# =============================================================================


def _round_to_nice_number(value: float, significant_digits: int = 2) -> float:
    """
    Round a value to a 'nice' number for human readability.

    Nice numbers are: 1, 2, 2.5, 5, 10 (and their powers of 10).
    """
    if value == 0:
        return 0.0

    # Handle negative values
    sign = 1 if value >= 0 else -1
    value = abs(value)

    # Get the order of magnitude
    magnitude = 10 ** math.floor(math.log10(value))

    # Normalize to 1-10 range
    normalized = value / magnitude

    # Round to nice number
    nice_numbers = [1.0, 2.0, 2.5, 5.0, 10.0]
    closest = min(nice_numbers, key=lambda x: abs(x - normalized))

    return sign * closest * magnitude


def _round_to_step(value: float, step: float) -> float:
    """Round value to the nearest step."""
    if step == 0:
        return value
    return round(value / step) * step


def _infer_step_size(values: Sequence[float]) -> float:
    """Infer a sensible step size from observed values."""
    if len(values) < 2:
        return 1.0

    # Get the range
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return 1.0

    # Infer step based on magnitude
    magnitude = 10 ** math.floor(math.log10(range_val))

    # Choose step to give reasonable granularity (5-20 steps in range)
    if range_val / magnitude < 2:
        return magnitude / 10
    elif range_val / magnitude < 5:
        return magnitude / 5
    else:
        return magnitude / 2


# =============================================================================
# Quantile Computation
# =============================================================================


def _compute_quantiles(
    values: Sequence[float],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> list[float]:
    """
    Compute quantiles from observed values.

    Args:
        values: Observed numeric values
        quantiles: Which quantiles to compute (0-1)

    Returns:
        List of quantile values
    """
    if len(values) == 0:
        return []

    sorted_values = sorted(values)
    n = len(sorted_values)

    results = []
    for q in quantiles:
        # Linear interpolation
        idx = q * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))

        if lower == upper:
            results.append(sorted_values[lower])
        else:
            frac = idx - lower
            val = sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
            results.append(val)

    return results


# =============================================================================
# Prior Knowledge
# =============================================================================


@dataclass
class ThresholdPrior:
    """Prior knowledge about thresholds for a field."""

    field: str
    suggested_thresholds: list[float]
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None
    description: str = ""


# =============================================================================
# Main Threshold Proposer
# =============================================================================


@dataclass
class ThresholdProposerConfig:
    """Configuration for threshold proposal."""

    # Maximum thresholds per field (to bound search space)
    max_thresholds: int = 12

    # Quantiles to compute from observed data
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    # Whether to use domain heuristics
    use_domain_heuristics: bool = True

    # Whether to round to nice numbers
    round_to_nice: bool = True

    # Whether to include min/max boundaries
    include_boundaries: bool = True

    # Minimum confidence to include a threshold
    min_confidence: float = 0.5

    # Enable trace logging
    enable_trace: bool = True


def propose_thresholds(
    field: str,
    observed_values: Sequence[float],
    priors: Optional[list[ThresholdPrior]] = None,
    config: Optional[ThresholdProposerConfig] = None,
) -> tuple[list[float], Optional[ThresholdTrace]]:
    """
    Propose candidate thresholds for a numeric field.

    Uses a combination of:
    - Quantiles over observed values
    - Domain heuristics for common VC fields
    - Prior knowledge if provided
    - Rounding to sensible steps

    Args:
        field: Field name (e.g., "traction.arr", "business_model.gross_margin")
        observed_values: List of observed numeric values for this field
        priors: Optional prior knowledge about thresholds
        config: Configuration options

    Returns:
        Tuple of (list of threshold values, optional trace)
    """
    config = config or ThresholdProposerConfig()
    candidates: list[ThresholdCandidate] = []

    # Initialize trace
    trace = ThresholdTrace(
        field=field,
        observed_count=len(observed_values),
        observed_min=min(observed_values) if observed_values else None,
        observed_max=max(observed_values) if observed_values else None,
        observed_mean=statistics.mean(observed_values) if observed_values else None,
    ) if config.enable_trace else None

    # 1. Add quantiles from observed values
    if observed_values:
        quantile_values = _compute_quantiles(observed_values, config.quantiles)
        for q, v in zip(config.quantiles, quantile_values):
            candidates.append(ThresholdCandidate(
                value=v,
                reason=ThresholdReason.QUANTILE,
                description=f"P{int(q*100)} of observed values",
                confidence=0.9,
            ))
            if trace:
                trace.quantiles_used.append(q)

    # 2. Add domain heuristics
    if config.use_domain_heuristics:
        heuristics = _get_domain_heuristics(field)
        for threshold, desc in heuristics:
            candidates.append(ThresholdCandidate(
                value=threshold,
                reason=ThresholdReason.DOMAIN_HEURISTIC,
                description=desc,
                confidence=0.8,
            ))
            if trace:
                trace.heuristics_applied.append(desc)

    # 3. Add prior knowledge
    if priors:
        for prior in priors:
            if prior.field == field or field.endswith(f".{prior.field}"):
                for threshold in prior.suggested_thresholds:
                    candidates.append(ThresholdCandidate(
                        value=threshold,
                        reason=ThresholdReason.PRIOR,
                        description=prior.description,
                        confidence=0.95,
                    ))
                    if trace:
                        trace.priors_used.append(f"{threshold} ({prior.description})")

    # 4. Add boundaries from observed data
    if config.include_boundaries and observed_values:
        min_val = min(observed_values)
        max_val = max(observed_values)
        candidates.append(ThresholdCandidate(
            value=min_val,
            reason=ThresholdReason.BOUNDARY,
            description="Observed minimum",
            confidence=0.7,
        ))
        candidates.append(ThresholdCandidate(
            value=max_val,
            reason=ThresholdReason.BOUNDARY,
            description="Observed maximum",
            confidence=0.7,
        ))

    if trace:
        trace.candidates_before_dedup = len(candidates)

    # 5. Round to nice numbers if enabled
    if config.round_to_nice:
        rounded_candidates = []
        for c in candidates:
            rounded_value = _round_to_nice_number(c.value)
            rounded_candidates.append(ThresholdCandidate(
                value=rounded_value,
                reason=c.reason,
                description=c.description,
                confidence=c.confidence,
            ))
        candidates = rounded_candidates

    # 6. Deduplicate by value (keep highest confidence)
    value_to_candidate: dict[float, ThresholdCandidate] = {}
    for c in candidates:
        if c.confidence < config.min_confidence:
            continue
        if c.value not in value_to_candidate:
            value_to_candidate[c.value] = c
        elif c.confidence > value_to_candidate[c.value].confidence:
            value_to_candidate[c.value] = c

    unique_candidates = list(value_to_candidate.values())

    if trace:
        trace.candidates_after_dedup = len(unique_candidates)

    # 7. Sort by value
    unique_candidates.sort(key=lambda c: c.value)

    # 8. Cap to max thresholds (keep highest confidence)
    if len(unique_candidates) > config.max_thresholds:
        # Sort by confidence descending, take top N, then re-sort by value
        by_confidence = sorted(unique_candidates, key=lambda c: -c.confidence)
        capped = by_confidence[:config.max_thresholds]
        unique_candidates = sorted(capped, key=lambda c: c.value)

    if trace:
        trace.candidates_after_cap = len(unique_candidates)
        trace.final_thresholds = [c.value for c in unique_candidates]

    # Log trace
    if trace:
        logger.debug(
            f"Threshold proposal for {field}: "
            f"{trace.candidates_before_dedup} -> {trace.candidates_after_dedup} -> "
            f"{trace.candidates_after_cap} candidates"
        )

    final_thresholds = [c.value for c in unique_candidates]
    return final_thresholds, trace


def propose_thresholds_for_template(
    template_id: str,
    observed_values: Sequence[float],
    priors: Optional[list[ThresholdPrior]] = None,
    config: Optional[ThresholdProposerConfig] = None,
) -> tuple[list[float], Optional[ThresholdTrace]]:
    """
    Propose thresholds for a predicate template.

    Looks up the template by ID and proposes thresholds for its field.

    Args:
        template_id: Template ID (e.g., "arr_min", "gross_margin_min")
        observed_values: Observed values for the template's field
        priors: Optional prior knowledge
        config: Configuration options

    Returns:
        Tuple of (list of threshold values, optional trace)
    """
    from .search_space import TEMPLATE_BY_ID

    template = TEMPLATE_BY_ID.get(template_id)
    if not template:
        logger.warning(f"Unknown template ID: {template_id}")
        return [], None

    return propose_thresholds(
        field=template.field,
        observed_values=observed_values,
        priors=priors,
        config=config,
    )


# =============================================================================
# Batch Processing
# =============================================================================


@dataclass
class FieldObservations:
    """Observations for a single field."""

    field: str
    values: list[float]


def propose_all_thresholds(
    observations: list[FieldObservations],
    priors: Optional[list[ThresholdPrior]] = None,
    config: Optional[ThresholdProposerConfig] = None,
) -> dict[str, tuple[list[float], Optional[ThresholdTrace]]]:
    """
    Propose thresholds for multiple fields at once.

    Args:
        observations: List of field observations
        priors: Optional prior knowledge
        config: Configuration options

    Returns:
        Dictionary mapping field names to (thresholds, trace) tuples
    """
    results = {}
    for obs in observations:
        thresholds, trace = propose_thresholds(
            field=obs.field,
            observed_values=obs.values,
            priors=priors,
            config=config,
        )
        results[obs.field] = (thresholds, trace)
    return results


# =============================================================================
# Convenience Functions
# =============================================================================


def get_thresholds_only(
    field: str,
    observed_values: Sequence[float],
    priors: Optional[list[ThresholdPrior]] = None,
    config: Optional[ThresholdProposerConfig] = None,
) -> list[float]:
    """Convenience function that returns only the threshold values."""
    thresholds, _ = propose_thresholds(field, observed_values, priors, config)
    return thresholds


def default_thresholds_for_field(field: str) -> list[float]:
    """
    Get default thresholds for a field using only domain heuristics.

    Useful when no observed data is available.
    """
    config = ThresholdProposerConfig(
        use_domain_heuristics=True,
        include_boundaries=False,
        enable_trace=False,
    )
    thresholds, _ = propose_thresholds(field, [], config=config)
    return thresholds


def merge_threshold_traces(traces: list[ThresholdTrace]) -> dict:
    """Merge multiple threshold traces into a summary."""
    return {
        "field_count": len(traces),
        "total_thresholds": sum(t.candidates_after_cap for t in traces),
        "fields": {t.field: t.to_dict() for t in traces},
    }
