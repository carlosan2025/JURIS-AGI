"""
Predicate types for JURIS VC decision-rule DSL.

Predicates are the atomic building blocks of decision rules.
Each predicate evaluates to True, False, or Unknown.

Supported predicates:
- has(field): Check if field exists
- eq(field, value): Field equals value
- in_(field, values): Field in list of values
- ge(field, number): Field >= number
- le(field, number): Field <= number
- between(field, lo, hi): lo <= field <= hi
- trend(field, window, kind): Trend analysis over time window
- conf_ge(field, c): Confidence gate (only allow if confidence >= c)
- source_in(field, sources): Field comes from approved sources
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from .typing import ValueType, normalize_value, TrendKind


class EvalResult(Enum):
    """Three-valued logic result."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:
        """Convert to boolean (UNKNOWN -> False)."""
        return self == EvalResult.TRUE

    def __and__(self, other: "EvalResult") -> "EvalResult":
        """Kleene AND: UNKNOWN propagates unless FALSE present."""
        if self == EvalResult.FALSE or other == EvalResult.FALSE:
            return EvalResult.FALSE
        if self == EvalResult.UNKNOWN or other == EvalResult.UNKNOWN:
            return EvalResult.UNKNOWN
        return EvalResult.TRUE

    def __or__(self, other: "EvalResult") -> "EvalResult":
        """Kleene OR: UNKNOWN propagates unless TRUE present."""
        if self == EvalResult.TRUE or other == EvalResult.TRUE:
            return EvalResult.TRUE
        if self == EvalResult.UNKNOWN or other == EvalResult.UNKNOWN:
            return EvalResult.UNKNOWN
        return EvalResult.FALSE

    def __invert__(self) -> "EvalResult":
        """NOT: UNKNOWN stays UNKNOWN."""
        if self == EvalResult.TRUE:
            return EvalResult.FALSE
        if self == EvalResult.FALSE:
            return EvalResult.TRUE
        return EvalResult.UNKNOWN


@dataclass
class FieldValue:
    """A field value with metadata from the evidence context."""

    value: Any
    confidence: float = 1.0
    source_type: Optional[str] = None
    as_of_date: Optional[str] = None
    exists: bool = True

    @classmethod
    def missing(cls) -> "FieldValue":
        """Create a missing field value."""
        return cls(value=None, confidence=0.0, exists=False)


@dataclass
class EvalContext:
    """Context for predicate evaluation."""

    fields: dict[str, FieldValue] = field(default_factory=dict)
    default_confidence_threshold: float = 0.0

    def get_field(self, field_name: str) -> FieldValue:
        """Get a field value, returning missing if not found."""
        return self.fields.get(field_name, FieldValue.missing())

    def has_field(self, field_name: str) -> bool:
        """Check if field exists in context."""
        fv = self.fields.get(field_name)
        return fv is not None and fv.exists


class Predicate(ABC):
    """Base class for all predicates."""

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> EvalResult:
        """Evaluate the predicate against a context."""
        pass

    @abstractmethod
    def to_dsl(self) -> str:
        """Convert to DSL string representation."""
        pass

    @abstractmethod
    def get_fields(self) -> list[str]:
        """Get list of fields referenced by this predicate."""
        pass

    def __repr__(self) -> str:
        return self.to_dsl()


# =============================================================================
# Basic Predicates
# =============================================================================


@dataclass
class Has(Predicate):
    """Check if a field exists in the evidence."""

    field: str

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        if ctx.has_field(self.field):
            return EvalResult.TRUE
        return EvalResult.FALSE

    def to_dsl(self) -> str:
        return f"has({self.field})"

    def get_fields(self) -> list[str]:
        return [self.field]


@dataclass
class Eq(Predicate):
    """Check if field equals a specific value."""

    field: str
    value: Any
    value_type: ValueType = ValueType.STRING

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        # Normalize both values for comparison
        normalized_field = normalize_value(fv.value, self.value_type)
        normalized_target = normalize_value(self.value, self.value_type)

        if normalized_field == normalized_target:
            return EvalResult.TRUE
        return EvalResult.FALSE

    def to_dsl(self) -> str:
        if isinstance(self.value, str):
            return f'eq({self.field}, "{self.value}")'
        return f"eq({self.field}, {self.value})"

    def get_fields(self) -> list[str]:
        return [self.field]


@dataclass
class In(Predicate):
    """Check if field value is in a list of allowed values."""

    field: str
    values: list[Any]
    value_type: ValueType = ValueType.STRING

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        normalized_field = normalize_value(fv.value, self.value_type)
        normalized_values = [normalize_value(v, self.value_type) for v in self.values]

        if normalized_field in normalized_values:
            return EvalResult.TRUE
        return EvalResult.FALSE

    def to_dsl(self) -> str:
        values_str = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v) for v in self.values
        )
        return f"in({self.field}, [{values_str}])"

    def get_fields(self) -> list[str]:
        return [self.field]


# =============================================================================
# Numeric Predicates
# =============================================================================


@dataclass
class Ge(Predicate):
    """Check if field >= threshold."""

    field: str
    threshold: float
    value_type: ValueType = ValueType.NUMERIC

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        try:
            normalized = normalize_value(fv.value, self.value_type)
            if normalized is None:
                return EvalResult.UNKNOWN
            if normalized >= self.threshold:
                return EvalResult.TRUE
            return EvalResult.FALSE
        except (TypeError, ValueError):
            return EvalResult.UNKNOWN

    def to_dsl(self) -> str:
        # Format threshold as int if it's a whole number
        thresh_str = str(int(self.threshold)) if self.threshold == int(self.threshold) else str(self.threshold)
        return f"ge({self.field}, {thresh_str})"

    def get_fields(self) -> list[str]:
        return [self.field]


@dataclass
class Le(Predicate):
    """Check if field <= threshold."""

    field: str
    threshold: float
    value_type: ValueType = ValueType.NUMERIC

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        try:
            normalized = normalize_value(fv.value, self.value_type)
            if normalized is None:
                return EvalResult.UNKNOWN
            if normalized <= self.threshold:
                return EvalResult.TRUE
            return EvalResult.FALSE
        except (TypeError, ValueError):
            return EvalResult.UNKNOWN

    def to_dsl(self) -> str:
        thresh_str = str(int(self.threshold)) if self.threshold == int(self.threshold) else str(self.threshold)
        return f"le({self.field}, {thresh_str})"

    def get_fields(self) -> list[str]:
        return [self.field]


@dataclass
class Between(Predicate):
    """Check if lo <= field <= hi."""

    field: str
    lo: float
    hi: float
    value_type: ValueType = ValueType.NUMERIC
    inclusive: bool = True  # Both bounds inclusive

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        try:
            normalized = normalize_value(fv.value, self.value_type)
            if normalized is None:
                return EvalResult.UNKNOWN

            if self.inclusive:
                if self.lo <= normalized <= self.hi:
                    return EvalResult.TRUE
            else:
                if self.lo < normalized < self.hi:
                    return EvalResult.TRUE
            return EvalResult.FALSE
        except (TypeError, ValueError):
            return EvalResult.UNKNOWN

    def to_dsl(self) -> str:
        lo_str = str(int(self.lo)) if self.lo == int(self.lo) else str(self.lo)
        hi_str = str(int(self.hi)) if self.hi == int(self.hi) else str(self.hi)
        return f"between({self.field}, {lo_str}, {hi_str})"

    def get_fields(self) -> list[str]:
        return [self.field]


# =============================================================================
# Trend Predicates
# =============================================================================


@dataclass
class Trend(Predicate):
    """
    Check trend of a field over a time window.

    Supports multiple data sources:
    1. Pre-computed trend stored in context as {field}_trend_{window}
    2. Time series data in context as {field}_timeseries (list of {t, value} dicts)
    3. Historical values as {field}_t0, {field}_t1, etc.

    Uses robust trend classification from timeseries module.
    """

    field: str
    window: int  # Number of periods
    kind: TrendKind

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        # 1. Look for pre-computed trend data in context
        trend_field = f"{self.field}_trend_{self.window}"
        fv = ctx.get_field(trend_field)

        if fv.exists:
            # Check if stored trend matches expected kind
            if fv.value == self.kind.value:
                return EvalResult.TRUE
            return EvalResult.FALSE

        # 2. Look for time series data
        ts_field = f"{self.field}_timeseries"
        ts_fv = ctx.get_field(ts_field)

        if ts_fv.exists and isinstance(ts_fv.value, list):
            return self._compute_from_timeseries(ts_fv.value)

        # 3. Try to compute from historical values (legacy format)
        return self._compute_from_historical(ctx)

    def _compute_from_timeseries(self, ts_data: list) -> EvalResult:
        """Compute trend from time series data using robust analysis."""
        try:
            from .timeseries import TimeSeries, classify_trend

            # Convert to TimeSeries
            series = TimeSeries.from_list(
                field=self.field,
                data=ts_data,
                time_key="t",
                value_key="value",
            )

            if len(series.points) < 2:
                return EvalResult.UNKNOWN

            # Classify trend
            result = classify_trend(series, window=self.window)

            # Check if classified trend matches expected kind
            if result.kind == self.kind:
                return EvalResult.TRUE
            return EvalResult.FALSE

        except (ImportError, Exception):
            # Fallback to simple analysis
            return self._simple_trend_from_list(ts_data)

    def _simple_trend_from_list(self, ts_data: list) -> EvalResult:
        """Simple trend analysis fallback."""
        values = []
        for item in ts_data:
            if isinstance(item, dict) and "value" in item:
                try:
                    values.append(float(item["value"]))
                except (TypeError, ValueError):
                    pass

        return self._classify_values(values)

    def _compute_from_historical(self, ctx: EvalContext) -> EvalResult:
        """Compute trend from historical field values (legacy format)."""
        values = []
        for i in range(self.window + 1):
            hist_field = f"{self.field}_t{i}"
            fv = ctx.get_field(hist_field)
            if fv.exists and fv.value is not None:
                try:
                    values.append(float(fv.value))
                except (TypeError, ValueError):
                    pass

        return self._classify_values(values)

    def _classify_values(self, values: list) -> EvalResult:
        """Classify trend from a list of values."""
        if len(values) < 2:
            return EvalResult.UNKNOWN

        # Compute deltas
        deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_delta = sum(deltas) / len(deltas)

        # Threshold for "flat" (5% of mean value)
        mean_value = sum(values) / len(values)
        flat_threshold = abs(mean_value) * 0.05 if mean_value != 0 else 0.01

        if self.kind == TrendKind.UP:
            return EvalResult.TRUE if avg_delta > flat_threshold else EvalResult.FALSE
        elif self.kind == TrendKind.DOWN:
            return EvalResult.TRUE if avg_delta < -flat_threshold else EvalResult.FALSE
        elif self.kind == TrendKind.FLAT:
            return EvalResult.TRUE if abs(avg_delta) <= flat_threshold else EvalResult.FALSE
        elif self.kind == TrendKind.ACCELERATING:
            # Check if deltas are increasing
            if len(deltas) < 2:
                return EvalResult.UNKNOWN
            delta_deltas = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
            avg_acceleration = sum(delta_deltas) / len(delta_deltas)
            return EvalResult.TRUE if avg_acceleration > 0 else EvalResult.FALSE

        return EvalResult.UNKNOWN

    def to_dsl(self) -> str:
        return f"trend({self.field}, {self.window}, {self.kind.value})"

    def get_fields(self) -> list[str]:
        return [self.field]


# =============================================================================
# Confidence Gate Predicates
# =============================================================================


@dataclass
class ConfGe(Predicate):
    """
    Confidence gate: only consider field if confidence >= threshold.

    Returns UNKNOWN if confidence is below threshold.
    """

    field: str
    min_confidence: float

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        if fv.confidence >= self.min_confidence:
            return EvalResult.TRUE
        return EvalResult.UNKNOWN  # Low confidence -> treat as unknown

    def to_dsl(self) -> str:
        return f"conf_ge({self.field}, {self.min_confidence})"

    def get_fields(self) -> list[str]:
        return [self.field]


@dataclass
class SourceIn(Predicate):
    """
    Source filter: only consider field if from approved sources.

    Returns UNKNOWN if source is not in approved list.
    """

    field: str
    sources: list[str]

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        fv = ctx.get_field(self.field)
        if not fv.exists:
            return EvalResult.UNKNOWN

        if fv.source_type is None:
            return EvalResult.UNKNOWN  # Unknown source

        if fv.source_type.lower() in [s.lower() for s in self.sources]:
            return EvalResult.TRUE
        return EvalResult.UNKNOWN  # Not from approved source

    def to_dsl(self) -> str:
        sources_str = ", ".join(f'"{s}"' for s in self.sources)
        return f"source_in({self.field}, [{sources_str}])"

    def get_fields(self) -> list[str]:
        return [self.field]


# =============================================================================
# Composite Predicates
# =============================================================================


@dataclass
class And(Predicate):
    """Logical AND of predicates."""

    predicates: list[Predicate]

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        result = EvalResult.TRUE
        for pred in self.predicates:
            result = result & pred.evaluate(ctx)
            if result == EvalResult.FALSE:
                return EvalResult.FALSE  # Short-circuit
        return result

    def to_dsl(self) -> str:
        return "and(" + ", ".join(p.to_dsl() for p in self.predicates) + ")"

    def get_fields(self) -> list[str]:
        fields = []
        for pred in self.predicates:
            fields.extend(pred.get_fields())
        return list(set(fields))


@dataclass
class Or(Predicate):
    """Logical OR of predicates."""

    predicates: list[Predicate]

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        result = EvalResult.FALSE
        for pred in self.predicates:
            result = result | pred.evaluate(ctx)
            if result == EvalResult.TRUE:
                return EvalResult.TRUE  # Short-circuit
        return result

    def to_dsl(self) -> str:
        return "or(" + ", ".join(p.to_dsl() for p in self.predicates) + ")"

    def get_fields(self) -> list[str]:
        fields = []
        for pred in self.predicates:
            fields.extend(pred.get_fields())
        return list(set(fields))


@dataclass
class Not(Predicate):
    """Logical NOT of a predicate."""

    predicate: Predicate

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        return ~self.predicate.evaluate(ctx)

    def to_dsl(self) -> str:
        return f"not({self.predicate.to_dsl()})"

    def get_fields(self) -> list[str]:
        return self.predicate.get_fields()


@dataclass
class Implies(Predicate):
    """
    Implication: if antecedent then consequent.

    P -> Q is equivalent to NOT P OR Q
    """

    antecedent: Predicate
    consequent: Predicate

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        ant_result = self.antecedent.evaluate(ctx)
        if ant_result == EvalResult.FALSE:
            return EvalResult.TRUE  # Vacuously true
        if ant_result == EvalResult.UNKNOWN:
            return EvalResult.UNKNOWN

        # Antecedent is TRUE, check consequent
        return self.consequent.evaluate(ctx)

    def to_dsl(self) -> str:
        return f"implies({self.antecedent.to_dsl()}, {self.consequent.to_dsl()})"

    def get_fields(self) -> list[str]:
        return list(set(self.antecedent.get_fields() + self.consequent.get_fields()))


# =============================================================================
# Confidence-Gated Predicates (Convenience Wrappers)
# =============================================================================


def require_confidence(pred: Predicate, min_confidence: float) -> Predicate:
    """
    Wrap a predicate to require minimum confidence on all its fields.

    Returns AND of:
    - ConfGe for each field
    - The original predicate
    """
    fields = pred.get_fields()
    if not fields:
        return pred

    conf_checks = [ConfGe(f, min_confidence) for f in fields]
    return And([*conf_checks, pred])


def require_source(pred: Predicate, sources: list[str]) -> Predicate:
    """
    Wrap a predicate to require approved sources for all its fields.
    """
    fields = pred.get_fields()
    if not fields:
        return pred

    source_checks = [SourceIn(f, sources) for f in fields]
    return And([*source_checks, pred])


# =============================================================================
# Predicate Registry
# =============================================================================

PREDICATE_REGISTRY: dict[str, type[Predicate]] = {
    "has": Has,
    "eq": Eq,
    "in": In,
    "ge": Ge,
    "le": Le,
    "between": Between,
    "trend": Trend,
    "conf_ge": ConfGe,
    "source_in": SourceIn,
    "and": And,
    "or": Or,
    "not": Not,
    "implies": Implies,
}
