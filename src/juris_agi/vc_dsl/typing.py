"""
Value typing and normalization for JURIS VC DSL.

Handles:
- Type classification (numeric, enum, string, boolean)
- Unit normalization (currency, percentages, months)
- Type coercion for comparisons
"""

import re
from enum import Enum
from typing import Any, Optional, Union


class ValueType(Enum):
    """Types of values in the DSL."""

    NUMERIC = "numeric"
    ENUM = "enum"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    UNKNOWN = "unknown"


class TrendKind(Enum):
    """Types of trends for trend predicates."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    ACCELERATING = "accelerating"


class UnitType(Enum):
    """Unit types for normalization."""

    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    MONTHS = "months"
    YEARS = "years"
    COUNT = "count"
    MULTIPLIER = "multiplier"
    NONE = "none"


# Unit multipliers for K/M/B/T suffixes
MAGNITUDE_MULTIPLIERS: dict[str, float] = {
    "": 1,
    "k": 1_000,
    "K": 1_000,
    "m": 1_000_000,
    "M": 1_000_000,
    "mm": 1_000_000,
    "MM": 1_000_000,
    "b": 1_000_000_000,
    "B": 1_000_000_000,
    "bn": 1_000_000_000,
    "BN": 1_000_000_000,
    "t": 1_000_000_000_000,
    "T": 1_000_000_000_000,
}

# Currency symbols
CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}

# Known enum fields and their valid values
ENUM_FIELDS: dict[str, list[str]] = {
    "stage": ["pre_seed", "seed", "series_a", "series_b", "series_c", "growth", "late_stage"],
    "product_stage": ["concept", "prototype", "mvp", "beta", "production"],
    "polarity": ["supportive", "risk", "neutral"],
    "instrument_type": ["safe", "convertible_note", "preferred", "common"],
}

# Enum normalization mappings
ENUM_MAPPINGS: dict[str, dict[str, str]] = {
    "stage": {
        "pre-seed": "pre_seed",
        "preseed": "pre_seed",
        "series a": "series_a",
        "series-a": "series_a",
        "a": "series_a",
        "series b": "series_b",
        "series-b": "series_b",
        "b": "series_b",
        "series c": "series_c",
        "series-c": "series_c",
        "c": "series_c",
        "series d": "series_d",
        "series-d": "series_d",
        "d": "series_d",
        "late": "late_stage",
        "late-stage": "late_stage",
    },
    "product_stage": {
        "idea": "concept",
        "proto": "prototype",
        "alpha": "beta",
        "private beta": "beta",
        "public beta": "beta",
        "prod": "production",
        "live": "production",
        "ga": "production",
        "launched": "production",
    },
}

# Known boolean field patterns
BOOLEAN_FIELDS: set[str] = {
    "is_profitable",
    "has_product",
    "has_revenue",
    "has_moat",
    "is_regulated",
    "founder_technical",
    "repeat_founder",
}

# Fields with specific units
FIELD_UNITS: dict[str, UnitType] = {
    # Currency
    "arr": UnitType.CURRENCY,
    "mrr": UnitType.CURRENCY,
    "revenue": UnitType.CURRENCY,
    "gmv": UnitType.CURRENCY,
    "pre_money_valuation": UnitType.CURRENCY,
    "post_money_valuation": UnitType.CURRENCY,
    "raise_amount": UnitType.CURRENCY,
    "monthly_burn": UnitType.CURRENCY,
    "tam": UnitType.CURRENCY,
    "sam": UnitType.CURRENCY,
    "som": UnitType.CURRENCY,
    "cac": UnitType.CURRENCY,
    "ltv": UnitType.CURRENCY,
    "arpu": UnitType.CURRENCY,
    # Percentages/Ratios
    "gross_margin": UnitType.PERCENTAGE,
    "net_margin": UnitType.PERCENTAGE,
    "growth_rate": UnitType.PERCENTAGE,
    "churn_rate": UnitType.PERCENTAGE,
    "retention_rate": UnitType.PERCENTAGE,
    "conversion_rate": UnitType.PERCENTAGE,
    # Months
    "runway_months": UnitType.MONTHS,
    "sales_cycle_months": UnitType.MONTHS,
    # Multipliers
    "ltv_cac_ratio": UnitType.MULTIPLIER,
    "revenue_multiple": UnitType.MULTIPLIER,
    # Counts
    "team_size": UnitType.COUNT,
    "employees": UnitType.COUNT,
    "customers": UnitType.COUNT,
    "users": UnitType.COUNT,
    "dau": UnitType.COUNT,
    "mau": UnitType.COUNT,
}


def infer_value_type(value: Any, field_name: Optional[str] = None) -> ValueType:
    """
    Infer the type of a value.

    Args:
        value: The value to type
        field_name: Optional field name for context

    Returns:
        Inferred ValueType
    """
    if value is None:
        return ValueType.UNKNOWN

    if isinstance(value, bool):
        return ValueType.BOOLEAN

    if isinstance(value, (int, float)):
        return ValueType.NUMERIC

    if isinstance(value, list):
        return ValueType.LIST

    if isinstance(value, str):
        # Check for known boolean fields
        if field_name and field_name in BOOLEAN_FIELDS:
            return ValueType.BOOLEAN

        # Check for known enum fields
        if field_name:
            for enum_field in ENUM_FIELDS:
                if enum_field in field_name:
                    return ValueType.ENUM

        # Check if it looks numeric (with units)
        if re.match(r"^[\$€£¥₹]?\d", value):
            return ValueType.NUMERIC

        # Check for boolean strings
        if value.lower() in {"true", "false", "yes", "no"}:
            return ValueType.BOOLEAN

        return ValueType.STRING

    return ValueType.UNKNOWN


def normalize_numeric(value: Any) -> Optional[float]:
    """
    Normalize a numeric value, handling units and suffixes.

    Handles:
    - Raw numbers
    - Strings with K/M/B/T suffixes
    - Currency symbols
    - Percentages

    Args:
        value: Value to normalize

    Returns:
        Normalized float, or None if not numeric
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # Handle percentages
    if value.endswith("%"):
        try:
            return float(value[:-1].replace(",", "")) / 100
        except ValueError:
            return None

    # Strip currency symbols
    for symbol in CURRENCY_SYMBOLS:
        if value.startswith(symbol):
            value = value[len(symbol):].strip()
            break

    # Try to extract number and suffix
    match = re.match(r"^([\d,\.]+)\s*([a-zA-Z]*)$", value)
    if not match:
        # Try without suffix
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None

    num_str = match.group(1).replace(",", "")
    suffix = match.group(2)

    try:
        num = float(num_str)
    except ValueError:
        return None

    # Apply multiplier
    multiplier = MAGNITUDE_MULTIPLIERS.get(suffix, 1)
    return num * multiplier


def normalize_boolean(value: Any) -> Optional[bool]:
    """Normalize a boolean value."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in {"true", "yes", "1", "y"}:
            return True
        if lower in {"false", "no", "0", "n"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return None


def normalize_enum(value: Any, field_name: Optional[str] = None) -> Optional[str]:
    """
    Normalize an enum value to its canonical form.

    Args:
        value: Value to normalize
        field_name: Field name for context

    Returns:
        Normalized enum string, or None if invalid
    """
    if not isinstance(value, str):
        return str(value) if value is not None else None

    normalized = value.lower().strip()

    # Try field-specific mapping
    if field_name:
        for enum_field, mappings in ENUM_MAPPINGS.items():
            if enum_field in field_name:
                if normalized in mappings:
                    return mappings[normalized]

    # Return as-is with underscores
    return normalized.replace(" ", "_").replace("-", "_")


def normalize_string(value: Any) -> str:
    """Normalize a string value."""
    if value is None:
        return ""
    return str(value).strip()


def normalize_value(
    value: Any,
    value_type: ValueType,
    field_name: Optional[str] = None,
) -> Any:
    """
    Normalize a value according to its type.

    Args:
        value: Value to normalize
        value_type: Expected type
        field_name: Optional field name for context

    Returns:
        Normalized value
    """
    if value_type == ValueType.NUMERIC:
        return normalize_numeric(value)
    elif value_type == ValueType.BOOLEAN:
        return normalize_boolean(value)
    elif value_type == ValueType.ENUM:
        return normalize_enum(value, field_name)
    elif value_type == ValueType.STRING:
        return normalize_string(value)
    elif value_type == ValueType.LIST:
        if isinstance(value, list):
            return value
        return [value] if value is not None else []
    else:
        return value


def get_field_unit(field_name: str) -> UnitType:
    """Get the expected unit type for a field."""
    return FIELD_UNITS.get(field_name, UnitType.NONE)


def convert_to_base_unit(value: float, from_unit: UnitType, field_name: str) -> float:
    """
    Convert a value to its base unit.

    For currencies: no conversion (assume USD)
    For percentages: convert to ratio (0-1)
    For months: keep as months
    """
    if from_unit == UnitType.PERCENTAGE:
        # If already a ratio (0-1), keep as is
        if 0 <= value <= 1:
            return value
        # Otherwise convert from percentage
        return value / 100

    return value


class TypedValue:
    """A value with type information and normalization."""

    def __init__(
        self,
        raw_value: Any,
        value_type: Optional[ValueType] = None,
        unit: Optional[UnitType] = None,
        field_name: Optional[str] = None,
    ):
        self.raw_value = raw_value
        self.field_name = field_name

        # Infer type if not provided
        if value_type is None:
            self.value_type = infer_value_type(raw_value, field_name)
        else:
            self.value_type = value_type

        # Infer unit if not provided
        if unit is None and field_name:
            self.unit = get_field_unit(field_name)
        else:
            self.unit = unit or UnitType.NONE

        # Normalize the value
        self._normalized: Optional[Any] = None

    @property
    def normalized(self) -> Any:
        """Get the normalized value (lazy computation)."""
        if self._normalized is None:
            self._normalized = normalize_value(
                self.raw_value, self.value_type, self.field_name
            )
        return self._normalized

    def compare_to(self, other: "TypedValue") -> int:
        """
        Compare to another typed value.

        Returns:
            -1 if self < other
            0 if self == other
            1 if self > other
            Raises TypeError if not comparable
        """
        if self.value_type != other.value_type:
            raise TypeError(f"Cannot compare {self.value_type} to {other.value_type}")

        if self.normalized is None or other.normalized is None:
            raise TypeError("Cannot compare None values")

        if self.normalized < other.normalized:
            return -1
        elif self.normalized > other.normalized:
            return 1
        return 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypedValue):
            return self.normalized == other.normalized
        return self.normalized == other

    def __lt__(self, other: "TypedValue") -> bool:
        return self.compare_to(other) < 0

    def __gt__(self, other: "TypedValue") -> bool:
        return self.compare_to(other) > 0

    def __le__(self, other: "TypedValue") -> bool:
        return self.compare_to(other) <= 0

    def __ge__(self, other: "TypedValue") -> bool:
        return self.compare_to(other) >= 0

    def __repr__(self) -> str:
        return f"TypedValue({self.raw_value!r}, type={self.value_type.value}, normalized={self.normalized})"
