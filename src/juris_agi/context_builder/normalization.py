"""
Normalization utilities for claim values.

Handles:
- Numeric unit conversion (K, M, B, T)
- Currency normalization
- Percentage normalization
- Enum/category standardization
- Date parsing and normalization
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union


@dataclass
class NormalizedValue:
    """A normalized value with metadata."""

    original: Any
    normalized: Any
    unit: Optional[str] = None
    conversion_applied: Optional[str] = None


# Unit multipliers
UNIT_MULTIPLIERS: dict[str, float] = {
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
    "tn": 1_000_000_000_000,
    "TN": 1_000_000_000_000,
}

# Currency symbols
CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₿": "BTC",
}

# Standard enum mappings
STAGE_MAPPINGS: dict[str, str] = {
    "pre-seed": "pre_seed",
    "preseed": "pre_seed",
    "pre seed": "pre_seed",
    "seed": "seed",
    "series a": "series_a",
    "series-a": "series_a",
    "seriesa": "series_a",
    "a": "series_a",
    "series b": "series_b",
    "series-b": "series_b",
    "seriesb": "series_b",
    "b": "series_b",
    "series c": "series_c",
    "series-c": "series_c",
    "seriesc": "series_c",
    "c": "series_c",
    "series d": "series_d",
    "series-d": "series_d",
    "seriesd": "series_d",
    "d": "series_d",
    "growth": "growth",
    "late": "late_stage",
    "late stage": "late_stage",
    "late-stage": "late_stage",
    "ipo": "ipo",
    "public": "public",
}

PRODUCT_STAGE_MAPPINGS: dict[str, str] = {
    "idea": "concept",
    "concept": "concept",
    "prototype": "prototype",
    "mvp": "mvp",
    "beta": "beta",
    "private beta": "beta",
    "public beta": "beta",
    "alpha": "alpha",
    "production": "production",
    "prod": "production",
    "live": "production",
    "launched": "production",
    "ga": "production",
    "general availability": "production",
}

POLARITY_MAPPINGS: dict[str, str] = {
    "positive": "supportive",
    "supportive": "supportive",
    "pro": "supportive",
    "strength": "supportive",
    "negative": "risk",
    "risk": "risk",
    "con": "risk",
    "weakness": "risk",
    "concern": "risk",
    "neutral": "neutral",
    "unknown": "neutral",
    "mixed": "neutral",
}


def normalize_numeric(
    value: Union[str, int, float],
    target_unit: Optional[str] = None,
) -> NormalizedValue:
    """
    Normalize numeric values with unit suffixes.

    Examples:
        "1.5M" -> 1500000
        "$500K" -> 500000 (with unit USD)
        "2.3B" -> 2300000000
        "15%" -> 0.15 (as ratio)

    Args:
        value: Numeric value with optional unit suffix
        target_unit: Optional target unit to convert to

    Returns:
        NormalizedValue with normalized numeric value
    """
    if isinstance(value, (int, float)):
        return NormalizedValue(original=value, normalized=value)

    if not isinstance(value, str):
        return NormalizedValue(original=value, normalized=value)

    original = value
    value = value.strip()

    # Check for percentage
    if value.endswith("%"):
        try:
            num = float(value[:-1].replace(",", ""))
            return NormalizedValue(
                original=original,
                normalized=num / 100,
                unit="ratio",
                conversion_applied="percentage to ratio",
            )
        except ValueError:
            pass

    # Extract currency symbol
    currency = None
    for symbol, code in CURRENCY_SYMBOLS.items():
        if value.startswith(symbol):
            currency = code
            value = value[len(symbol) :].strip()
            break

    # Extract numeric part and unit suffix
    match = re.match(r"^([\d,\.]+)\s*([a-zA-Z]*)$", value)
    if not match:
        return NormalizedValue(original=original, normalized=original)

    num_str = match.group(1).replace(",", "")
    unit_suffix = match.group(2)

    try:
        num = float(num_str)
    except ValueError:
        return NormalizedValue(original=original, normalized=original)

    # Apply unit multiplier
    multiplier = UNIT_MULTIPLIERS.get(unit_suffix, 1)
    normalized = num * multiplier

    conversion = None
    if multiplier != 1:
        conversion = f"multiplied by {multiplier:,.0f} ({unit_suffix} suffix)"

    return NormalizedValue(
        original=original,
        normalized=normalized,
        unit=currency,
        conversion_applied=conversion,
    )


def normalize_currency(
    value: Union[str, int, float],
    target_currency: str = "USD",
) -> NormalizedValue:
    """
    Normalize currency values.

    Currently assumes all values are in target currency if no symbol specified.
    Future: could integrate exchange rates.

    Args:
        value: Currency value
        target_currency: Target currency code

    Returns:
        NormalizedValue with normalized currency value
    """
    result = normalize_numeric(value)

    if result.unit is None:
        result.unit = target_currency

    return result


def normalize_enum(
    value: str,
    enum_type: str,
) -> NormalizedValue:
    """
    Normalize enum/category values to standard forms.

    Args:
        value: Enum value to normalize
        enum_type: Type of enum (stage, product_stage, polarity)

    Returns:
        NormalizedValue with normalized enum value
    """
    if not isinstance(value, str):
        return NormalizedValue(original=value, normalized=value)

    original = value
    normalized = value.lower().strip()

    mappings: dict[str, str] = {}
    if enum_type == "stage":
        mappings = STAGE_MAPPINGS
    elif enum_type == "product_stage":
        mappings = PRODUCT_STAGE_MAPPINGS
    elif enum_type == "polarity":
        mappings = POLARITY_MAPPINGS

    if normalized in mappings:
        return NormalizedValue(
            original=original,
            normalized=mappings[normalized],
            conversion_applied=f"mapped to standard {enum_type}",
        )

    return NormalizedValue(original=original, normalized=original)


def normalize_date(
    value: Union[str, datetime],
) -> NormalizedValue:
    """
    Normalize date values to datetime objects.

    Supports formats:
    - ISO: 2024-01-15, 2024-01-15T10:30:00
    - US: 01/15/2024, 1/15/24
    - EU: 15/01/2024
    - Textual: Jan 2024, January 15, 2024, Q1 2024

    Args:
        value: Date value to normalize

    Returns:
        NormalizedValue with datetime object
    """
    if isinstance(value, datetime):
        return NormalizedValue(original=value, normalized=value)

    if not isinstance(value, str):
        return NormalizedValue(original=value, normalized=value)

    original = value
    value = value.strip()

    # ISO format
    iso_patterns = [
        (r"^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$", "%Y-%m-%d"),
        (r"^(\d{4})/(\d{2})/(\d{2})$", "%Y/%m/%d"),
    ]

    for pattern, fmt in iso_patterns:
        if re.match(pattern, value):
            try:
                dt = datetime.strptime(value[:10], fmt[:8] if "T" in value else fmt)
                return NormalizedValue(
                    original=original,
                    normalized=dt,
                    conversion_applied="parsed ISO date",
                )
            except ValueError:
                pass

    # US format MM/DD/YYYY
    us_match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", value)
    if us_match:
        month, day, year = us_match.groups()
        if len(year) == 2:
            year = "20" + year if int(year) < 50 else "19" + year
        try:
            dt = datetime(int(year), int(month), int(day))
            return NormalizedValue(
                original=original,
                normalized=dt,
                conversion_applied="parsed US date format",
            )
        except ValueError:
            pass

    # Quarter format Q1 2024
    quarter_match = re.match(r"^Q([1-4])\s*(\d{4})$", value)
    if quarter_match:
        quarter, year = quarter_match.groups()
        month = (int(quarter) - 1) * 3 + 1
        dt = datetime(int(year), month, 1)
        return NormalizedValue(
            original=original,
            normalized=dt,
            conversion_applied="parsed quarter to first day",
        )

    # Month Year format: Jan 2024, January 2024
    months = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    month_year_match = re.match(r"^([a-zA-Z]+)\s*(\d{4})$", value)
    if month_year_match:
        month_str, year = month_year_match.groups()
        month = months.get(month_str.lower())
        if month:
            dt = datetime(int(year), month, 1)
            return NormalizedValue(
                original=original,
                normalized=dt,
                conversion_applied="parsed month-year to first day",
            )

    # Year only
    if re.match(r"^\d{4}$", value):
        dt = datetime(int(value), 1, 1)
        return NormalizedValue(
            original=original,
            normalized=dt,
            conversion_applied="parsed year to January 1",
        )

    return NormalizedValue(original=original, normalized=original)


def normalize_claim_value(
    value: Any,
    claim_type: str,
    field: str,
) -> NormalizedValue:
    """
    Normalize a claim value based on its type and field.

    Applies appropriate normalization based on known field types.

    Args:
        value: Value to normalize
        claim_type: Claim type (e.g., "traction", "round_terms")
        field: Field name (e.g., "arr", "pre_money_valuation")

    Returns:
        NormalizedValue with appropriate normalization applied
    """
    # Numeric financial fields
    financial_fields = {
        "arr", "mrr", "revenue", "gmv",
        "pre_money_valuation", "post_money_valuation", "valuation",
        "raise_amount", "investment_amount",
        "monthly_burn", "burn_rate",
        "tam", "sam", "som",
        "cac", "ltv", "arpu",
    }

    if field in financial_fields:
        return normalize_currency(value)

    # Percentage fields
    percentage_fields = {
        "gross_margin", "net_margin", "growth_rate",
        "churn_rate", "retention_rate",
        "conversion_rate",
    }

    if field in percentage_fields:
        result = normalize_numeric(value)
        # Ensure it's a ratio (0-1 range)
        if isinstance(result.normalized, (int, float)):
            if result.normalized > 1:
                result.normalized = result.normalized / 100
                result.conversion_applied = (result.conversion_applied or "") + ", converted to ratio"
        return result

    # Stage fields
    if field in {"stage", "round", "funding_stage"}:
        return normalize_enum(str(value) if value else "", "stage")

    if field in {"product_stage", "development_stage"}:
        return normalize_enum(str(value) if value else "", "product_stage")

    # Date fields
    date_fields = {"founded_date", "as_of_date", "last_update", "exit_date"}
    if field in date_fields:
        return normalize_date(value)

    # Numeric count fields
    count_fields = {
        "team_size", "headcount", "employees",
        "customers", "users", "dau", "mau",
        "runway_months",
    }

    if field in count_fields:
        return normalize_numeric(value)

    # Default: no normalization
    return NormalizedValue(original=value, normalized=value)
