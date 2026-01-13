"""
Time-series support for JURIS VC DSL.

Provides robust trend analysis over temporal claim data, suitable for
domains like insurance, pharma, and VC where time-series metrics are common.

Features:
- Flexible time-point parsing (quarters, months, dates)
- Robust trend classification (UP, DOWN, FLAT, ACCELERATING)
- Handling of missing points and noisy data
- Feature extraction (last value, slope, volatility, acceleration)
"""

import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional, Sequence, Union

from .typing import TrendKind

logger = logging.getLogger(__name__)


# =============================================================================
# Time Point Parsing
# =============================================================================


class TimeGranularity(Enum):
    """Granularity of time points."""

    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    UNKNOWN = "unknown"


@dataclass
class TimePoint:
    """A parsed time point with ordinal for sorting."""

    raw: str  # Original string representation
    ordinal: float  # Numeric ordinal for sorting/comparison
    granularity: TimeGranularity
    year: Optional[int] = None
    quarter: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

    def __lt__(self, other: "TimePoint") -> bool:
        return self.ordinal < other.ordinal

    def __le__(self, other: "TimePoint") -> bool:
        return self.ordinal <= other.ordinal


# Quarter patterns: Q1 2024, 2024-Q1, 2024Q1, Q1'24, 1Q24, etc.
QUARTER_PATTERNS = [
    r"Q([1-4])\s*['\-]?\s*(\d{2,4})",  # Q1'24, Q1 2024
    r"(\d{2,4})\s*['\-]?\s*Q([1-4])",  # 2024-Q1, 24Q1
    r"([1-4])Q\s*['\-]?\s*(\d{2,4})",  # 1Q24, 1Q'24
]

# Month patterns: Jan 2024, 2024-01, January 2024
MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _normalize_year(year_str: str) -> int:
    """Normalize 2-digit or 4-digit year."""
    year = int(year_str)
    if year < 100:
        # Assume 2000s for 00-99
        year = 2000 + year
    return year


def parse_time_point(t: Union[str, datetime, date, int, float]) -> TimePoint:
    """
    Parse a time point from various formats.

    Supports:
    - Quarters: "Q1 2024", "2024-Q1", "1Q24"
    - Months: "Jan 2024", "2024-01", "January 2024"
    - Dates: "2024-01-15", ISO format
    - Years: "2024", 2024
    - Unix timestamps
    """
    if isinstance(t, datetime):
        return TimePoint(
            raw=t.isoformat(),
            ordinal=t.timestamp(),
            granularity=TimeGranularity.DAILY,
            year=t.year,
            month=t.month,
            day=t.day,
        )

    if isinstance(t, date):
        dt = datetime.combine(t, datetime.min.time())
        return TimePoint(
            raw=t.isoformat(),
            ordinal=dt.timestamp(),
            granularity=TimeGranularity.DAILY,
            year=t.year,
            month=t.month,
            day=t.day,
        )

    if isinstance(t, (int, float)):
        # Could be a year or a timestamp
        if 1900 < t < 2200:
            # Likely a year
            return TimePoint(
                raw=str(int(t)),
                ordinal=t * 12,  # Convert year to months for sorting
                granularity=TimeGranularity.YEARLY,
                year=int(t),
            )
        else:
            # Assume Unix timestamp
            dt = datetime.fromtimestamp(t)
            return TimePoint(
                raw=dt.isoformat(),
                ordinal=t,
                granularity=TimeGranularity.DAILY,
                year=dt.year,
                month=dt.month,
                day=dt.day,
            )

    t_str = str(t).strip()

    # Try quarter patterns
    for pattern in QUARTER_PATTERNS:
        match = re.match(pattern, t_str, re.IGNORECASE)
        if match:
            groups = match.groups()
            # Determine which group is quarter and which is year
            if pattern.startswith(r"Q"):
                quarter, year = int(groups[0]), _normalize_year(groups[1])
            elif pattern.startswith(r"(\d"):
                year, quarter = _normalize_year(groups[0]), int(groups[1])
            else:
                quarter, year = int(groups[0]), _normalize_year(groups[1])

            # Ordinal: year * 4 + quarter
            ordinal = year * 4 + quarter
            return TimePoint(
                raw=t_str,
                ordinal=ordinal,
                granularity=TimeGranularity.QUARTERLY,
                year=year,
                quarter=quarter,
            )

    # Try month name patterns
    for month_name, month_num in MONTH_NAMES.items():
        pattern = rf"({month_name})\s+(\d{{2,4}})"
        match = re.match(pattern, t_str, re.IGNORECASE)
        if match:
            year = _normalize_year(match.group(2))
            ordinal = year * 12 + month_num
            return TimePoint(
                raw=t_str,
                ordinal=ordinal,
                granularity=TimeGranularity.MONTHLY,
                year=year,
                month=month_num,
            )

    # Try ISO date patterns
    iso_patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", "ymd"),  # YYYY-MM-DD
        (r"(\d{4})/(\d{2})/(\d{2})", "ymd"),  # YYYY/MM/DD
        (r"(\d{2})/(\d{2})/(\d{4})", "mdy"),  # MM/DD/YYYY
    ]
    for pattern, order in iso_patterns:
        match = re.match(pattern, t_str)
        if match:
            groups = [int(g) for g in match.groups()]
            if order == "ymd":
                year, month, day = groups[0], groups[1], groups[2]
            else:  # mdy
                month, day, year = groups[0], groups[1], groups[2]

            try:
                dt = datetime(year, month, day)
                return TimePoint(
                    raw=t_str,
                    ordinal=dt.timestamp(),
                    granularity=TimeGranularity.DAILY,
                    year=year,
                    month=month,
                    day=day,
                )
            except ValueError:
                pass

    # Try YYYY-MM pattern
    match = re.match(r"(\d{4})-(\d{2})", t_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        ordinal = year * 12 + month
        return TimePoint(
            raw=t_str,
            ordinal=ordinal,
            granularity=TimeGranularity.MONTHLY,
            year=year,
            month=month,
        )

    # Try plain year
    match = re.match(r"(\d{4})", t_str)
    if match:
        year = int(match.group(1))
        return TimePoint(
            raw=t_str,
            ordinal=year * 12,
            granularity=TimeGranularity.YEARLY,
            year=year,
        )

    # Unknown format - try to use as-is for ordering
    return TimePoint(
        raw=t_str,
        ordinal=hash(t_str),  # Fallback for unknown formats
        granularity=TimeGranularity.UNKNOWN,
    )


# =============================================================================
# Time Series Data Structures
# =============================================================================


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""

    time: TimePoint
    value: float
    confidence: float = 1.0
    is_interpolated: bool = False


@dataclass
class TimeSeries:
    """A time series of values."""

    field: str
    points: list[TimeSeriesPoint] = field(default_factory=list)
    unit: Optional[str] = None

    def __len__(self) -> int:
        return len(self.points)

    def sort(self) -> "TimeSeries":
        """Sort points by time and return self."""
        self.points.sort(key=lambda p: p.time.ordinal)
        return self

    def get_values(self) -> list[float]:
        """Get just the values in time order."""
        self.sort()
        return [p.value for p in self.points]

    def get_times(self) -> list[TimePoint]:
        """Get just the time points in order."""
        self.sort()
        return [p.time for p in self.points]

    @classmethod
    def from_list(
        cls,
        field: str,
        data: list[dict[str, Any]],
        time_key: str = "t",
        value_key: str = "value",
        confidence_key: str = "confidence",
    ) -> "TimeSeries":
        """
        Create TimeSeries from a list of dicts.

        Args:
            field: Field name
            data: List of dicts with time and value keys
            time_key: Key for time in each dict
            value_key: Key for value in each dict
            confidence_key: Key for confidence (optional)
        """
        points = []
        for item in data:
            if time_key not in item or value_key not in item:
                continue

            try:
                time_point = parse_time_point(item[time_key])
                value = float(item[value_key])
                confidence = float(item.get(confidence_key, 1.0))

                points.append(TimeSeriesPoint(
                    time=time_point,
                    value=value,
                    confidence=confidence,
                ))
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping invalid time series point: {item}, error: {e}")
                continue

        ts = cls(field=field, points=points)
        return ts.sort()


# =============================================================================
# Trend Classification
# =============================================================================


@dataclass
class TrendResult:
    """Result of trend analysis."""

    kind: TrendKind
    slope: float  # Slope of linear fit
    r_squared: float  # Goodness of fit
    volatility: float  # Standard deviation of residuals
    acceleration: float  # Second derivative (change in slope)
    last_value: Optional[float] = None
    first_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    confidence: float = 1.0  # Confidence in the trend classification


@dataclass
class TrendConfig:
    """Configuration for trend analysis."""

    # Threshold for "flat" - as a fraction of mean absolute value
    flat_threshold: float = 0.05

    # Minimum points required for trend analysis
    min_points: int = 2

    # Minimum points required for acceleration detection
    min_points_for_acceleration: int = 3

    # R-squared threshold for confident trend classification
    r_squared_threshold: float = 0.5

    # Maximum gap between points (as fraction of total range) before interpolation
    max_gap_fraction: float = 0.3

    # Weight decay for older points (0 = equal weight, 1 = only recent)
    recency_weight: float = 0.0


def _linear_regression(
    x: Sequence[float],
    y: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> tuple[float, float, float]:
    """
    Compute weighted linear regression.

    Returns:
        (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    if weights is None:
        weights = [1.0] * n

    # Weighted means
    sum_w = sum(weights)
    mean_x = sum(w * xi for w, xi in zip(weights, x)) / sum_w
    mean_y = sum(w * yi for w, yi in zip(weights, y)) / sum_w

    # Covariance and variance
    cov_xy = sum(w * (xi - mean_x) * (yi - mean_y) for w, xi, yi in zip(weights, x, y))
    var_x = sum(w * (xi - mean_x) ** 2 for w, xi in zip(weights, x))
    var_y = sum(w * (yi - mean_y) ** 2 for w, yi in zip(weights, y))

    if var_x == 0:
        return 0.0, mean_y, 0.0

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    # R-squared
    if var_y == 0:
        r_squared = 1.0  # Perfect fit (all y values identical)
    else:
        ss_res = sum(w * (yi - (intercept + slope * xi)) ** 2 for w, xi, yi in zip(weights, x, y))
        ss_tot = var_y * sum_w
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)

    return slope, intercept, r_squared


def _compute_volatility(
    values: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Compute volatility as RMSE of residuals."""
    if len(values) < 2:
        return 0.0

    residuals = [v - p for v, p in zip(values, predicted)]
    mse = sum(r ** 2 for r in residuals) / len(residuals)
    return math.sqrt(mse)


def _compute_acceleration(
    x: Sequence[float],
    y: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """
    Compute acceleration (second derivative) using quadratic fit.
    """
    n = len(x)
    if n < 3:
        return 0.0

    # Fit quadratic: y = a*x^2 + b*x + c
    # Acceleration = 2*a

    if weights is None:
        weights = [1.0] * n

    sum_w = sum(weights)

    # Create design matrix for quadratic
    x2 = [xi ** 2 for xi in x]

    mean_x = sum(w * xi for w, xi in zip(weights, x)) / sum_w
    mean_x2 = sum(w * xi for w, xi in zip(weights, x2)) / sum_w
    mean_y = sum(w * yi for w, yi in zip(weights, y)) / sum_w

    # Normal equations for quadratic fit (simplified)
    # We just need the coefficient of x^2
    try:
        # Use numpy-like calculation
        var_x2 = sum(w * (xi - mean_x2) ** 2 for w, xi in zip(weights, x2))
        cov_x2y = sum(w * (xi - mean_x2) * (yi - mean_y) for w, xi, yi in zip(weights, x2, y))

        if var_x2 == 0:
            return 0.0

        # This is an approximation - proper quadratic regression would need matrix solve
        a = cov_x2y / var_x2
        return 2 * a
    except (ValueError, ZeroDivisionError):
        return 0.0


def classify_trend(
    series: TimeSeries,
    window: Optional[int] = None,
    config: Optional[TrendConfig] = None,
) -> TrendResult:
    """
    Classify the trend of a time series.

    Args:
        series: Time series to analyze
        window: Number of most recent points to consider (None = all)
        config: Configuration for trend analysis

    Returns:
        TrendResult with classification and features
    """
    config = config or TrendConfig()
    series.sort()

    # Get points within window
    points = series.points
    if window is not None and len(points) > window:
        points = points[-window:]

    if len(points) < config.min_points:
        return TrendResult(
            kind=TrendKind.FLAT,
            slope=0.0,
            r_squared=0.0,
            volatility=0.0,
            acceleration=0.0,
            confidence=0.0,
        )

    # Extract values and normalize time to 0-indexed
    values = [p.value for p in points]
    times = [float(i) for i in range(len(points))]  # Normalize to 0, 1, 2, ...

    # Compute weights (optional recency weighting)
    if config.recency_weight > 0:
        n = len(points)
        weights = [
            (1 - config.recency_weight) + config.recency_weight * (i / (n - 1))
            for i in range(n)
        ]
    else:
        weights = [p.confidence for p in points]

    # Linear regression
    slope, intercept, r_squared = _linear_regression(times, values, weights)

    # Predicted values for volatility
    predicted = [intercept + slope * t for t in times]
    volatility = _compute_volatility(values, predicted)

    # Acceleration
    acceleration = 0.0
    if len(points) >= config.min_points_for_acceleration:
        acceleration = _compute_acceleration(times, values, weights)

    # Statistics
    last_value = values[-1] if values else None
    first_value = values[0] if values else None
    min_value = min(values) if values else None
    max_value = max(values) if values else None
    mean_value = statistics.mean(values) if values else None

    # Classify trend
    # Normalize slope by mean value for threshold comparison
    if mean_value and mean_value != 0:
        normalized_slope = slope / abs(mean_value)
    else:
        normalized_slope = slope

    # Determine trend kind
    if abs(normalized_slope) <= config.flat_threshold:
        kind = TrendKind.FLAT
    elif normalized_slope > 0:
        # Check for acceleration
        if acceleration > 0 and len(points) >= config.min_points_for_acceleration:
            kind = TrendKind.ACCELERATING
        else:
            kind = TrendKind.UP
    else:
        kind = TrendKind.DOWN

    # Compute confidence based on R-squared and number of points
    point_factor = min(1.0, len(points) / 5)  # More points = more confidence
    confidence = r_squared * point_factor

    return TrendResult(
        kind=kind,
        slope=slope,
        r_squared=r_squared,
        volatility=volatility,
        acceleration=acceleration,
        last_value=last_value,
        first_value=first_value,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        confidence=confidence,
    )


# =============================================================================
# Feature Extraction
# =============================================================================


@dataclass
class TimeSeriesFeatures:
    """Extracted features from a time series."""

    field: str

    # Core features
    last_value: Optional[float] = None
    first_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None

    # Trend features
    trend_kind: Optional[TrendKind] = None
    slope: Optional[float] = None
    slope_normalized: Optional[float] = None  # Slope / mean_value
    r_squared: Optional[float] = None
    volatility: Optional[float] = None
    acceleration: Optional[float] = None

    # Derived features
    range_value: Optional[float] = None  # max - min
    growth_rate: Optional[float] = None  # (last - first) / first
    cv: Optional[float] = None  # Coefficient of variation

    # Metadata
    num_points: int = 0
    time_span: Optional[str] = None  # e.g., "Q1 2023 - Q4 2024"
    confidence: float = 0.0


def extract_features(
    series: TimeSeries,
    window: Optional[int] = None,
    config: Optional[TrendConfig] = None,
) -> TimeSeriesFeatures:
    """
    Extract features from a time series for use in predicates.

    Args:
        series: Time series to analyze
        window: Number of most recent points to consider
        config: Configuration for trend analysis

    Returns:
        TimeSeriesFeatures with all extracted features
    """
    series.sort()

    points = series.points
    if window is not None and len(points) > window:
        points = points[-window:]

    if not points:
        return TimeSeriesFeatures(field=series.field, num_points=0)

    values = [p.value for p in points]

    # Basic statistics
    last_value = values[-1]
    first_value = values[0]
    min_value = min(values)
    max_value = max(values)
    mean_value = statistics.mean(values)

    # Derived statistics
    range_value = max_value - min_value

    growth_rate = None
    if first_value != 0:
        growth_rate = (last_value - first_value) / abs(first_value)

    cv = None
    if mean_value != 0 and len(values) > 1:
        cv = statistics.stdev(values) / abs(mean_value)

    # Trend analysis
    trend_result = classify_trend(series, window=window, config=config)

    # Time span
    times = [p.time for p in points]
    time_span = f"{times[0].raw} - {times[-1].raw}" if len(times) >= 2 else times[0].raw

    # Normalized slope
    slope_normalized = None
    if mean_value != 0:
        slope_normalized = trend_result.slope / abs(mean_value)

    return TimeSeriesFeatures(
        field=series.field,
        last_value=last_value,
        first_value=first_value,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        trend_kind=trend_result.kind,
        slope=trend_result.slope,
        slope_normalized=slope_normalized,
        r_squared=trend_result.r_squared,
        volatility=trend_result.volatility,
        acceleration=trend_result.acceleration,
        range_value=range_value,
        growth_rate=growth_rate,
        cv=cv,
        num_points=len(points),
        time_span=time_span,
        confidence=trend_result.confidence,
    )


# =============================================================================
# Integration with Working Set / Eval Context
# =============================================================================


def add_timeseries_features_to_context(
    context_fields: dict[str, Any],
    series: TimeSeries,
    window: Optional[int] = None,
    prefix: Optional[str] = None,
) -> dict[str, Any]:
    """
    Add extracted time series features to a context dictionary.

    Creates fields like:
    - {field}_last: last value
    - {field}_slope: slope
    - {field}_trend_{window}: trend kind
    - etc.

    Args:
        context_fields: Existing context dictionary to update
        series: Time series to extract features from
        window: Window size for trend analysis
        prefix: Optional prefix for field names (defaults to series.field)

    Returns:
        Updated context dictionary
    """
    features = extract_features(series, window=window)
    prefix = prefix or series.field

    # Add basic features
    if features.last_value is not None:
        context_fields[f"{prefix}_last"] = features.last_value
    if features.first_value is not None:
        context_fields[f"{prefix}_first"] = features.first_value
    if features.min_value is not None:
        context_fields[f"{prefix}_min"] = features.min_value
    if features.max_value is not None:
        context_fields[f"{prefix}_max"] = features.max_value
    if features.mean_value is not None:
        context_fields[f"{prefix}_mean"] = features.mean_value

    # Add trend features
    window_suffix = f"_{window}" if window else ""
    if features.trend_kind is not None:
        context_fields[f"{prefix}_trend{window_suffix}"] = features.trend_kind.value
    if features.slope is not None:
        context_fields[f"{prefix}_slope{window_suffix}"] = features.slope
    if features.slope_normalized is not None:
        context_fields[f"{prefix}_slope_pct{window_suffix}"] = features.slope_normalized
    if features.volatility is not None:
        context_fields[f"{prefix}_volatility{window_suffix}"] = features.volatility
    if features.acceleration is not None:
        context_fields[f"{prefix}_acceleration{window_suffix}"] = features.acceleration
    if features.r_squared is not None:
        context_fields[f"{prefix}_r2{window_suffix}"] = features.r_squared

    # Add derived features
    if features.growth_rate is not None:
        context_fields[f"{prefix}_growth_rate"] = features.growth_rate
    if features.cv is not None:
        context_fields[f"{prefix}_cv"] = features.cv

    return context_fields


def extract_all_timeseries_features(
    series_list: list[TimeSeries],
    windows: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Extract features from multiple time series.

    Args:
        series_list: List of time series to process
        windows: Window sizes to compute trends for (default: [3, 6, 12])

    Returns:
        Dictionary of all extracted features
    """
    windows = windows or [3, 6, 12]
    context = {}

    for series in series_list:
        # Add features for default (all points)
        add_timeseries_features_to_context(context, series)

        # Add features for each window
        for window in windows:
            if len(series.points) >= window:
                add_timeseries_features_to_context(context, series, window=window)

    return context


# =============================================================================
# Missing Data Handling
# =============================================================================


def interpolate_missing(
    series: TimeSeries,
    method: str = "linear",
) -> TimeSeries:
    """
    Interpolate missing points in a time series.

    Args:
        series: Series with potential gaps
        method: Interpolation method ("linear", "previous", "zero")

    Returns:
        New series with interpolated points marked
    """
    series.sort()

    if len(series.points) < 2:
        return series

    # Detect granularity and expected gap
    times = series.get_times()
    if len(times) < 2:
        return series

    # Calculate typical gap between points
    gaps = [times[i + 1].ordinal - times[i].ordinal for i in range(len(times) - 1)]
    typical_gap = statistics.median(gaps) if gaps else 1

    # Find and fill gaps
    new_points = []
    for i, point in enumerate(series.points):
        new_points.append(point)

        if i < len(series.points) - 1:
            next_point = series.points[i + 1]
            gap = next_point.time.ordinal - point.time.ordinal

            # If gap is significantly larger than typical, interpolate
            if gap > typical_gap * 1.5:
                num_missing = int(gap / typical_gap) - 1

                for j in range(1, num_missing + 1):
                    frac = j / (num_missing + 1)
                    interp_ordinal = point.time.ordinal + gap * frac

                    if method == "linear":
                        interp_value = point.value + (next_point.value - point.value) * frac
                    elif method == "previous":
                        interp_value = point.value
                    else:  # zero
                        interp_value = 0.0

                    interp_point = TimeSeriesPoint(
                        time=TimePoint(
                            raw=f"interpolated_{interp_ordinal}",
                            ordinal=interp_ordinal,
                            granularity=point.time.granularity,
                        ),
                        value=interp_value,
                        confidence=0.5,  # Lower confidence for interpolated
                        is_interpolated=True,
                    )
                    new_points.append(interp_point)

    return TimeSeries(
        field=series.field,
        points=new_points,
        unit=series.unit,
    ).sort()


def filter_outliers(
    series: TimeSeries,
    method: str = "iqr",
    threshold: float = 1.5,
) -> TimeSeries:
    """
    Filter outliers from a time series.

    Args:
        series: Series to filter
        method: "iqr" (interquartile range) or "zscore"
        threshold: Multiplier for outlier detection

    Returns:
        New series with outliers removed
    """
    if len(series.points) < 4:
        return series

    values = [p.value for p in series.points]

    if method == "iqr":
        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    else:  # zscore
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        lower = mean - threshold * std
        upper = mean + threshold * std

    filtered_points = [
        p for p in series.points
        if lower <= p.value <= upper
    ]

    return TimeSeries(
        field=series.field,
        points=filtered_points,
        unit=series.unit,
    )
