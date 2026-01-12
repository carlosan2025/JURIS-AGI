"""
Metrics for scoring solutions: accuracy, MDL (Minimum Description Length), robustness.

Also includes constraint helpers for synthesis pruning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple, TYPE_CHECKING
import numpy as np

from .types import Grid, ARCPair

if TYPE_CHECKING:
    from .types import ARCTask


@dataclass
class SolutionMetrics:
    """Comprehensive metrics for a candidate solution."""
    exact_match: bool
    pixel_accuracy: float  # 0.0 to 1.0
    dimension_match: bool
    palette_match: bool
    program_length: int  # MDL proxy
    robustness_score: float  # 0.0 to 1.0
    diff_summary: Dict[str, Any]


def compute_exact_match(predicted: Grid, expected: Grid) -> bool:
    """Check if two grids are exactly equal."""
    return predicted == expected


def compute_pixel_accuracy(predicted: Grid, expected: Grid) -> float:
    """
    Compute pixel-wise accuracy between grids.

    Returns 0.0 if dimensions don't match.
    """
    if predicted.shape != expected.shape:
        return 0.0

    matching = np.sum(predicted.data == expected.data)
    total = predicted.height * predicted.width
    return matching / total if total > 0 else 0.0


def compute_dimension_match(predicted: Grid, expected: Grid) -> bool:
    """Check if dimensions match."""
    return predicted.shape == expected.shape


def compute_palette_match(predicted: Grid, expected: Grid) -> bool:
    """Check if predicted palette is subset of expected."""
    return predicted.palette.issubset(expected.palette | {0})


def compute_grid_diff(predicted: Grid, expected: Grid) -> Dict[str, Any]:
    """
    Compute detailed diff between grids.

    Returns structured information about differences for refinement.
    """
    diff: Dict[str, Any] = {
        "dimension_match": predicted.shape == expected.shape,
        "diff_pixels": [],
        "extra_colors": [],
        "missing_colors": [],
    }

    if not diff["dimension_match"]:
        diff["predicted_shape"] = predicted.shape
        diff["expected_shape"] = expected.shape
        return diff

    # Find differing pixels
    for r in range(expected.height):
        for c in range(expected.width):
            if predicted[r, c] != expected[r, c]:
                diff["diff_pixels"].append({
                    "position": (r, c),
                    "predicted": int(predicted[r, c]),
                    "expected": int(expected[r, c]),
                })

    # Color differences
    diff["extra_colors"] = list(predicted.palette - expected.palette)
    diff["missing_colors"] = list(expected.palette - predicted.palette)

    return diff


def compute_mdl_score(program_length: int, base_weight: float = 0.1) -> float:
    """
    Compute MDL (Minimum Description Length) score.

    Lower is better. Penalizes longer programs.
    """
    return base_weight * program_length


def evaluate_on_pairs(
    program_fn,  # Callable[[Grid], Grid]
    pairs: List[ARCPair],
) -> Dict[str, Any]:
    """
    Evaluate a program on a list of input-output pairs.

    Returns aggregate metrics.
    """
    results = []

    for i, pair in enumerate(pairs):
        try:
            predicted = program_fn(pair.input)
            metrics = {
                "pair_index": i,
                "exact_match": compute_exact_match(predicted, pair.output),
                "pixel_accuracy": compute_pixel_accuracy(predicted, pair.output),
                "dimension_match": compute_dimension_match(predicted, pair.output),
                "diff": compute_grid_diff(predicted, pair.output),
                "error": None,
            }
        except Exception as e:
            metrics = {
                "pair_index": i,
                "exact_match": False,
                "pixel_accuracy": 0.0,
                "dimension_match": False,
                "diff": {},
                "error": str(e),
            }
        results.append(metrics)

    # Aggregate
    all_match = all(r["exact_match"] for r in results)
    avg_accuracy = np.mean([r["pixel_accuracy"] for r in results])
    num_errors = sum(1 for r in results if r["error"] is not None)

    return {
        "all_exact_match": all_match,
        "average_pixel_accuracy": float(avg_accuracy),
        "num_errors": num_errors,
        "pair_results": results,
    }


def score_solution(
    program_length: int,
    train_metrics: Dict[str, Any],
    robustness_score: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute overall solution score combining multiple factors.

    Higher is better.
    """
    if weights is None:
        weights = {
            "exact_match": 100.0,
            "pixel_accuracy": 10.0,
            "mdl": -0.1,  # Penalty for longer programs
            "robustness": 5.0,
        }

    score = 0.0

    # Hard requirement: exact match on all training pairs
    if train_metrics["all_exact_match"]:
        score += weights["exact_match"]
    else:
        score += weights["pixel_accuracy"] * train_metrics["average_pixel_accuracy"]

    # MDL penalty
    score += weights["mdl"] * program_length

    # Robustness bonus
    score += weights["robustness"] * robustness_score

    return score


# ============================================================================
# Constraint Helpers for Pruning
# ============================================================================

@dataclass
class DimensionConstraint:
    """Constraint on output dimensions."""
    fixed_dims: Optional[Tuple[int, int]] = None  # Exact dims if known
    h_ratio: Optional[float] = None  # Height ratio to input
    w_ratio: Optional[float] = None  # Width ratio to input
    max_height: int = 100  # ARC limit
    max_width: int = 100  # ARC limit
    min_height: int = 1
    min_width: int = 1

    def check(self, output: Grid, input_grid: Optional[Grid] = None) -> bool:
        """Check if output satisfies dimension constraints."""
        # Size bounds
        if output.height > self.max_height or output.width > self.max_width:
            return False
        if output.height < self.min_height or output.width < self.min_width:
            return False

        # Fixed dimensions
        if self.fixed_dims is not None:
            if output.shape != self.fixed_dims:
                return False

        # Ratio constraints
        if input_grid is not None:
            if self.h_ratio is not None:
                expected_h = int(input_grid.height * self.h_ratio)
                if output.height != expected_h:
                    return False
            if self.w_ratio is not None:
                expected_w = int(input_grid.width * self.w_ratio)
                if output.width != expected_w:
                    return False

        return True

    def violation_severity(self, output: Grid, input_grid: Optional[Grid] = None) -> float:
        """
        Compute how severely the constraint is violated (0.0 = satisfied, 1.0 = max violation).
        """
        if self.check(output, input_grid):
            return 0.0

        severity = 0.0

        # Size bounds violation
        if output.height > self.max_height:
            severity = max(severity, (output.height - self.max_height) / self.max_height)
        if output.width > self.max_width:
            severity = max(severity, (output.width - self.max_width) / self.max_width)
        if output.height < self.min_height or output.width < self.min_width:
            severity = max(severity, 1.0)

        # Fixed dimension violation
        if self.fixed_dims is not None:
            h_diff = abs(output.height - self.fixed_dims[0]) / max(self.fixed_dims[0], 1)
            w_diff = abs(output.width - self.fixed_dims[1]) / max(self.fixed_dims[1], 1)
            severity = max(severity, (h_diff + w_diff) / 2)

        return min(severity, 1.0)


@dataclass
class PaletteConstraint:
    """Constraint on output color palette."""
    allowed_colors: Optional[Set[int]] = None  # If set, output must be subset
    required_colors: Optional[Set[int]] = None  # If set, output must include these
    forbidden_colors: Optional[Set[int]] = None  # If set, output must not include these
    max_colors: int = 10  # ARC has 10 colors (0-9)

    def check(self, output: Grid) -> bool:
        """Check if output satisfies palette constraints."""
        output_palette = output.palette

        # Max colors
        if len(output_palette) > self.max_colors:
            return False

        # Allowed colors (output must be subset)
        if self.allowed_colors is not None:
            if not output_palette.issubset(self.allowed_colors):
                return False

        # Required colors
        if self.required_colors is not None:
            if not self.required_colors.issubset(output_palette):
                return False

        # Forbidden colors
        if self.forbidden_colors is not None:
            if output_palette & self.forbidden_colors:
                return False

        return True

    def violation_severity(self, output: Grid) -> float:
        """Compute severity of palette constraint violation."""
        if self.check(output):
            return 0.0

        output_palette = output.palette
        severity = 0.0

        # Allowed colors violation
        if self.allowed_colors is not None:
            extra = output_palette - self.allowed_colors
            if extra:
                severity = max(severity, len(extra) / max(len(output_palette), 1))

        # Required colors violation
        if self.required_colors is not None:
            missing = self.required_colors - output_palette
            if missing:
                severity = max(severity, len(missing) / len(self.required_colors))

        # Forbidden colors violation
        if self.forbidden_colors is not None:
            forbidden_present = output_palette & self.forbidden_colors
            if forbidden_present:
                severity = max(severity, len(forbidden_present) / max(len(output_palette), 1))

        return min(severity, 1.0)


@dataclass
class ObjectCountConstraint:
    """Soft constraint on object counts (advisory, not hard veto)."""
    expected_count: Optional[int] = None
    min_count: int = 0
    max_count: int = 100
    count_delta: Optional[int] = None  # Expected change from input

    def check(self, output: Grid, input_grid: Optional[Grid] = None) -> Tuple[bool, float]:
        """
        Check object count constraint.

        Returns (satisfied, confidence) where confidence indicates how soft the constraint is.
        """
        from ..representation.objects import extract_connected_objects

        try:
            output_objs = extract_connected_objects(output)
            output_count = len(output_objs)

            # Hard bounds
            if output_count < self.min_count or output_count > self.max_count:
                return False, 1.0  # Hard violation

            # Expected count (soft)
            if self.expected_count is not None:
                if output_count != self.expected_count:
                    # Soft violation - closer counts are less severe
                    diff = abs(output_count - self.expected_count)
                    severity = min(diff / max(self.expected_count, 1), 1.0)
                    return False, 0.5 * severity  # Low confidence violation

            # Count delta (soft)
            if self.count_delta is not None and input_grid is not None:
                input_objs = extract_connected_objects(input_grid)
                input_count = len(input_objs)
                expected_output = input_count + self.count_delta
                if output_count != expected_output:
                    diff = abs(output_count - expected_output)
                    severity = min(diff / max(expected_output, 1), 1.0)
                    return False, 0.3 * severity  # Very soft violation

            return True, 1.0

        except Exception:
            # If we can't check, don't penalize
            return True, 0.5


@dataclass
class ConstraintSet:
    """Combined constraint set for pruning."""
    dimension: Optional[DimensionConstraint] = None
    palette: Optional[PaletteConstraint] = None
    object_count: Optional[ObjectCountConstraint] = None

    def check_all(
        self,
        output: Grid,
        input_grid: Optional[Grid] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check all constraints.

        Returns (all_satisfied, details) where details contains per-constraint info.
        """
        details: Dict[str, Any] = {
            "dimension_ok": True,
            "palette_ok": True,
            "object_count_ok": True,
            "dimension_severity": 0.0,
            "palette_severity": 0.0,
            "object_count_severity": 0.0,
        }

        all_ok = True

        # Dimension constraint (hard)
        if self.dimension is not None:
            dim_ok = self.dimension.check(output, input_grid)
            details["dimension_ok"] = dim_ok
            details["dimension_severity"] = self.dimension.violation_severity(output, input_grid)
            if not dim_ok:
                all_ok = False

        # Palette constraint (hard)
        if self.palette is not None:
            pal_ok = self.palette.check(output)
            details["palette_ok"] = pal_ok
            details["palette_severity"] = self.palette.violation_severity(output)
            if not pal_ok:
                all_ok = False

        # Object count constraint (soft)
        if self.object_count is not None:
            obj_ok, confidence = self.object_count.check(output, input_grid)
            details["object_count_ok"] = obj_ok
            details["object_count_confidence"] = confidence
            # Don't fail overall for soft constraint
            if not obj_ok:
                details["object_count_severity"] = 1.0 - confidence

        return all_ok, details

    def compute_pruning_score(
        self,
        output: Grid,
        input_grid: Optional[Grid] = None,
    ) -> float:
        """
        Compute a score for pruning decisions.

        Lower score = more constraint violations = should prune.
        Returns 1.0 for fully satisfied, 0.0 for fully violated.
        """
        _, details = self.check_all(output, input_grid)

        # Weighted combination of constraint satisfaction
        score = 1.0

        # Hard constraints have high weight
        if not details["dimension_ok"]:
            score -= 0.4 * (1.0 + details["dimension_severity"])
        if not details["palette_ok"]:
            score -= 0.3 * (1.0 + details["palette_severity"])

        # Soft constraint has lower weight
        if not details.get("object_count_ok", True):
            score -= 0.1 * details.get("object_count_severity", 0.0)

        return max(0.0, score)


def extract_constraints_from_task(task: 'ARCTask') -> ConstraintSet:
    """
    Extract constraints from a task's training examples.

    Analyzes patterns across examples to determine what constraints apply.
    """
    from ..representation.features import compute_task_features

    features = compute_task_features(task)

    # Build dimension constraint
    dim_constraint = DimensionConstraint()
    if features.fixed_output_dims is not None:
        dim_constraint.fixed_dims = features.fixed_output_dims
    if features.dimension_ratio is not None:
        dim_constraint.h_ratio, dim_constraint.w_ratio = features.dimension_ratio

    # Build palette constraint
    pal_constraint = PaletteConstraint()
    # Allow input palette + output palette + background
    pal_constraint.allowed_colors = features.input_palette | features.output_palette | {0}

    # Build object count constraint (if consistent)
    obj_constraint = None
    if features.object_count_delta is not None:
        obj_constraint = ObjectCountConstraint(count_delta=features.object_count_delta)

    return ConstraintSet(
        dimension=dim_constraint,
        palette=pal_constraint,
        object_count=obj_constraint,
    )


def fast_dimension_check(
    output: Grid,
    expected_dims: Tuple[int, int],
    tolerance: float = 0.0,
) -> bool:
    """Fast check if output dimensions match expected (with optional tolerance)."""
    if tolerance == 0.0:
        return output.shape == expected_dims
    h_ok = abs(output.height - expected_dims[0]) <= tolerance * expected_dims[0]
    w_ok = abs(output.width - expected_dims[1]) <= tolerance * expected_dims[1]
    return h_ok and w_ok


def fast_palette_check(
    output: Grid,
    allowed: Set[int],
) -> bool:
    """Fast check if output palette is subset of allowed colors."""
    return output.palette.issubset(allowed)
