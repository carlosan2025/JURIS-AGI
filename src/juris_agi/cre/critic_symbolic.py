"""
Symbolic Critic - verifies programs and computes diffs.

The symbolic critic has JURISDICTION - it can veto any solution that
doesn't pass symbolic verification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from ..core.types import Grid, ARCPair, ARCTask
from ..core.metrics import compute_exact_match, compute_pixel_accuracy, compute_grid_diff
from ..dsl.ast import ASTNode
from ..dsl.interpreter import make_program


@dataclass
class DiffEntry:
    """A single difference between predicted and expected."""
    position: Tuple[int, int]
    predicted: int
    expected: int
    diff_type: str  # "wrong_color", "extra_pixel", "missing_pixel"


@dataclass
class SymbolicDiff:
    """Detailed symbolic diff between two grids."""
    dimension_match: bool
    predicted_shape: Tuple[int, int]
    expected_shape: Tuple[int, int]
    exact_match: bool
    pixel_accuracy: float
    diff_entries: List[DiffEntry] = field(default_factory=list)
    extra_colors: Set[int] = field(default_factory=set)
    missing_colors: Set[int] = field(default_factory=set)
    diff_regions: List[Dict[str, Any]] = field(default_factory=list)
    # Extracted invariants
    output_object_count: Optional[int] = None
    input_object_count: Optional[int] = None

    @property
    def num_errors(self) -> int:
        return len(self.diff_entries)

    @property
    def severity(self) -> float:
        """Compute diff severity (0=perfect, 1=completely wrong)."""
        if self.exact_match:
            return 0.0
        if not self.dimension_match:
            return 1.0
        return 1.0 - self.pixel_accuracy


@dataclass
class CriticResult:
    """Result from symbolic critic evaluation."""
    approved: bool  # Did the program pass verification?
    exact_match_all: bool
    pair_results: List[Dict[str, Any]]
    diffs: List[SymbolicDiff]
    invariants_satisfied: List[str]
    invariants_violated: List[str]
    veto_reason: Optional[str] = None

    @property
    def is_certified(self) -> bool:
        """A solution is certified if approved and exact match on all pairs."""
        return self.approved and self.exact_match_all


class SymbolicCritic:
    """
    Symbolic critic for program verification.

    Has JURISDICTION: can veto solutions that fail symbolic checks.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize critic.

        Args:
            strict_mode: If True, require exact match for approval
        """
        self.strict_mode = strict_mode

    def evaluate(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> CriticResult:
        """
        Evaluate a program on a task.

        Returns detailed critique including diffs and invariant checks.
        """
        try:
            program_fn = make_program(program)
        except Exception as e:
            return CriticResult(
                approved=False,
                exact_match_all=False,
                pair_results=[],
                diffs=[],
                invariants_satisfied=[],
                invariants_violated=["program_execution"],
                veto_reason=f"Program failed to compile: {e}",
            )

        pair_results = []
        diffs = []
        all_exact = True

        # Evaluate on training pairs
        for i, pair in enumerate(task.train):
            try:
                predicted = program_fn(pair.input)
                diff = compute_symbolic_diff(predicted, pair.output)
                diffs.append(diff)

                exact = diff.exact_match
                if not exact:
                    all_exact = False

                pair_results.append({
                    "pair_index": i,
                    "pair_type": "train",
                    "exact_match": exact,
                    "pixel_accuracy": diff.pixel_accuracy,
                    "dimension_match": diff.dimension_match,
                })
            except Exception as e:
                all_exact = False
                pair_results.append({
                    "pair_index": i,
                    "pair_type": "train",
                    "exact_match": False,
                    "error": str(e),
                })
                diffs.append(SymbolicDiff(
                    dimension_match=False,
                    predicted_shape=(0, 0),
                    expected_shape=pair.output.shape,
                    exact_match=False,
                    pixel_accuracy=0.0,
                ))

        # Check invariants
        invariants_satisfied, invariants_violated = self._check_invariants(
            program_fn, task, pair_results
        )

        # Determine approval
        approved = all_exact
        veto_reason = None

        if not approved:
            if not all_exact:
                veto_reason = "Not all training pairs matched exactly"
            elif invariants_violated:
                veto_reason = f"Invariants violated: {invariants_violated}"

        return CriticResult(
            approved=approved,
            exact_match_all=all_exact,
            pair_results=pair_results,
            diffs=diffs,
            invariants_satisfied=invariants_satisfied,
            invariants_violated=invariants_violated,
            veto_reason=veto_reason,
        )

    def _check_invariants(
        self,
        program_fn,
        task: ARCTask,
        pair_results: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """Check various invariants that a valid solution should satisfy."""
        satisfied = []
        violated = []

        # Invariant: Output dimensions consistent across training pairs
        if self._check_dimension_consistency(task, program_fn):
            satisfied.append("dimension_consistency")
        else:
            violated.append("dimension_consistency")

        # Invariant: Color palette consistency
        if self._check_palette_consistency(task, program_fn):
            satisfied.append("palette_consistency")
        else:
            violated.append("palette_consistency")

        # Invariant: Determinism (same input -> same output)
        if self._check_determinism(task, program_fn):
            satisfied.append("determinism")
        else:
            violated.append("determinism")

        # Invariant: Object count consistency (if applicable)
        if self._check_object_count_consistency(task, program_fn):
            satisfied.append("object_count_consistency")
        # Note: object count may legitimately change, so don't add to violated

        return satisfied, violated

    def _check_object_count_consistency(
        self,
        task: ARCTask,
        program_fn,
    ) -> bool:
        """Check if object count relationship is consistent across pairs."""
        try:
            from ..representation.objects import extract_connected_objects

            count_diffs = []
            for pair in task.train:
                output = program_fn(pair.input)
                input_objs = len(extract_connected_objects(pair.input))
                output_objs = len(extract_connected_objects(output))
                expected_objs = len(extract_connected_objects(pair.output))

                # Check if our output matches expected object count
                if output_objs != expected_objs:
                    return False
                count_diffs.append(output_objs - input_objs)

            # Check consistency of count differences
            if len(count_diffs) > 1:
                return all(d == count_diffs[0] for d in count_diffs)
            return True
        except Exception:
            return True  # If we can't check, don't penalize

    def _check_dimension_consistency(
        self,
        task: ARCTask,
        program_fn,
    ) -> bool:
        """Check if output dimensions follow a consistent pattern."""
        # Compute dimension ratios
        ratios = []
        for pair in task.train:
            try:
                output = program_fn(pair.input)
                h_ratio = output.height / pair.input.height if pair.input.height > 0 else 0
                w_ratio = output.width / pair.input.width if pair.input.width > 0 else 0
                ratios.append((h_ratio, w_ratio))
            except Exception:
                return False

        # Check if ratios are consistent
        if len(ratios) <= 1:
            return True

        first = ratios[0]
        return all(abs(r[0] - first[0]) < 0.01 and abs(r[1] - first[1]) < 0.01 for r in ratios)

    def _check_palette_consistency(
        self,
        task: ARCTask,
        program_fn,
    ) -> bool:
        """Check if program preserves palette appropriately."""
        for pair in task.train:
            try:
                output = program_fn(pair.input)
                # Check that output palette is subset of combined input/expected palette
                allowed = pair.input.palette | pair.output.palette | {0}
                if not output.palette.issubset(allowed):
                    return False
            except Exception:
                return False
        return True

    def _check_determinism(
        self,
        task: ARCTask,
        program_fn,
    ) -> bool:
        """Check that program is deterministic."""
        for pair in task.train:
            try:
                out1 = program_fn(pair.input)
                out2 = program_fn(pair.input)
                if out1 != out2:
                    return False
            except Exception:
                return False
        return True

    def compute_refinement_hints(
        self,
        diffs: List[SymbolicDiff],
    ) -> List[Dict[str, Any]]:
        """
        Analyze diffs to provide hints for refinement.

        Returns structured hints that the refinement engine can use.
        """
        hints = []

        for i, diff in enumerate(diffs):
            if diff.exact_match:
                continue

            hint: Dict[str, Any] = {"pair_index": i}

            # Dimension mismatch hint
            if not diff.dimension_match:
                hint["dimension_hint"] = {
                    "predicted": diff.predicted_shape,
                    "expected": diff.expected_shape,
                    "suggestion": self._suggest_dimension_fix(
                        diff.predicted_shape, diff.expected_shape
                    ),
                }

            # Color error hints
            if diff.extra_colors:
                hint["extra_colors"] = list(diff.extra_colors)
            if diff.missing_colors:
                hint["missing_colors"] = list(diff.missing_colors)

            # Localized error hints
            if diff.diff_entries:
                error_locations = [(e.position, e.diff_type) for e in diff.diff_entries[:10]]
                hint["error_locations"] = error_locations

                # Try to detect patterns in errors
                error_pattern = self._detect_error_pattern(diff.diff_entries)
                if error_pattern:
                    hint["error_pattern"] = error_pattern

            hints.append(hint)

        return hints

    def _suggest_dimension_fix(
        self,
        predicted: Tuple[int, int],
        expected: Tuple[int, int],
    ) -> str:
        """Suggest how to fix dimension mismatch."""
        ph, pw = predicted
        eh, ew = expected

        if ph > eh and pw > ew:
            return "Try cropping output"
        elif ph < eh and pw < ew:
            factor_h = eh / ph if ph > 0 else 0
            factor_w = ew / pw if pw > 0 else 0
            if abs(factor_h - factor_w) < 0.1 and factor_h == int(factor_h):
                return f"Try scaling by {int(factor_h)}"
            return "Try padding or tiling"
        elif ph == ew and pw == eh:
            return "Try transposing"
        else:
            return "Dimension relationship unclear"

    def _detect_error_pattern(
        self,
        diff_entries: List[DiffEntry],
    ) -> Optional[str]:
        """Detect patterns in error locations."""
        if not diff_entries:
            return None

        # Check if errors form a row/column pattern
        rows = set(e.position[0] for e in diff_entries)
        cols = set(e.position[1] for e in diff_entries)

        if len(rows) == 1:
            return f"Errors concentrated in row {list(rows)[0]}"
        if len(cols) == 1:
            return f"Errors concentrated in col {list(cols)[0]}"

        # Check if errors are all same type
        types = set(e.diff_type for e in diff_entries)
        if len(types) == 1:
            return f"All errors are {list(types)[0]}"

        return None


def compute_symbolic_diff(
    predicted: Grid,
    expected: Grid,
    input_grid: Optional[Grid] = None,
) -> SymbolicDiff:
    """Compute detailed symbolic diff between two grids."""
    dim_match = predicted.shape == expected.shape

    diff = SymbolicDiff(
        dimension_match=dim_match,
        predicted_shape=predicted.shape,
        expected_shape=expected.shape,
        exact_match=predicted == expected,
        pixel_accuracy=compute_pixel_accuracy(predicted, expected),
    )

    # Try to extract object counts
    try:
        from ..representation.objects import extract_connected_objects
        diff.output_object_count = len(extract_connected_objects(predicted))
        if input_grid is not None:
            diff.input_object_count = len(extract_connected_objects(input_grid))
    except Exception:
        pass

    if not dim_match:
        return diff

    # Compute pixel-level diffs
    for r in range(expected.height):
        for c in range(expected.width):
            pred_val = predicted[r, c]
            exp_val = expected[r, c]
            if pred_val != exp_val:
                if exp_val == 0:
                    diff_type = "extra_pixel"
                elif pred_val == 0:
                    diff_type = "missing_pixel"
                else:
                    diff_type = "wrong_color"

                diff.diff_entries.append(DiffEntry(
                    position=(r, c),
                    predicted=int(pred_val),
                    expected=int(exp_val),
                    diff_type=diff_type,
                ))

    # Color analysis
    diff.extra_colors = predicted.palette - expected.palette
    diff.missing_colors = expected.palette - predicted.palette

    return diff


def extract_invariants(task: ARCTask) -> Dict[str, Any]:
    """Extract invariants from a task's training examples."""
    invariants: Dict[str, Any] = {}

    if not task.train:
        return invariants

    # Output dimension invariants
    output_dims = [pair.output.shape for pair in task.train]
    if all(d == output_dims[0] for d in output_dims):
        invariants["fixed_output_dims"] = output_dims[0]

    # Dimension ratio invariants
    ratios = []
    for pair in task.train:
        if pair.input.height > 0 and pair.input.width > 0:
            h_ratio = pair.output.height / pair.input.height
            w_ratio = pair.output.width / pair.input.width
            ratios.append((h_ratio, w_ratio))
    if ratios and all(r == ratios[0] for r in ratios):
        invariants["dimension_ratio"] = ratios[0]

    # Palette invariants
    all_input_colors = set()
    all_output_colors = set()
    for pair in task.train:
        all_input_colors |= pair.input.palette
        all_output_colors |= pair.output.palette

    invariants["input_palette"] = all_input_colors
    invariants["output_palette"] = all_output_colors
    invariants["palette_preserved"] = all_output_colors.issubset(all_input_colors | {0})

    # Object count invariants
    try:
        from ..representation.objects import extract_connected_objects
        obj_count_diffs = []
        for pair in task.train:
            in_count = len(extract_connected_objects(pair.input))
            out_count = len(extract_connected_objects(pair.output))
            obj_count_diffs.append(out_count - in_count)

        if all(d == obj_count_diffs[0] for d in obj_count_diffs):
            invariants["object_count_delta"] = obj_count_diffs[0]
    except Exception:
        pass

    return invariants
