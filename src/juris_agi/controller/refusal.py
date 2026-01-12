"""
Refusal checker for safety and validity.

Determines when to refuse solving a task.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict, Any

from ..core.types import ARCTask


class RefusalReason(Enum):
    """Reasons for refusing to solve a task."""
    INVALID_FORMAT = auto()
    MISSING_DATA = auto()
    EXCESSIVE_SIZE = auto()
    TIMEOUT_EXPECTED = auto()
    CONSTRAINT_VIOLATION = auto()
    SAFETY_CONCERN = auto()


@dataclass
class RefusalDecision:
    """Decision about whether to refuse a task."""
    should_refuse: bool
    reason: Optional[RefusalReason] = None
    explanation: str = ""
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class RefusalChecker:
    """
    Checks whether a task should be refused.

    Validates task format and constraints before attempting solution.
    """

    def __init__(
        self,
        max_grid_size: int = 30,
        max_train_pairs: int = 10,
        max_test_pairs: int = 5,
    ):
        self.max_grid_size = max_grid_size
        self.max_train_pairs = max_train_pairs
        self.max_test_pairs = max_test_pairs

    def check(self, task: ARCTask) -> RefusalDecision:
        """
        Check if a task should be refused.

        Returns RefusalDecision with details if refusing.
        """
        # Check for missing data
        if not task.train:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.MISSING_DATA,
                explanation="Task has no training examples",
                suggestions=["Provide at least one training pair"],
            )

        if not task.test:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.MISSING_DATA,
                explanation="Task has no test examples",
                suggestions=["Provide at least one test pair"],
            )

        # Check number of pairs
        if len(task.train) > self.max_train_pairs:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.EXCESSIVE_SIZE,
                explanation=f"Too many training pairs ({len(task.train)} > {self.max_train_pairs})",
                suggestions=["Reduce number of training pairs"],
            )

        if len(task.test) > self.max_test_pairs:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.EXCESSIVE_SIZE,
                explanation=f"Too many test pairs ({len(task.test)} > {self.max_test_pairs})",
                suggestions=["Reduce number of test pairs"],
            )

        # Check grid sizes
        for i, pair in enumerate(task.train):
            size_check = self._check_grid_size(pair.input, f"train[{i}].input")
            if size_check.should_refuse:
                return size_check

            size_check = self._check_grid_size(pair.output, f"train[{i}].output")
            if size_check.should_refuse:
                return size_check

        for i, pair in enumerate(task.test):
            size_check = self._check_grid_size(pair.input, f"test[{i}].input")
            if size_check.should_refuse:
                return size_check

        # Check for valid color values
        for i, pair in enumerate(task.train):
            color_check = self._check_colors(pair.input, f"train[{i}].input")
            if color_check.should_refuse:
                return color_check

            color_check = self._check_colors(pair.output, f"train[{i}].output")
            if color_check.should_refuse:
                return color_check

        # All checks passed
        return RefusalDecision(
            should_refuse=False,
            explanation="Task passed all validation checks",
        )

    def _check_grid_size(self, grid: Any, name: str) -> RefusalDecision:
        """Check if grid size is within limits."""
        if grid.height > self.max_grid_size or grid.width > self.max_grid_size:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.EXCESSIVE_SIZE,
                explanation=f"{name} is too large ({grid.height}x{grid.width} > {self.max_grid_size}x{self.max_grid_size})",
                suggestions=["Reduce grid dimensions"],
            )

        if grid.height == 0 or grid.width == 0:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.INVALID_FORMAT,
                explanation=f"{name} has zero dimension ({grid.height}x{grid.width})",
                suggestions=["Provide non-empty grids"],
            )

        return RefusalDecision(should_refuse=False)

    def _check_colors(self, grid: Any, name: str) -> RefusalDecision:
        """Check if all colors are valid (0-9)."""
        for color in grid.palette:
            if color < 0 or color > 9:
                return RefusalDecision(
                    should_refuse=True,
                    reason=RefusalReason.INVALID_FORMAT,
                    explanation=f"{name} contains invalid color {color} (must be 0-9)",
                    suggestions=["Use only colors 0-9"],
                )

        return RefusalDecision(should_refuse=False)

    def estimate_difficulty(self, task: ARCTask) -> Dict[str, Any]:
        """
        Estimate task difficulty.

        Returns metrics that help predict solvability.
        """
        if not task.train:
            return {"difficulty": "unknown", "solvable_estimate": 0.0}

        metrics: Dict[str, Any] = {}

        # Grid complexity
        avg_input_size = sum(
            p.input.height * p.input.width for p in task.train
        ) / len(task.train)
        avg_output_size = sum(
            p.output.height * p.output.width for p in task.train
        ) / len(task.train)

        metrics["avg_input_size"] = avg_input_size
        metrics["avg_output_size"] = avg_output_size

        # Color complexity
        all_colors = set()
        for pair in task.train:
            all_colors |= pair.input.palette
            all_colors |= pair.output.palette
        metrics["num_colors"] = len(all_colors)

        # Dimension change complexity
        same_dims = all(
            p.input.shape == p.output.shape for p in task.train
        )
        metrics["same_dimensions"] = same_dims

        # Estimate difficulty
        difficulty_score = 0.0
        if not same_dims:
            difficulty_score += 0.3
        if len(all_colors) > 5:
            difficulty_score += 0.2
        if avg_input_size > 100:
            difficulty_score += 0.2
        if len(task.train) < 3:
            difficulty_score += 0.2

        if difficulty_score < 0.3:
            metrics["difficulty"] = "easy"
        elif difficulty_score < 0.6:
            metrics["difficulty"] = "medium"
        else:
            metrics["difficulty"] = "hard"

        metrics["solvable_estimate"] = max(0.0, 1.0 - difficulty_score)

        return metrics


def validate_task(task: ARCTask) -> RefusalDecision:
    """Convenience function to validate a task."""
    checker = RefusalChecker()
    return checker.check(task)


# ============================================================================
# Uncertainty-Based Refusal and Output
# ============================================================================

@dataclass
class UncertaintyOutput:
    """
    Output when no consistent program is found within budget.

    Provides structured uncertainty information rather than just failing.
    """
    task_id: str
    resolved: bool  # True if a solution was found
    best_program: Optional[str]  # Best program found (may not be exact match)
    best_score: float  # Score of best program (0-100)

    # Uncertainty decomposition
    epistemic_uncertainty: float  # Reducible uncertainty (more search might help)
    aleatoric_uncertainty: float  # Irreducible uncertainty (from WME confidence)
    total_uncertainty: float

    # Details
    num_consistent_hypotheses: int  # Number of programs that partially fit
    diff_variance: float  # Variance in diff scores across training pairs
    budget_exhausted: bool
    budget_summary: Dict[str, Any]

    # Reasoning
    refusal_reason: Optional[str]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resolved": self.resolved,
            "best_program": self.best_program,
            "best_score": self.best_score,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "num_consistent_hypotheses": self.num_consistent_hypotheses,
            "diff_variance": self.diff_variance,
            "budget_exhausted": self.budget_exhausted,
            "budget_summary": self.budget_summary,
            "refusal_reason": self.refusal_reason,
            "suggestions": self.suggestions,
        }

    @property
    def is_confident_refusal(self) -> bool:
        """
        Whether this is a confident refusal (high certainty no solution exists).

        Returns True if epistemic uncertainty is low but no solution found.
        This means more search is unlikely to help.
        """
        return (
            not self.resolved
            and self.epistemic_uncertainty < 0.3
            and self.budget_exhausted
        )

    @property
    def should_retry(self) -> bool:
        """
        Whether retrying with more budget might help.

        Returns True if epistemic uncertainty is high.
        """
        return (
            not self.resolved
            and self.epistemic_uncertainty > 0.5
            and not self.budget_exhausted
        )


def compute_uncertainty_output(
    task_id: str,
    best_program: Optional[str],
    best_score: float,
    candidates: List[Dict[str, Any]],
    wme_confidence: float,
    budget_summary: Dict[str, Any],
) -> UncertaintyOutput:
    """
    Compute structured uncertainty output when no exact solution found.

    Args:
        task_id: Task identifier
        best_program: Best program source found
        best_score: Score of best program
        candidates: List of candidate programs with scores
        wme_confidence: WME confidence in task understanding
        budget_summary: Budget usage summary

    Returns:
        UncertaintyOutput with decomposed uncertainty
    """
    # Count consistent hypotheses (programs with score > threshold)
    consistent_threshold = 30.0
    consistent_hypotheses = [c for c in candidates if c.get("score", 0) > consistent_threshold]
    num_consistent = len(consistent_hypotheses)

    # Compute diff variance
    scores = [c.get("score", 0) for c in candidates]
    diff_variance = _compute_variance(scores) if scores else 0.0

    # Compute epistemic uncertainty
    # High when: few candidates, low scores, budget not exhausted
    if num_consistent == 0:
        epistemic = 0.9  # Very high - barely explored
    elif num_consistent < 5:
        epistemic = 0.7 - (best_score / 100) * 0.3  # Moderate
    elif num_consistent < 20:
        epistemic = 0.5 - (best_score / 100) * 0.2  # Some exploration done
    else:
        epistemic = max(0.1, 0.3 - (best_score / 100) * 0.2)  # Well explored

    # Budget exhaustion reduces epistemic uncertainty
    # (we've done what we can)
    budget_exhausted = budget_summary.get("remaining_time", 0) <= 0
    if budget_exhausted:
        epistemic = min(epistemic, 0.3)

    # Compute aleatoric uncertainty from WME confidence
    # High when WME is uncertain about task interpretation
    aleatoric = 1.0 - wme_confidence

    # High variance in candidate scores suggests multiple valid interpretations
    if diff_variance > 200:
        aleatoric = min(1.0, aleatoric + 0.2)

    # Total uncertainty
    total = min(1.0, (epistemic + aleatoric) / 2)

    # Generate suggestions
    suggestions = []
    if epistemic > 0.5:
        suggestions.append("Increase synthesis budget or depth")
    if aleatoric > 0.5:
        suggestions.append("Task may have multiple valid interpretations")
    if num_consistent > 5 and best_score < 100:
        suggestions.append("Multiple near-solutions exist - consider refinement")
    if diff_variance > 200:
        suggestions.append("High variance in candidates - task pattern unclear")

    # Determine refusal reason
    refusal_reason = None
    if not best_program:
        refusal_reason = "No valid program found"
    elif best_score < consistent_threshold:
        refusal_reason = "Best program score below threshold"
    elif budget_exhausted and best_score < 100:
        refusal_reason = "Budget exhausted without exact solution"

    return UncertaintyOutput(
        task_id=task_id,
        resolved=best_score >= 100.0,
        best_program=best_program,
        best_score=best_score,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=aleatoric,
        total_uncertainty=total,
        num_consistent_hypotheses=num_consistent,
        diff_variance=diff_variance,
        budget_exhausted=budget_exhausted,
        budget_summary=budget_summary,
        refusal_reason=refusal_reason,
        suggestions=suggestions,
    )


def _compute_variance(values: List[float]) -> float:
    """Compute variance of a list of values."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance
