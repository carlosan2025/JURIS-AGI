"""
Robustness checking for programs.

Tests programs against counterfactual inputs to assess generalization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from ..core.types import Grid, ARCTask
from ..dsl.ast import ASTNode
from ..dsl.interpreter import make_program
from .counterfactuals import (
    Counterfactual,
    CounterfactualGenerator,
    GridPerturbationGenerator,
    StructuralCounterfactualGenerator,
)


@dataclass
class RobustnessTestResult:
    """Result of a single robustness test."""
    counterfactual: Counterfactual
    success: bool  # Did the program run without error?
    output: Optional[Grid] = None
    error: Optional[str] = None
    consistency_score: float = 0.0  # How consistent with expected behavior?


@dataclass
class RobustnessResult:
    """Aggregate robustness result."""
    overall_score: float  # 0.0 to 1.0
    num_tests: int
    num_passed: int
    num_failed: int
    num_errors: int
    test_results: List[RobustnessTestResult] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)


class RobustnessChecker:
    """
    Checks program robustness against perturbations.

    Tests that programs generalize beyond exact training examples.
    """

    def __init__(
        self,
        generators: Optional[List[CounterfactualGenerator]] = None,
        num_tests_per_generator: int = 5,
    ):
        if generators is None:
            generators = [
                GridPerturbationGenerator(),
                StructuralCounterfactualGenerator(),
            ]
        self.generators = generators
        self.num_tests_per_generator = num_tests_per_generator

    def check_robustness(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> RobustnessResult:
        """
        Check program robustness on a task.

        Tests program on counterfactual inputs derived from training data.
        """
        try:
            program_fn = make_program(program)
        except Exception as e:
            return RobustnessResult(
                overall_score=0.0,
                num_tests=0,
                num_passed=0,
                num_failed=0,
                num_errors=1,
                analysis={"error": f"Program compilation failed: {e}"},
            )

        all_results: List[RobustnessTestResult] = []

        # Generate and test counterfactuals for each training input
        for pair in task.train:
            for generator in self.generators:
                counterfactuals = generator.generate(
                    pair.input,
                    num_counterfactuals=self.num_tests_per_generator,
                )

                for cf in counterfactuals:
                    result = self._test_counterfactual(program_fn, cf, pair.output)
                    all_results.append(result)

        # Compute aggregate statistics
        num_tests = len(all_results)
        num_passed = sum(1 for r in all_results if r.success and r.consistency_score > 0.5)
        num_failed = sum(1 for r in all_results if r.success and r.consistency_score <= 0.5)
        num_errors = sum(1 for r in all_results if not r.success)

        if num_tests > 0:
            overall_score = sum(r.consistency_score for r in all_results) / num_tests
        else:
            overall_score = 0.0

        # Analyze failure patterns
        analysis = self._analyze_failures(all_results)

        return RobustnessResult(
            overall_score=overall_score,
            num_tests=num_tests,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
            test_results=all_results,
            analysis=analysis,
        )

    def _test_counterfactual(
        self,
        program_fn: Callable[[Grid], Grid],
        cf: Counterfactual,
        original_output: Grid,
    ) -> RobustnessTestResult:
        """Test program on a single counterfactual."""
        try:
            output = program_fn(cf.modified)

            # Compute consistency score based on expected behavior
            if cf.expected_behavior == "same_transformation":
                # For same_transformation, we expect similar output structure
                consistency = self._compute_structural_similarity(output, original_output)
            elif cf.expected_behavior == "same_structure":
                # Output structure should be preserved even if colors differ
                consistency = self._compute_shape_similarity(output, original_output)
            else:
                # May differ - just check that output is valid
                consistency = 1.0 if output.height > 0 and output.width > 0 else 0.0

            return RobustnessTestResult(
                counterfactual=cf,
                success=True,
                output=output,
                consistency_score=consistency,
            )

        except Exception as e:
            return RobustnessTestResult(
                counterfactual=cf,
                success=False,
                error=str(e),
                consistency_score=0.0,
            )

    def _compute_structural_similarity(
        self,
        output: Grid,
        reference: Grid,
    ) -> float:
        """Compute structural similarity between grids."""
        # Dimension similarity
        if output.shape != reference.shape:
            dim_sim = 0.5  # Partial credit for different dims
        else:
            dim_sim = 1.0

        # Palette similarity
        output_palette = output.palette
        ref_palette = reference.palette
        if output_palette == ref_palette:
            palette_sim = 1.0
        elif output_palette.issubset(ref_palette) or ref_palette.issubset(output_palette):
            palette_sim = 0.8
        else:
            common = len(output_palette & ref_palette)
            total = len(output_palette | ref_palette)
            palette_sim = common / total if total > 0 else 0.0

        return (dim_sim + palette_sim) / 2

    def _compute_shape_similarity(
        self,
        output: Grid,
        reference: Grid,
    ) -> float:
        """Compute shape similarity (ignoring colors)."""
        if output.shape != reference.shape:
            return 0.3

        # Compare non-zero patterns
        out_mask = output.data != 0
        ref_mask = reference.data != 0

        intersection = (out_mask & ref_mask).sum()
        union = (out_mask | ref_mask).sum()

        if union == 0:
            return 1.0  # Both empty

        return intersection / union

    def _analyze_failures(
        self,
        results: List[RobustnessTestResult],
    ) -> Dict[str, Any]:
        """Analyze failure patterns."""
        analysis: Dict[str, Any] = {}

        # Group by modification type
        by_type: Dict[str, List[RobustnessTestResult]] = {}
        for r in results:
            mod_type = r.counterfactual.modification_type
            if mod_type not in by_type:
                by_type[mod_type] = []
            by_type[mod_type].append(r)

        # Compute per-type statistics
        type_stats = {}
        for mod_type, type_results in by_type.items():
            num_type = len(type_results)
            if num_type > 0:
                avg_score = sum(r.consistency_score for r in type_results) / num_type
                error_rate = sum(1 for r in type_results if not r.success) / num_type
                type_stats[mod_type] = {
                    "count": num_type,
                    "avg_score": avg_score,
                    "error_rate": error_rate,
                }

        analysis["by_modification_type"] = type_stats

        # Identify most problematic modification types
        worst_types = sorted(
            type_stats.items(),
            key=lambda x: x[1]["avg_score"],
        )
        if worst_types:
            analysis["worst_modification_type"] = worst_types[0][0]

        return analysis


def compute_robustness_score(
    program: ASTNode,
    task: ARCTask,
) -> float:
    """Convenience function to compute robustness score."""
    checker = RobustnessChecker()
    result = checker.check_robustness(program, task)
    return result.overall_score


# ============================================================================
# Direct Counterfactual Robustness Check
# ============================================================================

@dataclass
class CounterfactualRobustnessResult:
    """Result of robustness check on provided counterfactuals."""
    overall_score: float
    num_tested: int
    num_succeeded: int
    num_failed: int
    failure_reasons: List[str]
    by_type: Dict[str, float]  # Score by modification type


def robustness_check(
    program: ASTNode,
    counterfactuals: List[Counterfactual],
    reference_outputs: Optional[List[Grid]] = None,
) -> CounterfactualRobustnessResult:
    """
    Check program robustness against a specific set of counterfactuals.

    This is a simpler interface than RobustnessChecker when you already
    have a set of counterfactuals to test against.

    Args:
        program: The AST program to test
        counterfactuals: List of counterfactual inputs to test
        reference_outputs: Optional reference outputs for comparison
            (if not provided, we just check the program doesn't crash)

    Returns:
        CounterfactualRobustnessResult with detailed statistics
    """
    if not counterfactuals:
        return CounterfactualRobustnessResult(
            overall_score=1.0,
            num_tested=0,
            num_succeeded=0,
            num_failed=0,
            failure_reasons=[],
            by_type={},
        )

    try:
        program_fn = make_program(program)
    except Exception as e:
        return CounterfactualRobustnessResult(
            overall_score=0.0,
            num_tested=len(counterfactuals),
            num_succeeded=0,
            num_failed=len(counterfactuals),
            failure_reasons=[f"Program compilation failed: {e}"],
            by_type={},
        )

    num_succeeded = 0
    num_failed = 0
    failure_reasons: List[str] = []
    scores_by_type: Dict[str, List[float]] = {}

    for i, cf in enumerate(counterfactuals):
        mod_type = cf.modification_type

        if mod_type not in scores_by_type:
            scores_by_type[mod_type] = []

        try:
            output = program_fn(cf.modified)

            # Check if output is valid
            if output is None or output.height == 0 or output.width == 0:
                num_failed += 1
                failure_reasons.append(f"CF {i}: empty output for {mod_type}")
                scores_by_type[mod_type].append(0.0)
                continue

            # Check against reference if available
            if reference_outputs and i < len(reference_outputs):
                ref = reference_outputs[i]
                if cf.expected_behavior == "same_transformation":
                    # Should produce similar output structure
                    if output.shape == ref.shape:
                        # Compute pixel similarity
                        matching = (output.data == ref.data).sum()
                        total = output.height * output.width
                        score = matching / total if total > 0 else 0.0
                    else:
                        score = 0.5  # Partial credit for different dims
                else:
                    # Just needs to be valid
                    score = 1.0

                scores_by_type[mod_type].append(score)
                if score >= 0.5:
                    num_succeeded += 1
                else:
                    num_failed += 1
                    failure_reasons.append(f"CF {i}: low score {score:.2f} for {mod_type}")
            else:
                # No reference - just check it ran
                num_succeeded += 1
                scores_by_type[mod_type].append(1.0)

        except Exception as e:
            num_failed += 1
            failure_reasons.append(f"CF {i}: {mod_type} raised {e}")
            scores_by_type[mod_type].append(0.0)

    # Aggregate scores
    total_tested = num_succeeded + num_failed
    overall_score = num_succeeded / total_tested if total_tested > 0 else 0.0

    by_type: Dict[str, float] = {}
    for mod_type, scores in scores_by_type.items():
        by_type[mod_type] = sum(scores) / len(scores) if scores else 0.0

    return CounterfactualRobustnessResult(
        overall_score=overall_score,
        num_tested=total_tested,
        num_succeeded=num_succeeded,
        num_failed=num_failed,
        failure_reasons=failure_reasons[:10],  # Limit to 10 reasons
        by_type=by_type,
    )


def quick_robustness_check(
    program: ASTNode,
    grid: Grid,
    num_counterfactuals: int = 5,
) -> float:
    """
    Quick robustness check with auto-generated counterfactuals.

    Args:
        program: The program to test
        grid: Input grid to generate counterfactuals from
        num_counterfactuals: Number of counterfactuals to test

    Returns:
        Robustness score (0.0 to 1.0)
    """
    from .counterfactuals import GridPerturbationGenerator

    generator = GridPerturbationGenerator()
    counterfactuals = generator.generate(grid, num_counterfactuals)

    result = robustness_check(program, counterfactuals)
    return result.overall_score
