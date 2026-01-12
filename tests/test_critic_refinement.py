"""Tests for symbolic critic and refinement engine."""

import pytest

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.dsl.ast import PrimitiveNode, ComposeNode, LiteralNode
from juris_agi.dsl.interpreter import make_program
from juris_agi.cre.critic_symbolic import (
    SymbolicCritic,
    SymbolicDiff,
    CriticResult,
    compute_symbolic_diff,
    extract_invariants,
)
from juris_agi.cre.refinement import (
    RefinementEngine,
    RefinementResult,
    EditType,
    EditOperation,
)


def create_identity_task() -> ARCTask:
    """Simple identity task."""
    return ARCTask(
        task_id="test_identity",
        train=[
            ARCPair(
                input=Grid.from_list([[1, 2], [3, 4]]),
                output=Grid.from_list([[1, 2], [3, 4]]),
            ),
            ARCPair(
                input=Grid.from_list([[5, 5, 5]]),
                output=Grid.from_list([[5, 5, 5]]),
            ),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[7, 8]]),
                output=Grid.from_list([[7, 8]]),
            ),
        ],
    )


def create_rotate_task() -> ARCTask:
    """90 degree rotation task."""
    inp1 = Grid.from_list([
        [1, 0],
        [0, 0],
    ])
    out1 = Grid.from_list([
        [0, 1],
        [0, 0],
    ])

    inp2 = Grid.from_list([
        [2, 2],
        [0, 0],
    ])
    out2 = Grid.from_list([
        [0, 2],
        [0, 2],
    ])

    return ARCTask(
        task_id="test_rotate",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[3, 0], [0, 0]]),
                output=Grid.from_list([[0, 3], [0, 0]]),
            ),
        ],
    )


class TestSymbolicDiff:
    """Tests for symbolic diff computation."""

    def test_exact_match(self):
        """Identical grids should have zero diff."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        diff = compute_symbolic_diff(grid, grid)

        assert diff.exact_match
        assert diff.dimension_match
        assert diff.pixel_accuracy == 1.0
        assert diff.num_errors == 0
        assert diff.severity == 0.0

    def test_dimension_mismatch(self):
        """Different sized grids should fail dimension match."""
        grid1 = Grid.from_list([[1, 2]])
        grid2 = Grid.from_list([[1, 2], [3, 4]])

        diff = compute_symbolic_diff(grid1, grid2)

        assert not diff.dimension_match
        assert not diff.exact_match
        assert diff.severity == 1.0

    def test_partial_match(self):
        """Partial match should have intermediate severity."""
        grid1 = Grid.from_list([[1, 2], [3, 4]])
        grid2 = Grid.from_list([[1, 2], [3, 5]])  # One pixel different

        diff = compute_symbolic_diff(grid1, grid2)

        assert diff.dimension_match
        assert not diff.exact_match
        assert diff.pixel_accuracy == 0.75
        assert diff.num_errors == 1
        assert 0 < diff.severity < 1.0

    def test_color_analysis(self):
        """Should detect extra and missing colors."""
        grid1 = Grid.from_list([[1, 2]])  # Has colors 1, 2
        grid2 = Grid.from_list([[1, 3]])  # Has colors 1, 3

        diff = compute_symbolic_diff(grid1, grid2)

        assert 2 in diff.extra_colors
        assert 3 in diff.missing_colors

    def test_diff_entry_types(self):
        """Should correctly classify diff types."""
        # Wrong color
        grid1 = Grid.from_list([[1]])
        grid2 = Grid.from_list([[2]])
        diff = compute_symbolic_diff(grid1, grid2)
        assert diff.diff_entries[0].diff_type == "wrong_color"

        # Extra pixel (predicted has color where expected is 0)
        grid1 = Grid.from_list([[1]])
        grid2 = Grid.from_list([[0]])
        diff = compute_symbolic_diff(grid1, grid2)
        assert diff.diff_entries[0].diff_type == "extra_pixel"

        # Missing pixel (predicted is 0 where expected has color)
        grid1 = Grid.from_list([[0]])
        grid2 = Grid.from_list([[1]])
        diff = compute_symbolic_diff(grid1, grid2)
        assert diff.diff_entries[0].diff_type == "missing_pixel"


class TestSymbolicCritic:
    """Tests for the symbolic critic."""

    def test_approve_correct_program(self):
        """Critic should approve correct program."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert result.approved
        assert result.exact_match_all
        assert result.is_certified
        assert len(result.diffs) == len(task.train)

    def test_reject_wrong_program(self):
        """Critic should reject incorrect program."""
        task = create_identity_task()
        # Reflection is wrong for identity task
        program = PrimitiveNode("reflect_h")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        # Check if it's actually wrong (asymmetric grids will fail)
        # For [[1, 2], [3, 4]], reflection gives [[2, 1], [4, 3]]
        assert not result.exact_match_all or not result.is_certified

    def test_approve_rotation(self):
        """Critic should approve correct rotation."""
        task = create_rotate_task()
        program = PrimitiveNode("rotate90", [LiteralNode(1)])
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert result.is_certified

    def test_diffs_populated(self):
        """Diffs should be populated for each pair."""
        task = create_identity_task()
        program = PrimitiveNode("reflect_h")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert len(result.diffs) == len(task.train)
        for diff in result.diffs:
            assert isinstance(diff, SymbolicDiff)

    def test_invariant_checking_determinism(self):
        """Should check determinism invariant."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert "determinism" in result.invariants_satisfied

    def test_invariant_checking_palette(self):
        """Should check palette consistency."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert "palette_consistency" in result.invariants_satisfied

    def test_refinement_hints(self):
        """Should compute refinement hints from diffs."""
        task = create_identity_task()
        program = PrimitiveNode("reflect_h")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)
        hints = critic.compute_refinement_hints(result.diffs)

        # Should have hints for non-exact matches
        assert isinstance(hints, list)


class TestExtractInvariants:
    """Tests for invariant extraction."""

    def test_extract_fixed_output_dims(self):
        """Should detect fixed output dimensions."""
        task = ARCTask(
            task_id="fixed_dims",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2, 3]]),
                    output=Grid.from_list([[1]]),
                ),
                ARCPair(
                    input=Grid.from_list([[4, 5, 6, 7]]),
                    output=Grid.from_list([[2]]),
                ),
            ],
            test=[],
        )

        invariants = extract_invariants(task)

        assert "fixed_output_dims" in invariants
        assert invariants["fixed_output_dims"] == (1, 1)

    def test_extract_dimension_ratio(self):
        """Should detect consistent dimension ratios."""
        task = ARCTask(
            task_id="ratio",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2], [3, 4]]),
                    output=Grid.from_list([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
                ),
            ],
            test=[],
        )

        invariants = extract_invariants(task)

        assert "dimension_ratio" in invariants
        assert invariants["dimension_ratio"] == (2.0, 2.0)

    def test_extract_palette_info(self):
        """Should extract palette information."""
        task = create_identity_task()
        invariants = extract_invariants(task)

        assert "input_palette" in invariants
        assert "output_palette" in invariants
        assert "palette_preserved" in invariants


class TestRefinementEngine:
    """Tests for the refinement engine."""

    def test_refine_already_correct(self):
        """Should recognize already correct program."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        engine = RefinementEngine()

        result = engine.refine(program, task)

        assert result.success
        assert not result.improved
        assert result.original_score == 100.0

    def test_refine_improves_score(self):
        """Refinement should improve or maintain score."""
        # Create a task where rotation is correct
        task = create_rotate_task()
        # Start with wrong program
        program = PrimitiveNode("identity")
        engine = RefinementEngine(max_iterations=30)

        result = engine.refine(program, task)

        # Should either improve or stay same
        assert result.new_score >= result.original_score

    def test_swap_primitive_edits(self):
        """Should generate swap primitive edits."""
        program = PrimitiveNode("rotate90", [LiteralNode(1)])
        engine = RefinementEngine()

        edits = engine._generate_swap_edits(program, [])

        # Should find swap candidates for rotation
        edit_types = [e[0].edit_type for e in edits]
        assert EditType.SWAP_PRIMITIVE in edit_types

    def test_arg_tweak_edits(self):
        """Should generate argument tweak edits."""
        program = PrimitiveNode("rotate90", [LiteralNode(1)])
        engine = RefinementEngine()

        edits = engine._generate_arg_tweaks(program, [])

        # Should suggest other rotation values
        assert len(edits) > 0
        for edit, _ in edits:
            assert edit.edit_type == EditType.TWEAK_ARG

    def test_insert_edits(self):
        """Should generate insert edits."""
        program = PrimitiveNode("identity")
        engine = RefinementEngine()

        edits = engine._generate_insert_edits(program, [])

        assert len(edits) > 0
        for edit, _ in edits:
            assert edit.edit_type == EditType.INSERT_PRIMITIVE

    def test_removal_edits(self):
        """Should generate removal edits for compositions."""
        program = ComposeNode([
            PrimitiveNode("rotate90", [LiteralNode(1)]),
            PrimitiveNode("reflect_h"),
        ])
        engine = RefinementEngine()

        edits = engine._generate_removal_edits(program, [])

        assert len(edits) > 0
        for edit, _ in edits:
            assert edit.edit_type == EditType.REMOVE_PRIMITIVE

    def test_order_swap_edits(self):
        """Should generate order swap edits for compositions."""
        program = ComposeNode([
            PrimitiveNode("rotate90", [LiteralNode(1)]),
            PrimitiveNode("reflect_h"),
        ])
        engine = RefinementEngine()

        edits = engine._generate_order_swap_edits(program, [])

        assert len(edits) > 0
        for edit, _ in edits:
            assert edit.edit_type == EditType.SWAP_ORDER

    def test_refinement_result_properties(self):
        """Refinement result should have expected properties."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        engine = RefinementEngine()

        result = engine.refine(program, task)

        assert isinstance(result, RefinementResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.improved, bool)
        assert isinstance(result.original_score, float)
        assert isinstance(result.new_score, float)
        assert result.original_program is not None


class TestEditOperation:
    """Tests for edit operations."""

    def test_describe_swap(self):
        """Should describe swap edits."""
        edit = EditOperation(
            edit_type=EditType.SWAP_PRIMITIVE,
            location=0,
            original="rotate90",
            replacement="reflect_h",
        )
        desc = edit.describe()
        assert "Swap" in desc
        assert "rotate90" in desc
        assert "reflect_h" in desc

    def test_describe_tweak(self):
        """Should describe tweak edits."""
        edit = EditOperation(
            edit_type=EditType.TWEAK_ARG,
            location=0,
            details={"new_value": 2},
        )
        desc = edit.describe()
        assert "Adjust" in desc or "argument" in desc.lower()

    def test_describe_swap_order(self):
        """Should describe order swap edits."""
        edit = EditOperation(
            edit_type=EditType.SWAP_ORDER,
            location=0,
        )
        desc = edit.describe()
        assert "Swap" in desc and "order" in desc.lower()


class TestCriticRefinementIntegration:
    """Integration tests for critic and refinement."""

    def test_critic_then_refine(self):
        """Critic result should be usable for refinement."""
        task = create_rotate_task()
        program = PrimitiveNode("identity")  # Wrong for rotation
        critic = SymbolicCritic()
        engine = RefinementEngine()

        critique = critic.evaluate(program, task)
        assert not critique.is_certified

        # Refinement should accept critique
        result = engine.refine(program, task, critique)
        assert isinstance(result, RefinementResult)

    def test_refinement_hints_used(self):
        """Refinement should use critic hints."""
        task = create_identity_task()
        # Use a program that produces wrong colors
        program = PrimitiveNode("reflect_h")
        critic = SymbolicCritic()
        engine = RefinementEngine()

        critique = critic.evaluate(program, task)
        hints = critic.compute_refinement_hints(critique.diffs)

        # Hints should be generated
        assert isinstance(hints, list)

        # These hints are used internally by refinement
        result = engine.refine(program, task, critique)
        assert isinstance(result, RefinementResult)
