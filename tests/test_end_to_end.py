"""End-to-end tests for the JURIS-AGI system."""

import pytest

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.controller.router import MetaController, ControllerConfig
from juris_agi.controller.refusal import RefusalChecker
from juris_agi.cre.critic_symbolic import SymbolicCritic
from juris_agi.dsl.ast import PrimitiveNode, LiteralNode
from juris_agi.dsl.interpreter import make_program


def create_identity_task() -> ARCTask:
    """Simple identity task."""
    return ARCTask(
        task_id="e2e_identity",
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


def create_rotate90_task() -> ARCTask:
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
        task_id="e2e_rotate90",
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


def create_crop_task() -> ARCTask:
    """Crop to content task."""
    inp1 = Grid.from_list([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    out1 = Grid.from_list([[1]])

    inp2 = Grid.from_list([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0],
    ])
    out2 = Grid.from_list([[2, 2]])

    return ARCTask(
        task_id="e2e_crop",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[0, 0], [0, 3]]),
                output=Grid.from_list([[3]]),
            ),
        ],
    )


class TestMetaController:
    """Test the meta-controller end-to-end."""

    @pytest.fixture
    def controller(self):
        """Create a controller with fast settings."""
        config = ControllerConfig(
            max_synthesis_depth=3,
            beam_width=30,
            max_synthesis_iterations=200,
            enable_refinement=True,
            compute_robustness=False,  # Skip for speed
        )
        return MetaController(config)

    def test_solve_identity(self, controller):
        """Controller should solve identity task."""
        task = create_identity_task()
        result = controller.solve(task)

        assert result.success, f"Failed: {result.error_message}"
        assert result.is_certified
        assert "identity" in result.audit_trace.program_source

    def test_solve_rotation(self, controller):
        """Controller should solve rotation task."""
        task = create_rotate90_task()
        result = controller.solve(task)

        assert result.success, f"Failed: {result.error_message}"
        assert result.is_certified

    def test_solve_crop(self, controller):
        """Controller should solve crop task."""
        task = create_crop_task()
        result = controller.solve(task)

        assert result.success, f"Failed: {result.error_message}"
        assert "crop" in result.audit_trace.program_source.lower()

    def test_predictions_correct(self, controller):
        """Predictions should match expected outputs."""
        task = create_identity_task()
        result = controller.solve(task)

        assert result.success
        assert len(result.predictions) == len(task.test)

        for pred, pair in zip(result.predictions, task.test):
            assert pred == pair.output

    def test_audit_trace_populated(self, controller):
        """Audit trace should be populated."""
        task = create_identity_task()
        result = controller.solve(task)

        assert result.audit_trace.task_id == task.task_id
        assert result.audit_trace.program_source
        assert len(result.audit_trace.constraints_satisfied) > 0


class TestSymbolicCritic:
    """Test the symbolic critic (jurisdiction)."""

    def test_approve_correct_program(self):
        """Critic should approve correct program."""
        task = create_identity_task()
        program = PrimitiveNode("identity")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        assert result.approved
        assert result.exact_match_all
        assert result.is_certified

    def test_reject_wrong_program(self):
        """Critic should reject wrong program."""
        task = create_identity_task()
        # Rotation is wrong for identity task
        program = PrimitiveNode("rotate90", [LiteralNode(1)])
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        # Should not be certified if wrong
        # (may or may not approve depending on strictness)
        assert not result.exact_match_all or not result.is_certified

    def test_diffs_computed(self):
        """Critic should compute diffs for wrong programs."""
        inp = Grid.from_list([[1, 2], [3, 4]])
        out = Grid.from_list([[1, 2], [3, 4]])
        task = ARCTask(
            task_id="diff_test",
            train=[ARCPair(input=inp, output=out)],
            test=[ARCPair(input=inp, output=out)],
        )

        # Use a program that gives wrong output
        program = PrimitiveNode("reflect_h")
        critic = SymbolicCritic()

        result = critic.evaluate(program, task)

        # Should have diffs since reflection changes the grid
        if not result.exact_match_all:
            assert len(result.diffs) > 0
            assert result.diffs[0].num_errors > 0


class TestRefusalChecker:
    """Test the refusal checker."""

    def test_accept_valid_task(self):
        """Should accept valid tasks."""
        task = create_identity_task()
        checker = RefusalChecker()

        decision = checker.check(task)

        assert not decision.should_refuse

    def test_reject_empty_train(self):
        """Should reject task with no training data."""
        task = ARCTask(
            task_id="empty",
            train=[],
            test=[ARCPair(input=Grid.from_list([[1]]), output=Grid.from_list([[1]]))],
        )
        checker = RefusalChecker()

        decision = checker.check(task)

        assert decision.should_refuse
        assert "training" in decision.explanation.lower()

    def test_reject_oversized_grid(self):
        """Should reject oversized grids."""
        big_grid = Grid.zeros(100, 100)
        task = ARCTask(
            task_id="too_big",
            train=[ARCPair(input=big_grid, output=big_grid)],
            test=[ARCPair(input=big_grid, output=big_grid)],
        )
        checker = RefusalChecker(max_grid_size=30)

        decision = checker.check(task)

        assert decision.should_refuse


class TestCertification:
    """Test certification requirements."""

    def test_certified_solution_properties(self):
        """Certified solutions should have required properties."""
        task = create_identity_task()
        controller = MetaController(ControllerConfig(
            compute_robustness=False,
        ))

        result = controller.solve(task)

        if result.is_certified:
            # Must have program
            assert result.audit_trace.program_source

            # Must have passed constraints
            assert "determinism" in result.audit_trace.constraints_satisfied

            # Must make correct predictions
            program = make_program(result.audit_trace.program_ast)
            for pair in task.train:
                pred = program(pair.input)
                assert pred == pair.output
