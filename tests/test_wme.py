"""
Tests for WME (World Model Expert) module.

Tests cover:
- propose_priors() heuristic
- generate_counterfactuals() with invariant preservation
- robustness_check() functionality
- Hard/soft veto selection logic
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.dsl.ast import PrimitiveNode, ComposeNode, LiteralNode
from juris_agi.wme import (
    # Priors
    ProposedPriors,
    ProposedInvariant,
    TransformFamily,
    propose_priors,
    extract_invariants_from_task,
    # World Model
    WorldModelState,
    # Counterfactuals
    Counterfactual,
    InvariantSpec,
    InvariantPreservingGenerator,
    generate_counterfactuals,
    # Robustness
    CounterfactualRobustnessResult,
    robustness_check,
    quick_robustness_check,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_grid() -> Grid:
    """A simple 3x3 grid for testing."""
    return Grid.from_list([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ])


@pytest.fixture
def larger_grid() -> Grid:
    """A larger 5x5 grid for testing."""
    return Grid.from_list([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ])


@pytest.fixture
def same_dims_task() -> ARCTask:
    """Task where input and output have same dimensions."""
    train = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([[4, 3], [2, 1]]),
        ),
        ARCPair(
            input=Grid.from_list([[5, 6], [7, 8]]),
            output=Grid.from_list([[8, 7], [6, 5]]),
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([[9, 0], [1, 2]]),
            output=Grid.from_list([[2, 1], [0, 9]]),
        ),
    ]
    return ARCTask(task_id="same_dims_test", train=train, test=test)


@pytest.fixture
def scaling_task() -> ARCTask:
    """Task with 2x scaling."""
    train = [
        ARCPair(
            input=Grid.from_list([[1, 2], [3, 4]]),
            output=Grid.from_list([
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]),
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([[5]]),
            output=Grid.from_list([[5, 5], [5, 5]]),
        ),
    ]
    return ARCTask(task_id="scaling_test", train=train, test=test)


@pytest.fixture
def cropping_task() -> ARCTask:
    """Task with cropping operation."""
    train = [
        ARCPair(
            input=Grid.from_list([
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0],
            ]),
            output=Grid.from_list([[1, 2], [3, 4]]),
        ),
    ]
    test = [
        ARCPair(
            input=Grid.from_list([
                [0, 0, 0],
                [0, 5, 0],
                [0, 0, 0],
            ]),
            output=Grid.from_list([[5]]),
        ),
    ]
    return ARCTask(task_id="cropping_test", train=train, test=test)


# ============================================================================
# Tests for propose_priors()
# ============================================================================

class TestProposePriors:
    """Tests for the propose_priors() function."""

    def test_same_dims_features(self):
        """Test priors proposed for same-dimension features."""
        state = WorldModelState()
        state.task_features = {
            "same_dims": True,
            "same_palette": True,
            "input_height": 3,
            "input_width": 3,
            "h_ratio": 1.0,
            "w_ratio": 1.0,
        }

        priors = propose_priors(state)

        assert isinstance(priors, ProposedPriors)
        assert priors.confidence > 0.5

        # Should suggest geometric transforms
        suggested = priors.get_suggested_primitives()
        assert any(p in suggested for p in ["rotate90", "reflect_h", "reflect_v"])

        # Should have dims_preserved invariant
        hard_invariants = priors.get_hard_invariants()
        assert any(inv.name == "dims_preserved" for inv in hard_invariants)

    def test_scaling_features(self):
        """Test priors proposed for scaling features."""
        state = WorldModelState()
        state.task_features = {
            "same_dims": False,
            "same_palette": True,
            "h_ratio": 2.0,
            "w_ratio": 2.0,
        }

        priors = propose_priors(state)

        # Should suggest scaling transforms
        families = [f.name for f in priors.transform_families]
        assert "uniform_scaling" in families or "tiling" in families

        suggested = priors.get_suggested_primitives()
        assert any(p in suggested for p in ["scale", "tile_repeat", "tile_h", "tile_v"])

    def test_cropping_features(self):
        """Test priors proposed for cropping features."""
        state = WorldModelState()
        state.task_features = {
            "same_dims": False,
            "same_palette": True,
            "h_ratio": 0.5,
            "w_ratio": 0.5,
        }

        priors = propose_priors(state)

        # Should suggest cropping transforms
        families = [f.name for f in priors.transform_families]
        assert "cropping" in families

        suggested = priors.get_suggested_primitives()
        assert any(p in suggested for p in ["crop_to_content", "crop_to_bbox"])

    def test_recoloring_features(self):
        """Test priors proposed when palette changes."""
        state = WorldModelState()
        state.task_features = {
            "same_dims": True,
            "same_palette": False,
            "h_ratio": 1.0,
            "w_ratio": 1.0,
        }

        priors = propose_priors(state)

        # Should suggest recoloring transforms
        families = [f.name for f in priors.transform_families]
        assert "recoloring" in families

        suggested = priors.get_suggested_primitives()
        assert any(p in suggested for p in ["recolor", "recolor_map", "fill_background"])

    def test_empty_features(self):
        """Test priors with empty features."""
        state = WorldModelState()
        state.task_features = {}

        priors = propose_priors(state)

        # Should return some priors (recoloring triggered by missing same_palette)
        assert priors.confidence > 0
        assert len(priors.transform_families) > 0


class TestExtractInvariantsFromTask:
    """Tests for extract_invariants_from_task()."""

    def test_same_dims_task(self, same_dims_task):
        """Test invariant extraction from same-dims task."""
        invariants = extract_invariants_from_task(same_dims_task)

        assert len(invariants) > 0
        names = [inv.name for inv in invariants]
        assert "dims_preserved" in names

    def test_scaling_task(self, scaling_task):
        """Test invariant extraction from scaling task."""
        invariants = extract_invariants_from_task(scaling_task)

        # Should detect consistent output dimensions
        names = [inv.name for inv in invariants]
        # Scaling changes dims, so dims_preserved should NOT be present
        assert "dims_preserved" not in names

    def test_empty_task(self):
        """Test invariant extraction from empty task."""
        task = ARCTask(task_id="empty", train=[], test=[])
        invariants = extract_invariants_from_task(task)
        assert invariants == []


# ============================================================================
# Tests for generate_counterfactuals()
# ============================================================================

class TestGenerateCounterfactuals:
    """Tests for the generate_counterfactuals() function."""

    def test_basic_generation(self, simple_grid):
        """Test basic counterfactual generation."""
        state = WorldModelState()
        state.task_features = {"same_dims": True}

        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=[],
            k=5,
            grid=simple_grid,
        )

        assert len(counterfactuals) > 0
        assert len(counterfactuals) <= 5

        for cf in counterfactuals:
            assert isinstance(cf, Counterfactual)
            assert cf.original == simple_grid
            assert cf.modified is not None
            assert cf.modification_type is not None

    def test_with_dims_invariant(self, simple_grid):
        """Test counterfactual generation with dims invariant."""
        state = WorldModelState()

        invariants = [
            ProposedInvariant(
                name="dims_preserved",
                confidence=0.9,
                value=(3, 3),
                is_hard=True,
            ),
        ]

        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=invariants,
            k=5,
            grid=simple_grid,
        )

        # All counterfactuals should preserve dimensions
        for cf in counterfactuals:
            if cf.modification_type not in ("pad", "crop"):
                # Invariant-preserving modifications should keep dims
                assert cf.modified.shape == simple_grid.shape, \
                    f"Dims not preserved for {cf.modification_type}"

    def test_with_palette_invariant(self, simple_grid):
        """Test counterfactual generation with palette invariant."""
        state = WorldModelState()

        invariants = [
            ProposedInvariant(
                name="palette_preserved",
                confidence=0.85,
                value=None,
                is_hard=True,
            ),
        ]

        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=invariants,
            k=10,
            grid=simple_grid,
        )

        original_palette = simple_grid.palette

        # Most counterfactuals should preserve palette (except add_color type)
        preserved_count = sum(
            1 for cf in counterfactuals
            if cf.modified.palette.issubset(original_palette)
            or cf.modification_type == "add_color"
        )
        assert preserved_count > 0

    def test_no_grid_returns_empty(self):
        """Test that no grid returns empty list."""
        state = WorldModelState()
        counterfactuals = generate_counterfactuals(state, [], k=5, grid=None)
        assert counterfactuals == []

    def test_valid_grids_produced(self, larger_grid):
        """Test that all generated counterfactuals are valid grids."""
        state = WorldModelState()

        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=[],
            k=10,
            grid=larger_grid,
        )

        for cf in counterfactuals:
            # Grid should be valid
            assert cf.modified.height > 0
            assert cf.modified.width > 0
            # All values should be valid colors (0-9)
            assert cf.modified.data.min() >= 0
            assert cf.modified.data.max() <= 9


class TestInvariantPreservingGenerator:
    """Tests for InvariantPreservingGenerator class."""

    def test_pixel_flip_preserves_dims(self, simple_grid):
        """Test pixel_flip preserves dimensions."""
        generator = InvariantPreservingGenerator(seed=42)
        cf = generator._pixel_flip_invariant(simple_grid)

        assert cf.modified.shape == simple_grid.shape

    def test_pixel_move_preserves_dims(self, simple_grid):
        """Test pixel_move preserves dimensions."""
        generator = InvariantPreservingGenerator(seed=42)
        cf = generator._pixel_move_invariant(simple_grid)

        if cf is not None:
            assert cf.modified.shape == simple_grid.shape

    def test_region_swap_preserves_both(self, larger_grid):
        """Test region_swap preserves both dims and palette."""
        generator = InvariantPreservingGenerator(seed=42)
        cf = generator._region_swap_invariant(larger_grid)

        assert cf.modified.shape == larger_grid.shape
        assert cf.modified.palette == larger_grid.palette

    def test_color_swap_preserves_palette(self, simple_grid):
        """Test color_swap preserves palette set."""
        generator = InvariantPreservingGenerator(seed=42)
        cf = generator._color_swap_invariant(simple_grid)

        if cf is not None:
            assert cf.modified.palette == simple_grid.palette

    def test_shift_wrap_preserves_dims(self, simple_grid):
        """Test shift_wrap preserves dimensions."""
        generator = InvariantPreservingGenerator(seed=42)
        cf = generator._shift_invariant(simple_grid)

        assert cf.modified.shape == simple_grid.shape


# ============================================================================
# Tests for robustness_check()
# ============================================================================

class TestRobustnessCheck:
    """Tests for robustness_check() function."""

    def test_empty_counterfactuals(self):
        """Test with empty counterfactual list."""
        program = PrimitiveNode("identity")
        result = robustness_check(program, [])

        assert result.overall_score == 1.0
        assert result.num_tested == 0

    def test_identity_program_robustness(self, simple_grid):
        """Test identity program robustness."""
        program = PrimitiveNode("identity")

        # Generate some counterfactuals
        generator = InvariantPreservingGenerator(seed=42)
        counterfactuals = generator.generate(simple_grid, num_counterfactuals=5)

        result = robustness_check(program, counterfactuals)

        assert isinstance(result, CounterfactualRobustnessResult)
        assert result.num_tested == len(counterfactuals)
        # Identity should succeed on all (just returns input)
        assert result.num_succeeded > 0

    def test_with_failing_program(self, simple_grid):
        """Test robustness with a program that will fail."""
        # Create a program that requires specific dimensions
        program = PrimitiveNode("crop_to_content")

        # Generate counterfactuals that might break it
        generator = InvariantPreservingGenerator(seed=42)
        counterfactuals = generator.generate(simple_grid, num_counterfactuals=5)

        result = robustness_check(program, counterfactuals)

        # Should have some results
        assert result.num_tested > 0

    def test_by_type_statistics(self, larger_grid):
        """Test that by_type statistics are computed."""
        program = PrimitiveNode("identity")

        # Generate counterfactuals of different types
        generator = InvariantPreservingGenerator(seed=42)
        counterfactuals = generator.generate(larger_grid, num_counterfactuals=10)

        result = robustness_check(program, counterfactuals)

        # Should have by_type breakdown
        assert isinstance(result.by_type, dict)
        assert len(result.by_type) > 0


class TestQuickRobustnessCheck:
    """Tests for quick_robustness_check() function."""

    def test_basic_call(self, simple_grid):
        """Test basic quick robustness check."""
        program = PrimitiveNode("identity")

        score = quick_robustness_check(program, simple_grid, num_counterfactuals=3)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identity_high_robustness(self, simple_grid):
        """Test that identity has high robustness."""
        program = PrimitiveNode("identity")

        score = quick_robustness_check(program, simple_grid, num_counterfactuals=5)

        # Identity should be robust (just returns what it gets)
        assert score > 0.5


# ============================================================================
# Tests for Hard/Soft Veto Selection
# ============================================================================

class TestSelectionLogic:
    """Tests for hard/soft veto selection in synthesizer."""

    def test_hard_veto_on_dimension_mismatch(self, same_dims_task):
        """Test that dimension mismatch triggers hard veto."""
        from juris_agi.cre.synthesizer import BeamSearchSynthesizer, SynthesisConfig

        synth = BeamSearchSynthesizer()

        # Create a program that changes dimensions (scale)
        program = PrimitiveNode("scale", [LiteralNode(2)])

        # Evaluate on training pairs
        score, results = synth._evaluate_candidate(program, same_dims_task.train)

        # Apply hard veto
        should_veto, reason = synth._apply_hard_veto(
            program, same_dims_task, results
        )

        # Should be vetoed because output dims don't match expected
        assert should_veto
        assert "Dimension mismatch" in reason

    def test_no_veto_on_correct_dims(self, same_dims_task):
        """Test that correct dimensions don't trigger veto."""
        from juris_agi.cre.synthesizer import BeamSearchSynthesizer

        synth = BeamSearchSynthesizer()

        # Create a program that preserves dimensions
        program = PrimitiveNode("identity")

        # Evaluate on training pairs
        score, results = synth._evaluate_candidate(program, same_dims_task.train)

        # Apply hard veto
        should_veto, reason = synth._apply_hard_veto(
            program, same_dims_task, results
        )

        # Should NOT be vetoed
        assert not should_veto

    def test_soft_scoring_with_wme(self, same_dims_task):
        """Test soft scoring includes length penalty."""
        from juris_agi.cre.synthesizer import BeamSearchSynthesizer, SynthesisConfig

        cfg = SynthesisConfig(
            use_wme=True,
            wme_length_weight=0.1,
            wme_robustness_weight=0.0,  # Disable robustness for this test
        )
        synth = BeamSearchSynthesizer(config=cfg)

        # Short program
        short_program = PrimitiveNode("identity")
        short_score, short_breakdown = synth._apply_selection_scoring(
            short_program, 50.0, same_dims_task, cfg
        )

        # Longer program
        long_program = ComposeNode([
            PrimitiveNode("identity"),
            PrimitiveNode("identity"),
            PrimitiveNode("identity"),
        ])
        long_score, long_breakdown = synth._apply_selection_scoring(
            long_program, 50.0, same_dims_task, cfg
        )

        # Shorter program should have higher score
        assert short_score > long_score
        assert "length_penalty" in short_breakdown
        assert short_breakdown["length_penalty"] > long_breakdown["length_penalty"]

    def test_soft_scoring_without_wme(self, same_dims_task):
        """Test soft scoring is disabled when use_wme=False."""
        from juris_agi.cre.synthesizer import BeamSearchSynthesizer, SynthesisConfig

        cfg = SynthesisConfig(use_wme=False)
        synth = BeamSearchSynthesizer(config=cfg)

        program = PrimitiveNode("identity")
        score, breakdown = synth._apply_selection_scoring(
            program, 50.0, same_dims_task, cfg
        )

        # Score should be unchanged
        assert score == 50.0
        assert breakdown == {"base": 50.0}


# ============================================================================
# Integration Tests
# ============================================================================

class TestWMEIntegration:
    """Integration tests for WME components working together."""

    def test_priors_to_counterfactuals_pipeline(self, same_dims_task):
        """Test full pipeline from priors to counterfactuals."""
        from juris_agi.wme.world_model import HeuristicWorldModel

        # Step 1: Analyze task with world model
        wm = HeuristicWorldModel()
        state = wm.analyze_task(same_dims_task)

        # Step 2: Propose priors
        priors = propose_priors(state, same_dims_task)

        # Step 3: Generate counterfactuals respecting invariants
        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=priors.invariants,
            k=5,
            grid=same_dims_task.train[0].input,
        )

        assert len(counterfactuals) > 0

        # Step 4: Check robustness of a simple program
        program = PrimitiveNode("identity")
        robustness = robustness_check(program, counterfactuals)

        assert robustness.num_tested > 0
        assert robustness.overall_score >= 0.0

    def test_invariant_extraction_to_generation(self, scaling_task):
        """Test invariant extraction feeding into counterfactual generation."""
        # Extract invariants from task
        invariants = extract_invariants_from_task(scaling_task)

        # Generate counterfactuals respecting those invariants
        state = WorldModelState()
        counterfactuals = generate_counterfactuals(
            state=state,
            invariants=invariants,
            k=5,
            grid=scaling_task.train[0].input,
        )

        # All should be valid
        for cf in counterfactuals:
            assert cf.modified.height > 0
            assert cf.modified.width > 0
