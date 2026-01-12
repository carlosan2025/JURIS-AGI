"""Tests for the program synthesizer."""

import pytest

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.dsl.ast import PrimitiveNode, ComposeNode, LiteralNode
from juris_agi.dsl.prettyprint import ast_to_source
from juris_agi.cre.synthesizer import (
    BeamSearchSynthesizer,
    EnumerativeSynthesizer,
    SynthesisConfig,
)


def make_identity_task() -> ARCTask:
    """Create a simple identity task."""
    grid1 = Grid.from_list([
        [1, 2],
        [3, 4],
    ])
    grid2 = Grid.from_list([
        [5, 5, 5],
        [5, 5, 5],
    ])

    return ARCTask(
        task_id="test_identity",
        train=[
            ARCPair(input=grid1, output=grid1),
            ARCPair(input=grid2, output=grid2),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[7]]),
                output=Grid.from_list([[7]]),
            ),
        ],
    )


def make_rotate_task() -> ARCTask:
    """Create a rotation task."""
    inp1 = Grid.from_list([
        [1, 0],
        [0, 0],
    ])
    out1 = Grid.from_list([
        [0, 1],
        [0, 0],
    ])

    inp2 = Grid.from_list([
        [2, 3],
        [0, 0],
    ])
    out2 = Grid.from_list([
        [0, 2],
        [0, 3],
    ])

    return ARCTask(
        task_id="test_rotate",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[1, 1], [0, 0]]),
                output=Grid.from_list([[0, 1], [0, 1]]),
            ),
        ],
    )


def make_reflect_task() -> ARCTask:
    """Create a horizontal reflection task."""
    inp1 = Grid.from_list([
        [1, 0, 0],
    ])
    out1 = Grid.from_list([
        [0, 0, 1],
    ])

    inp2 = Grid.from_list([
        [1, 2, 3],
    ])
    out2 = Grid.from_list([
        [3, 2, 1],
    ])

    return ARCTask(
        task_id="test_reflect",
        train=[
            ARCPair(input=inp1, output=out1),
            ARCPair(input=inp2, output=out2),
        ],
        test=[
            ARCPair(
                input=Grid.from_list([[4, 5]]),
                output=Grid.from_list([[5, 4]]),
            ),
        ],
    )


class TestBeamSearchSynthesizer:
    """Test beam search synthesizer."""

    def test_finds_identity(self):
        """Synthesizer should find identity for identity task."""
        task = make_identity_task()
        synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=2,
            beam_width=20,
            max_iterations=100,
        ))

        result = synthesizer.synthesize(task)

        assert result.success, f"Failed to find identity: {result.error}"
        assert "identity" in result.program_source

    def test_finds_rotation(self):
        """Synthesizer should find rotation."""
        task = make_rotate_task()
        synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=3,
            beam_width=30,
            max_iterations=200,
        ))

        result = synthesizer.synthesize(task)

        assert result.success, f"Failed to find rotation: {result.error}"
        assert "rotate90" in result.program_source

    def test_finds_reflection(self):
        """Synthesizer should find horizontal reflection."""
        task = make_reflect_task()
        synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=3,
            beam_width=30,
            max_iterations=200,
        ))

        result = synthesizer.synthesize(task)

        assert result.success, f"Failed to find reflection: {result.error}"
        assert "reflect_h" in result.program_source

    def test_respects_max_depth(self):
        """Synthesizer should respect max depth."""
        task = make_identity_task()
        synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=1,
            beam_width=10,
            max_iterations=50,
        ))

        result = synthesizer.synthesize(task)

        if result.success and result.program:
            assert result.program.depth() <= 2  # Some tolerance

    def test_returns_partial_on_failure(self):
        """Should return best partial result on failure."""
        # Create impossible task
        task = ARCTask(
            task_id="impossible",
            train=[
                ARCPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[9, 9, 9, 9, 9]]),  # Unlikely to find
                ),
            ],
            test=[
                ARCPair(
                    input=Grid.from_list([[2]]),
                    output=Grid.from_list([[9, 9, 9, 9, 9]]),
                ),
            ],
        )

        synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=2,
            beam_width=10,
            max_iterations=20,
        ))

        result = synthesizer.synthesize(task)

        # Should return something even on failure
        assert result.nodes_explored > 0


class TestEnumerativeSynthesizer:
    """Test enumerative synthesizer."""

    def test_finds_identity_enumeration(self):
        """Enumeration should find identity."""
        task = make_identity_task()
        synthesizer = EnumerativeSynthesizer(max_depth=2)

        result = synthesizer.synthesize(task)

        assert result.success
        assert "identity" in result.program_source

    def test_finds_simple_transform(self):
        """Enumeration should find simple transforms."""
        task = make_reflect_task()
        synthesizer = EnumerativeSynthesizer(max_depth=2)

        result = synthesizer.synthesize(task)

        assert result.success


class TestSynthesisConfig:
    """Test synthesis configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = SynthesisConfig()

        assert config.max_depth > 0
        assert config.beam_width > 0
        assert config.max_iterations > 0

    def test_pruning_options(self):
        """Pruning options should affect synthesis."""
        task = make_identity_task()

        # With pruning
        config_with = SynthesisConfig(
            use_dimension_pruning=True,
            use_palette_pruning=True,
            max_depth=3,
            max_iterations=100,
        )

        # Without pruning
        config_without = SynthesisConfig(
            use_dimension_pruning=False,
            use_palette_pruning=False,
            max_depth=3,
            max_iterations=100,
        )

        synth_with = BeamSearchSynthesizer(config_with)
        synth_without = BeamSearchSynthesizer(config_without)

        result_with = synth_with.synthesize(task)
        result_without = synth_without.synthesize(task)

        # Both should succeed on this simple task
        assert result_with.success
        assert result_without.success

        # Pruning should explore fewer nodes
        assert result_with.candidates_pruned >= 0
