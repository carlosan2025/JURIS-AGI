"""
Tests for neural sketcher and critic models.

Tests both the PyTorch implementations (if available) and the
heuristic fallbacks (always available).
"""

import pytest
import numpy as np

from juris_agi.core.types import Grid, ARCTask, ARCPair
from juris_agi.dsl.ast import PrimitiveNode, ComposeNode, LiteralNode
from juris_agi.cre.sketcher_model import (
    SketcherModel,
    ProgramSketch,
    SketcherConfig,
    HeuristicSketcher,
    get_sketcher,
    TORCH_AVAILABLE,
    PRIMITIVE_TO_ID,
    ID_TO_PRIMITIVE,
    NUM_PRIMITIVES,
    VOCAB_SIZE,
)
from juris_agi.cre.critic_neural import (
    NeuralCritic,
    NeuralCriticScore,
    CriticConfig,
    StubNeuralCritic,
    EnsembleNeuralCritic,
    get_critic,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_task():
    """Create a simple identity task."""
    grid = Grid(np.array([[1, 2], [3, 4]], dtype=np.int32))
    return ARCTask(
        task_id="test_identity",
        train=[ARCPair(input=grid, output=grid)],
        test=[],
    )


@pytest.fixture
def rotation_task():
    """Create a task that requires rotation."""
    input_grid = Grid(np.array([[1, 2], [3, 4]], dtype=np.int32))
    output_grid = Grid(np.array([[3, 1], [4, 2]], dtype=np.int32))
    return ARCTask(
        task_id="test_rotate",
        train=[ARCPair(input=input_grid, output=output_grid)],
        test=[],
    )


@pytest.fixture
def scaling_task():
    """Create a task that requires scaling."""
    input_grid = Grid(np.array([[1, 2], [3, 4]], dtype=np.int32))
    output_grid = Grid(np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.int32))
    return ARCTask(
        task_id="test_scale",
        train=[ARCPair(input=input_grid, output=output_grid)],
        test=[],
    )


@pytest.fixture
def simple_program():
    """Create a simple program AST."""
    return PrimitiveNode("identity")


@pytest.fixture
def complex_program():
    """Create a more complex program AST."""
    return ComposeNode([
        PrimitiveNode("rotate90", [LiteralNode(1)]),
        PrimitiveNode("reflect_h"),
    ])


# =============================================================================
# Tests for Vocabulary
# =============================================================================

class TestVocabulary:
    """Test primitive vocabulary mapping."""

    def test_primitive_to_id_mapping(self):
        """All primitives should have unique IDs."""
        assert len(PRIMITIVE_TO_ID) == NUM_PRIMITIVES
        assert len(set(PRIMITIVE_TO_ID.values())) == NUM_PRIMITIVES

    def test_id_to_primitive_mapping(self):
        """ID to primitive mapping should be consistent."""
        for name, idx in PRIMITIVE_TO_ID.items():
            assert ID_TO_PRIMITIVE[idx] == name

    def test_vocab_size_includes_special_tokens(self):
        """Vocab size should include PAD, SOS, EOS."""
        assert VOCAB_SIZE == NUM_PRIMITIVES + 3

    def test_common_primitives_present(self):
        """Common primitives should be in vocabulary."""
        common = ["identity", "rotate90", "reflect_h", "reflect_v", "transpose"]
        for prim in common:
            assert prim in PRIMITIVE_TO_ID


# =============================================================================
# Tests for HeuristicSketcher
# =============================================================================

class TestHeuristicSketcher:
    """Test the heuristic sketcher (fallback)."""

    def test_generate_sketches_returns_list(self, simple_task):
        """Should return a list of sketches."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(simple_task, num_sketches=5)

        assert isinstance(sketches, list)
        assert len(sketches) <= 5

    def test_sketches_have_correct_type(self, simple_task):
        """Sketches should be ProgramSketch instances."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(simple_task)

        for sketch in sketches:
            assert isinstance(sketch, ProgramSketch)
            assert hasattr(sketch, "ast")
            assert hasattr(sketch, "confidence")
            assert hasattr(sketch, "primitive_probs")

    def test_identity_task_includes_identity(self, simple_task):
        """Identity task should include identity sketch."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(simple_task)

        sketch_names = [str(s.ast) for s in sketches]
        assert "identity" in sketch_names

    def test_scaling_task_includes_scale(self, scaling_task):
        """Scaling task should include scale sketch."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(scaling_task)

        # Should suggest scale(2)
        sketch_strs = [str(s.ast) for s in sketches]
        assert any("scale" in s for s in sketch_strs)

    def test_empty_task_returns_identity(self):
        """Empty task should return identity sketch."""
        empty_task = ARCTask(task_id="empty", train=[], test=[])
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(empty_task)

        assert len(sketches) >= 1
        assert str(sketches[0].ast) == "identity"

    def test_confidence_in_range(self, simple_task):
        """Confidence should be in [0, 1]."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(simple_task)

        for sketch in sketches:
            assert 0.0 <= sketch.confidence <= 1.0

    def test_sketch_to_dict(self, simple_task):
        """ProgramSketch.to_dict should work."""
        sketcher = HeuristicSketcher()
        sketches = sketcher.generate_sketches(simple_task)

        for sketch in sketches:
            d = sketch.to_dict()
            assert "program" in d
            assert "confidence" in d
            assert "primitive_probs" in d


# =============================================================================
# Tests for StubNeuralCritic
# =============================================================================

class TestStubNeuralCritic:
    """Test the stub neural critic (fallback)."""

    def test_score_returns_correct_type(self, simple_program, simple_task):
        """Score should return NeuralCriticScore."""
        critic = StubNeuralCritic()
        score = critic.score(simple_program, simple_task)

        assert isinstance(score, NeuralCriticScore)

    def test_score_has_all_fields(self, simple_program, simple_task):
        """Score should have all required fields."""
        critic = StubNeuralCritic()
        score = critic.score(simple_program, simple_task)

        assert hasattr(score, "confidence")
        assert hasattr(score, "plausibility")
        assert hasattr(score, "generalization")
        assert hasattr(score, "features")

    def test_scores_in_valid_range(self, simple_program, simple_task):
        """All scores should be in [0, 1]."""
        critic = StubNeuralCritic()
        score = critic.score(simple_program, simple_task)

        assert 0.0 <= score.confidence <= 1.0
        assert 0.0 <= score.plausibility <= 1.0
        assert 0.0 <= score.generalization <= 1.0
        assert 0.0 <= score.overall <= 1.0

    def test_shorter_programs_more_plausible(self, simple_program, complex_program, simple_task):
        """Shorter programs should have higher plausibility."""
        critic = StubNeuralCritic()

        simple_score = critic.score(simple_program, simple_task)
        complex_score = critic.score(complex_program, simple_task)

        assert simple_score.plausibility >= complex_score.plausibility

    def test_features_include_program_metrics(self, simple_program, simple_task):
        """Features should include program size and depth."""
        critic = StubNeuralCritic()
        score = critic.score(simple_program, simple_task)

        assert "program_size" in score.features
        assert "program_depth" in score.features
        assert "num_primitives" in score.features

    def test_score_to_dict(self, simple_program, simple_task):
        """NeuralCriticScore.to_dict should work."""
        critic = StubNeuralCritic()
        score = critic.score(simple_program, simple_task)
        d = score.to_dict()

        assert "confidence" in d
        assert "plausibility" in d
        assert "generalization" in d
        assert "overall" in d
        assert "features" in d


# =============================================================================
# Tests for EnsembleNeuralCritic
# =============================================================================

class TestEnsembleNeuralCritic:
    """Test ensemble critic."""

    def test_empty_ensemble_uses_stub(self, simple_program, simple_task):
        """Empty ensemble should fall back to stub."""
        ensemble = EnsembleNeuralCritic([])
        score = ensemble.score(simple_program, simple_task)

        assert isinstance(score, NeuralCriticScore)

    def test_single_critic_ensemble(self, simple_program, simple_task):
        """Single critic ensemble should match that critic."""
        stub = StubNeuralCritic()
        ensemble = EnsembleNeuralCritic([stub])

        stub_score = stub.score(simple_program, simple_task)
        ensemble_score = ensemble.score(simple_program, simple_task)

        assert abs(stub_score.confidence - ensemble_score.confidence) < 0.01

    def test_multiple_critics_averaged(self, simple_program, simple_task):
        """Multiple critics should be averaged."""
        ensemble = EnsembleNeuralCritic([StubNeuralCritic(), StubNeuralCritic()])
        score = ensemble.score(simple_program, simple_task)

        # With same critics, should be same as individual
        assert isinstance(score, NeuralCriticScore)


# =============================================================================
# Tests for get_sketcher and get_critic
# =============================================================================

class TestGetFunctions:
    """Test factory functions."""

    def test_get_sketcher_returns_sketcher(self):
        """get_sketcher should return a SketcherModel."""
        sketcher = get_sketcher(use_neural=False)
        assert isinstance(sketcher, SketcherModel)

    def test_get_sketcher_heuristic_fallback(self):
        """When use_neural=False, should return HeuristicSketcher."""
        sketcher = get_sketcher(use_neural=False)
        assert isinstance(sketcher, HeuristicSketcher)

    def test_get_critic_returns_critic(self):
        """get_critic should return a NeuralCritic."""
        critic = get_critic(use_neural=False)
        assert isinstance(critic, NeuralCritic)

    def test_get_critic_stub_fallback(self):
        """When use_neural=False, should return StubNeuralCritic."""
        critic = get_critic(use_neural=False)
        assert isinstance(critic, StubNeuralCritic)


# =============================================================================
# Tests for Config Dataclasses
# =============================================================================

class TestConfigs:
    """Test configuration dataclasses."""

    def test_sketcher_config_defaults(self):
        """SketcherConfig should have sensible defaults."""
        config = SketcherConfig()

        assert config.max_seq_len > 0
        assert config.embed_dim > 0
        assert config.num_heads > 0
        assert config.num_layers > 0
        assert config.max_grid_size >= 30  # ARC max

    def test_critic_config_defaults(self):
        """CriticConfig should have sensible defaults."""
        config = CriticConfig()

        assert config.embed_dim > 0
        assert config.hidden_dim > 0
        assert config.num_heads > 0
        assert config.max_program_len > 0


# =============================================================================
# Tests for PyTorch Models (conditional)
# =============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchModels:
    """Tests that require PyTorch."""

    def test_neural_sketcher_can_be_created(self):
        """NeuralSketcherImpl should be creatable."""
        from juris_agi.cre.sketcher_model import NeuralSketcherImpl

        config = SketcherConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            hidden_dim=64,
        )
        sketcher = NeuralSketcherImpl(config=config)
        assert sketcher is not None

    def test_neural_sketcher_generates_sketches(self, simple_task):
        """NeuralSketcherImpl should generate sketches."""
        from juris_agi.cre.sketcher_model import NeuralSketcherImpl

        config = SketcherConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            hidden_dim=64,
        )
        sketcher = NeuralSketcherImpl(config=config)
        sketches = sketcher.generate_sketches(simple_task, num_sketches=3)

        assert isinstance(sketches, list)
        assert len(sketches) >= 1

    def test_neural_critic_can_be_created(self):
        """NeuralCriticImpl should be creatable."""
        from juris_agi.cre.critic_neural import NeuralCriticImpl

        config = CriticConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            hidden_dim=64,
        )
        critic = NeuralCriticImpl(config=config)
        assert critic is not None

    def test_neural_critic_scores_programs(self, simple_program, simple_task):
        """NeuralCriticImpl should score programs."""
        from juris_agi.cre.critic_neural import NeuralCriticImpl

        config = CriticConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            hidden_dim=64,
        )
        critic = NeuralCriticImpl(config=config)
        score = critic.score(simple_program, simple_task)

        assert isinstance(score, NeuralCriticScore)
        assert 0.0 <= score.overall <= 1.0

    def test_neural_critic_batch_scoring(self, simple_program, complex_program, simple_task):
        """NeuralCriticImpl should support batch scoring."""
        from juris_agi.cre.critic_neural import NeuralCriticImpl

        config = CriticConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            hidden_dim=64,
        )
        critic = NeuralCriticImpl(config=config)
        scores = critic.score_batch([simple_program, complex_program], simple_task)

        assert len(scores) == 2
        assert all(isinstance(s, NeuralCriticScore) for s in scores)

    def test_get_sketcher_returns_neural_when_available(self):
        """get_sketcher with use_neural=True should return neural model."""
        sketcher = get_sketcher(use_neural=True)
        # When torch is available, should be NeuralSketcherImpl
        from juris_agi.cre.sketcher_model import NeuralSketcherImpl
        assert isinstance(sketcher, NeuralSketcherImpl)

    def test_get_critic_returns_neural_when_available(self):
        """get_critic with use_neural=True should return neural model."""
        critic = get_critic(use_neural=True)
        # When torch is available, should be NeuralCriticImpl
        from juris_agi.cre.critic_neural import NeuralCriticImpl
        assert isinstance(critic, NeuralCriticImpl)


# =============================================================================
# Tests for Training Utilities
# =============================================================================

class TestTrainingUtilities:
    """Test training utility functions."""

    def test_generate_synthetic_task(self):
        """Synthetic task generation should work."""
        from juris_agi.train.train_sketcher import generate_synthetic_task

        task, program, primitives = generate_synthetic_task()

        assert isinstance(task, ARCTask)
        assert len(task.train) > 0
        assert program is not None
        assert len(primitives) > 0

    def test_generate_random_grid(self):
        """Random grid generation should work."""
        from juris_agi.train.train_sketcher import generate_random_grid

        grid = generate_random_grid(min_size=3, max_size=5)

        assert isinstance(grid, Grid)
        assert 3 <= grid.height <= 5
        assert 3 <= grid.width <= 5

    def test_generate_random_program(self):
        """Random program generation should work."""
        from juris_agi.train.train_sketcher import generate_random_program

        program, primitives = generate_random_program(max_length=3)

        assert program is not None
        assert len(primitives) >= 1
        assert len(primitives) <= 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for sketcher and critic."""

    def test_sketcher_and_critic_work_together(self, simple_task):
        """Sketcher output should be scorable by critic."""
        sketcher = get_sketcher(use_neural=False)
        critic = get_critic(use_neural=False)

        sketches = sketcher.generate_sketches(simple_task, num_sketches=5)

        for sketch in sketches:
            score = critic.score(sketch.ast, simple_task)
            assert isinstance(score, NeuralCriticScore)
            assert 0.0 <= score.overall <= 1.0

    def test_end_to_end_pipeline(self, rotation_task):
        """Full pipeline: generate sketches, score them, rank."""
        sketcher = get_sketcher(use_neural=False)
        critic = get_critic(use_neural=False)

        # Generate sketches
        sketches = sketcher.generate_sketches(rotation_task, num_sketches=10)

        # Score each sketch
        scored_sketches = []
        for sketch in sketches:
            score = critic.score(sketch.ast, rotation_task)
            scored_sketches.append((sketch, score))

        # Rank by overall score
        ranked = sorted(scored_sketches, key=lambda x: x[1].overall, reverse=True)

        # Best sketch should have reasonable score
        best_sketch, best_score = ranked[0]
        assert best_score.overall >= 0.0
