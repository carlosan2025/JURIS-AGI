"""
Neural Critic - provides advisory scoring (NO JURISDICTION).

This module provides:
1. ProgramEncoder: encodes program AST into embeddings
2. GeneralizationCritic: scores programs for generalization ability
3. NeuralCritic: top-level API for scoring candidate programs

The neural critic provides soft signals but cannot veto solutions.
Only the symbolic critic (which verifies programs on training examples)
has veto power.

Falls back to heuristic-based StubNeuralCritic when PyTorch is unavailable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math

from ..core.types import Grid, ARCTask
from ..dsl.ast import ASTNode, PrimitiveNode, ComposeNode, walk_ast
from ..dsl.primitives import list_primitives, PRIMITIVES

# Try importing PyTorch, fall back gracefully if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# =============================================================================
# Constants and Configuration
# =============================================================================

# Primitive vocabulary
PRIMITIVE_NAMES = list_primitives()
PRIMITIVE_TO_ID = {name: i for i, name in enumerate(PRIMITIVE_NAMES)}
NUM_PRIMITIVES = len(PRIMITIVE_NAMES)

# Special tokens
UNKNOWN_PRIM_ID = NUM_PRIMITIVES
COMPOSE_ID = NUM_PRIMITIVES + 1
LITERAL_ID = NUM_PRIMITIVES + 2
VOCAB_SIZE = NUM_PRIMITIVES + 3

# ARC colors
NUM_COLORS = 10

# Default config
DEFAULT_EMBED_DIM = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_MAX_PROGRAM_LEN = 32


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class NeuralCriticScore:
    """Score from neural critic."""
    confidence: float  # 0.0 to 1.0 - how confident is the critic
    plausibility: float  # 0.0 to 1.0 - how plausible is this program
    generalization: float  # 0.0 to 1.0 - expected generalization ability
    features: Dict[str, float]  # Additional features for debugging

    @property
    def overall(self) -> float:
        """Weighted overall score."""
        return (
            self.confidence * 0.4 +
            self.plausibility * 0.3 +
            self.generalization * 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "plausibility": self.plausibility,
            "generalization": self.generalization,
            "overall": self.overall,
            "features": self.features,
        }


@dataclass
class CriticConfig:
    """Configuration for neural critic model."""
    embed_dim: int = DEFAULT_EMBED_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    num_layers: int = DEFAULT_NUM_LAYERS
    max_program_len: int = DEFAULT_MAX_PROGRAM_LEN
    dropout: float = 0.1
    max_grid_size: int = 30


# =============================================================================
# Abstract Interface
# =============================================================================

class NeuralCritic(ABC):
    """
    Abstract neural critic interface.

    Neural critics provide advisory signals but have NO JURISDICTION.
    They cannot veto solutions - only the symbolic critic can.
    """

    @abstractmethod
    def score(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> NeuralCriticScore:
        """
        Score a program for a task.

        Returns a soft score, not a hard pass/fail.
        """
        pass

    def score_batch(
        self,
        programs: List[ASTNode],
        task: ARCTask,
    ) -> List[NeuralCriticScore]:
        """Score multiple programs for efficiency."""
        return [self.score(p, task) for p in programs]


# =============================================================================
# PyTorch Models (only defined if torch available)
# =============================================================================

if TORCH_AVAILABLE:

    class ProgramEncoder(nn.Module):
        """
        Encodes a program AST into a fixed-size embedding.

        Uses a tree-structured encoder that processes primitives
        and their compositions.
        """

        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config

            # Primitive embedding
            self.prim_embed = nn.Embedding(VOCAB_SIZE, config.embed_dim)

            # Tree composition network
            self.compose_mlp = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embed_dim),
            )

            # Position encoding for sequences
            self.pos_encoding = nn.Parameter(
                self._create_sinusoidal_encoding(config.max_program_len, config.embed_dim)
            )

            # Transformer for sequence encoding
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_layers
            )

            # Final projection
            self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
            self.layer_norm = nn.LayerNorm(config.embed_dim)

        def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
            """Create sinusoidal positional encoding."""
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            return pe

        def _ast_to_token_ids(self, ast: ASTNode) -> List[int]:
            """Convert AST to a sequence of token IDs."""
            nodes = walk_ast(ast)
            token_ids = []

            for node in nodes:
                if isinstance(node, PrimitiveNode):
                    token_id = PRIMITIVE_TO_ID.get(node.name, UNKNOWN_PRIM_ID)
                    token_ids.append(token_id)
                elif isinstance(node, ComposeNode):
                    token_ids.append(COMPOSE_ID)
                else:
                    token_ids.append(LITERAL_ID)

            return token_ids[:self.config.max_program_len]

        def forward(self, programs: List[ASTNode]) -> torch.Tensor:
            """
            Encode a batch of programs.

            Args:
                programs: List of AST nodes

            Returns:
                (batch, embed_dim) program embeddings
            """
            batch_size = len(programs)

            # Convert to token sequences
            sequences = [self._ast_to_token_ids(p) for p in programs]
            max_len = max(len(s) for s in sequences)
            max_len = min(max_len, self.config.max_program_len)

            # Pad sequences
            device = self.prim_embed.weight.device
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

            for i, seq in enumerate(sequences):
                seq_len = min(len(seq), max_len)
                padded[i, :seq_len] = torch.tensor(seq[:seq_len], device=device)
                mask[i, :seq_len] = False

            # Embed tokens
            embedded = self.prim_embed(padded)  # (batch, seq, embed)
            embedded = embedded + self.pos_encoding[:max_len]

            # Transformer encoding
            encoded = self.transformer(embedded, src_key_padding_mask=mask)

            # Mean pooling over non-padded positions
            mask_float = (~mask).float().unsqueeze(-1)  # (batch, seq, 1)
            pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

            # Final projection
            output = self.output_proj(pooled)
            return self.layer_norm(output)


    class TaskContextEncoder(nn.Module):
        """
        Encodes task context (input/output grids) for the critic.

        Uses a simplified grid encoder focused on extracting features
        relevant for judging program quality.
        """

        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config

            # Color embedding
            self.color_embed = nn.Embedding(NUM_COLORS, config.embed_dim // 2)

            # Dimension embedding (for grid size features)
            self.dim_embed = nn.Linear(4, config.embed_dim // 2)  # h, w for input and output

            # Pair combiner
            self.pair_mlp = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.embed_dim),
            )

            # Aggregator for multiple pairs
            self.aggregator = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
            )

        def _encode_grid_simple(self, grid: torch.Tensor) -> torch.Tensor:
            """Simple grid encoding - color histogram + dimensions."""
            batch_size = grid.shape[0]
            device = grid.device

            # Color histogram
            histograms = []
            for i in range(batch_size):
                hist = torch.histc(
                    grid[i].float(), bins=NUM_COLORS, min=0, max=NUM_COLORS - 1
                )
                hist = hist / hist.sum().clamp(min=1)  # Normalize
                histograms.append(hist)

            hist_tensor = torch.stack(histograms)  # (batch, NUM_COLORS)

            # Embed histogram
            color_emb = self.color_embed.weight.mean(dim=0)  # Average embedding
            color_features = hist_tensor @ self.color_embed.weight  # (batch, embed/2)

            return color_features

        def forward(
            self,
            input_grids: List[torch.Tensor],
            output_grids: List[torch.Tensor]
        ) -> torch.Tensor:
            """
            Encode task context.

            Args:
                input_grids: List of (batch, h, w) tensors
                output_grids: List of (batch, h, w) tensors

            Returns:
                (batch, embed_dim) context encoding
            """
            pair_embeddings = []
            batch_size = input_grids[0].shape[0]
            device = input_grids[0].device

            for inp, out in zip(input_grids, output_grids):
                # Get color features
                inp_color = self._encode_grid_simple(inp)
                out_color = self._encode_grid_simple(out)

                # Get dimension features
                inp_dims = torch.tensor(
                    [[inp.shape[1], inp.shape[2], out.shape[1], out.shape[2]]],
                    dtype=torch.float, device=device
                ).expand(batch_size, -1) / 30.0  # Normalize by max size

                dim_features = self.dim_embed(inp_dims)  # (batch, embed/2)

                # Combine
                inp_emb = torch.cat([inp_color, dim_features], dim=-1)
                out_emb = torch.cat([out_color, dim_features], dim=-1)
                combined = torch.cat([inp_emb, out_emb], dim=-1)

                pair_emb = self.pair_mlp(combined)
                pair_embeddings.append(pair_emb)

            # Aggregate pairs
            stacked = torch.stack(pair_embeddings, dim=1)  # (batch, num_pairs, embed)
            aggregated = self.aggregator(stacked.mean(dim=1))

            return aggregated


    class GeneralizationCriticModel(nn.Module):
        """
        Neural model for scoring program generalization.

        Takes (program, task_context) and outputs:
        - plausibility: how likely is this a valid program
        - generalization: expected generalization to unseen examples
        """

        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config

            # Encoders
            self.program_encoder = ProgramEncoder(config)
            self.context_encoder = TaskContextEncoder(config)

            # Scoring head
            self.scorer = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 3),  # plausibility, generalization, confidence
            )

        def forward(
            self,
            programs: List[ASTNode],
            input_grids: List[torch.Tensor],
            output_grids: List[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Score programs.

            Returns:
                plausibility: (batch,) plausibility scores
                generalization: (batch,) generalization scores
                confidence: (batch,) confidence scores
            """
            # Encode programs
            program_emb = self.program_encoder(programs)

            # Encode task context
            context_emb = self.context_encoder(input_grids, output_grids)

            # Expand context for each program (assumes 1 task for batch of programs)
            if context_emb.shape[0] == 1 and program_emb.shape[0] > 1:
                context_emb = context_emb.expand(program_emb.shape[0], -1)

            # Combine and score
            combined = torch.cat([program_emb, context_emb], dim=-1)
            scores = self.scorer(combined)

            # Apply sigmoid to get [0, 1] range
            scores = torch.sigmoid(scores)

            return scores[:, 0], scores[:, 1], scores[:, 2]


    class NeuralCriticImpl(NeuralCritic):
        """
        Neural network-based critic implementation.

        Scores programs using a learned model that considers
        both program structure and task context.
        """

        def __init__(
            self,
            config: Optional[CriticConfig] = None,
            model_path: Optional[str] = None,
        ):
            self.config = config or CriticConfig()
            self.model = GeneralizationCriticModel(self.config)
            self.model_path = model_path

            # Load pretrained weights if provided
            if model_path is not None:
                self._load_weights(model_path)

            # Move to available device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        def _load_weights(self, path: str) -> None:
            """Load model weights from file."""
            try:
                state_dict = torch.load(path, map_location="cpu")
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Failed to load critic weights: {e}")

        def _grid_to_tensor(self, grid: Grid) -> torch.Tensor:
            """Convert Grid to tensor."""
            return torch.tensor(grid.data, dtype=torch.long, device=self.device)

        def _extract_features(self, program: ASTNode) -> Dict[str, float]:
            """Extract program features for debugging."""
            return {
                "program_size": float(program.size()),
                "program_depth": float(program.depth()),
                "num_primitives": float(len([
                    n for n in walk_ast(program)
                    if isinstance(n, PrimitiveNode)
                ])),
            }

        def score(
            self,
            program: ASTNode,
            task: ARCTask,
        ) -> NeuralCriticScore:
            """Score a single program."""
            return self.score_batch([program], task)[0]

        def score_batch(
            self,
            programs: List[ASTNode],
            task: ARCTask,
        ) -> List[NeuralCriticScore]:
            """Score multiple programs efficiently."""
            if not programs or not task.train:
                return [self._neutral_score(p) for p in programs]

            # Prepare input tensors
            input_grids = [
                self._grid_to_tensor(pair.input).unsqueeze(0)
                for pair in task.train
            ]
            output_grids = [
                self._grid_to_tensor(pair.output).unsqueeze(0)
                for pair in task.train
            ]

            # Run model
            self.model.eval()
            with torch.no_grad():
                plausibility, generalization, confidence = self.model(
                    programs, input_grids, output_grids
                )

            # Convert to scores
            scores = []
            for i, program in enumerate(programs):
                scores.append(NeuralCriticScore(
                    confidence=float(confidence[i]),
                    plausibility=float(plausibility[i]),
                    generalization=float(generalization[i]),
                    features=self._extract_features(program),
                ))

            return scores

        def _neutral_score(self, program: ASTNode) -> NeuralCriticScore:
            """Return neutral score for edge cases."""
            return NeuralCriticScore(
                confidence=0.5,
                plausibility=0.5,
                generalization=0.5,
                features=self._extract_features(program),
            )


# =============================================================================
# Heuristic Fallback (always available)
# =============================================================================

class StubNeuralCritic(NeuralCritic):
    """
    Stub implementation of neural critic.

    Returns heuristic-based scores. Replace with actual neural model when available.
    """

    def score(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> NeuralCriticScore:
        """Return heuristic-based scores."""
        # Heuristic: shorter programs are more plausible
        program_size = program.size()
        program_depth = program.depth()

        # Count primitives
        num_primitives = len([
            n for n in walk_ast(program)
            if isinstance(n, PrimitiveNode)
        ])

        # Plausibility: prefer shorter, shallower programs
        plausibility = max(0.0, 1.0 - program_size * 0.05)
        plausibility = min(plausibility, max(0.0, 1.0 - program_depth * 0.1))

        # Generalization: prefer programs with common primitives
        common_primitives = {
            "identity", "rotate90", "reflect_h", "reflect_v",
            "transpose", "crop_to_content", "recolor",
        }

        common_count = 0
        total_prims = 0
        for node in walk_ast(program):
            if isinstance(node, PrimitiveNode):
                total_prims += 1
                if node.name in common_primitives:
                    common_count += 1

        generalization = common_count / max(1, total_prims)

        # Confidence: based on program complexity
        confidence = 0.5 + 0.3 * (1.0 - min(1.0, program_size / 10))

        return NeuralCriticScore(
            confidence=confidence,
            plausibility=plausibility,
            generalization=generalization,
            features={
                "program_size": float(program_size),
                "program_depth": float(program_depth),
                "num_primitives": float(num_primitives),
                "common_primitive_ratio": generalization,
            },
        )


class EnsembleNeuralCritic(NeuralCritic):
    """
    Ensemble of multiple neural critics.

    Averages scores from multiple critics for more robust predictions.
    """

    def __init__(self, critics: List[NeuralCritic]):
        self.critics = critics

    def score(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> NeuralCriticScore:
        """Average scores from all critics."""
        if not self.critics:
            return StubNeuralCritic().score(program, task)

        scores = [c.score(program, task) for c in self.critics]

        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        avg_plausibility = sum(s.plausibility for s in scores) / len(scores)
        avg_generalization = sum(s.generalization for s in scores) / len(scores)

        # Merge features
        all_features: Dict[str, float] = {}
        for s in scores:
            for k, v in s.features.items():
                if k in all_features:
                    all_features[k] = (all_features[k] + v) / 2
                else:
                    all_features[k] = v

        return NeuralCriticScore(
            confidence=avg_confidence,
            plausibility=avg_plausibility,
            generalization=avg_generalization,
            features=all_features,
        )


# =============================================================================
# Public API
# =============================================================================

def get_critic(
    use_neural: bool = True,
    config: Optional[CriticConfig] = None,
    model_path: Optional[str] = None,
) -> NeuralCritic:
    """
    Get the best available critic model.

    Args:
        use_neural: Whether to use neural model (if available)
        config: Configuration for neural model
        model_path: Path to pretrained weights

    Returns:
        NeuralCritic instance (neural if available and requested, otherwise stub)
    """
    if use_neural and TORCH_AVAILABLE:
        return NeuralCriticImpl(config=config, model_path=model_path)
    return StubNeuralCritic()


# Convenience alias
GeneralizationCritic = NeuralCriticImpl if TORCH_AVAILABLE else StubNeuralCritic
