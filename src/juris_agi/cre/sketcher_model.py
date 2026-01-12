"""
Neural Sketcher Model - outputs distribution over primitive sequences.

This module provides:
1. GridEncoder: encodes input/output grid pairs into embeddings
2. SketcherTransformer: transformer-based sequence generator over primitive IDs
3. NeuralSketcher: top-level API for generating program sketches

Falls back to HeuristicSketcher when PyTorch is unavailable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math

from ..core.types import Grid, ARCTask
from ..dsl.ast import ASTNode, PrimitiveNode, ComposeNode, LiteralNode
from ..dsl.primitives import list_primitives

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

# Primitive vocabulary - maps primitive names to IDs
PRIMITIVE_NAMES = list_primitives()
PRIMITIVE_TO_ID = {name: i for i, name in enumerate(PRIMITIVE_NAMES)}
ID_TO_PRIMITIVE = {i: name for i, name in enumerate(PRIMITIVE_NAMES)}
NUM_PRIMITIVES = len(PRIMITIVE_NAMES)

# Special tokens
PAD_ID = NUM_PRIMITIVES
SOS_ID = NUM_PRIMITIVES + 1  # Start of sequence
EOS_ID = NUM_PRIMITIVES + 2  # End of sequence
VOCAB_SIZE = NUM_PRIMITIVES + 3

# ARC colors (0-9)
NUM_COLORS = 10

# Default model config
DEFAULT_MAX_SEQ_LEN = 8
DEFAULT_EMBED_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 3
DEFAULT_HIDDEN_DIM = 256


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ProgramSketch:
    """A program sketch with confidence scores."""
    ast: ASTNode
    confidence: float
    primitive_probs: Dict[str, float]  # Per-primitive confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "program": str(self.ast),
            "confidence": self.confidence,
            "primitive_probs": self.primitive_probs,
        }


@dataclass
class SketcherConfig:
    """Configuration for neural sketcher model."""
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    num_layers: int = DEFAULT_NUM_LAYERS
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout: float = 0.1
    max_grid_size: int = 30  # ARC grids are at most 30x30


# =============================================================================
# Abstract Interface
# =============================================================================

class SketcherModel(ABC):
    """
    Abstract base class for program sketcher models.

    The sketcher generates initial program candidates that are
    then verified and refined by symbolic components.
    """

    @abstractmethod
    def generate_sketches(
        self,
        task: ARCTask,
        num_sketches: int = 10,
    ) -> List[ProgramSketch]:
        """
        Generate program sketches for a task.

        Args:
            task: The ARC task
            num_sketches: Number of sketches to generate

        Returns:
            List of program sketches with confidence scores
        """
        pass


# =============================================================================
# PyTorch Models (only defined if torch available)
# =============================================================================

if TORCH_AVAILABLE:

    class GridEncoder(nn.Module):
        """
        Encodes ARC grids into fixed-size embeddings.

        Uses a combination of:
        1. Per-cell color embeddings
        2. Positional encodings
        3. CNN for local patterns
        4. Mean pooling to fixed size
        """

        def __init__(self, config: SketcherConfig):
            super().__init__()
            self.config = config

            # Color embedding
            self.color_embed = nn.Embedding(NUM_COLORS, config.embed_dim // 2)

            # Positional encoding
            self.pos_embed = nn.Embedding(
                config.max_grid_size * config.max_grid_size,
                config.embed_dim // 2
            )

            # CNN for local patterns
            self.conv1 = nn.Conv2d(config.embed_dim, config.embed_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(config.embed_dim, config.embed_dim, 3, padding=1)

            # Layer norm
            self.layer_norm = nn.LayerNorm(config.embed_dim)

        def forward(self, grid: torch.Tensor) -> torch.Tensor:
            """
            Encode a grid tensor.

            Args:
                grid: (batch, height, width) integer tensor of colors

            Returns:
                (batch, embed_dim) encoding
            """
            batch_size, height, width = grid.shape

            # Flatten grid for embedding
            flat_grid = grid.view(batch_size, -1)  # (batch, h*w)

            # Color embeddings
            color_emb = self.color_embed(flat_grid)  # (batch, h*w, embed_dim/2)

            # Position embeddings
            positions = torch.arange(
                height * width, device=grid.device
            ).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embed(positions)  # (batch, h*w, embed_dim/2)

            # Combine
            combined = torch.cat([color_emb, pos_emb], dim=-1)  # (batch, h*w, embed_dim)

            # Reshape for CNN
            h_pad = self.config.max_grid_size
            w_pad = self.config.max_grid_size

            # Pad to fixed size
            padded = torch.zeros(
                batch_size, h_pad * w_pad, self.config.embed_dim,
                device=grid.device
            )
            padded[:, :height * width, :] = combined

            # Reshape to 2D for conv
            reshaped = padded.view(batch_size, h_pad, w_pad, -1)
            reshaped = reshaped.permute(0, 3, 1, 2)  # (batch, embed, h, w)

            # Apply convolutions
            x = F.relu(self.conv1(reshaped))
            x = F.relu(self.conv2(x))

            # Global average pooling
            x = x.mean(dim=[2, 3])  # (batch, embed_dim)

            return self.layer_norm(x)


    class TaskEncoder(nn.Module):
        """
        Encodes a full ARC task (multiple input/output pairs).

        Combines encodings of all training pairs into a single embedding.
        """

        def __init__(self, config: SketcherConfig):
            super().__init__()
            self.config = config
            self.grid_encoder = GridEncoder(config)

            # Combine input and output embeddings
            self.pair_combiner = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.embed_dim),
            )

            # Self-attention over pairs
            self.pair_attention = nn.MultiheadAttention(
                config.embed_dim, config.num_heads,
                dropout=config.dropout, batch_first=True
            )

            self.layer_norm = nn.LayerNorm(config.embed_dim)

        def forward(
            self,
            input_grids: List[torch.Tensor],
            output_grids: List[torch.Tensor]
        ) -> torch.Tensor:
            """
            Encode a task.

            Args:
                input_grids: List of (batch, h, w) tensors
                output_grids: List of (batch, h, w) tensors

            Returns:
                (batch, embed_dim) task encoding
            """
            pair_embeddings = []

            for inp, out in zip(input_grids, output_grids):
                inp_emb = self.grid_encoder(inp)
                out_emb = self.grid_encoder(out)
                combined = torch.cat([inp_emb, out_emb], dim=-1)
                pair_emb = self.pair_combiner(combined)
                pair_embeddings.append(pair_emb)

            # Stack pairs
            pairs = torch.stack(pair_embeddings, dim=1)  # (batch, num_pairs, embed)

            # Self-attention over pairs
            attended, _ = self.pair_attention(pairs, pairs, pairs)
            attended = self.layer_norm(pairs + attended)

            # Mean pool over pairs
            return attended.mean(dim=1)


    class SketcherTransformer(nn.Module):
        """
        Transformer decoder for generating primitive sequences.

        Takes task encoding as context and generates a sequence of
        primitive IDs that form a program sketch.
        """

        def __init__(self, config: SketcherConfig):
            super().__init__()
            self.config = config

            # Task encoder
            self.task_encoder = TaskEncoder(config)

            # Token embedding for primitives
            self.token_embed = nn.Embedding(VOCAB_SIZE, config.embed_dim)

            # Positional encoding for sequence
            self.pos_encoding = nn.Parameter(
                self._create_sinusoidal_encoding(config.max_seq_len, config.embed_dim)
            )

            # Transformer decoder layers
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=config.num_layers
            )

            # Output projection to vocabulary
            self.output_proj = nn.Linear(config.embed_dim, VOCAB_SIZE)

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

        def forward(
            self,
            input_grids: List[torch.Tensor],
            output_grids: List[torch.Tensor],
            target_seq: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass for training.

            Args:
                input_grids: List of input grid tensors
                output_grids: List of output grid tensors
                target_seq: (batch, seq_len) target primitive sequence

            Returns:
                (batch, seq_len, vocab_size) logits
            """
            # Encode task
            task_encoding = self.task_encoder(input_grids, output_grids)
            task_encoding = task_encoding.unsqueeze(1)  # (batch, 1, embed)

            # Embed target sequence
            if target_seq is None:
                # Create SOS token for inference
                batch_size = input_grids[0].shape[0]
                target_seq = torch.full(
                    (batch_size, 1), SOS_ID,
                    dtype=torch.long, device=input_grids[0].device
                )

            seq_len = target_seq.shape[1]
            token_emb = self.token_embed(target_seq)  # (batch, seq, embed)
            token_emb = token_emb + self.pos_encoding[:seq_len]

            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=target_seq.device
            )

            # Decode
            decoded = self.transformer_decoder(
                token_emb, task_encoding, tgt_mask=causal_mask
            )

            # Project to vocabulary
            logits = self.output_proj(decoded)

            return logits

        @torch.no_grad()
        def generate(
            self,
            input_grids: List[torch.Tensor],
            output_grids: List[torch.Tensor],
            num_samples: int = 10,
            temperature: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Generate primitive sequences autoregressively.

            Args:
                input_grids: List of input grid tensors
                output_grids: List of output grid tensors
                num_samples: Number of sequences to generate
                temperature: Sampling temperature

            Returns:
                sequences: (num_samples, max_seq_len) generated sequences
                log_probs: (num_samples,) sequence log probabilities
            """
            self.eval()
            device = input_grids[0].device
            batch_size = input_grids[0].shape[0]

            # Replicate for sampling
            input_grids_rep = [g.repeat(num_samples, 1, 1) for g in input_grids]
            output_grids_rep = [g.repeat(num_samples, 1, 1) for g in output_grids]

            # Encode task
            task_encoding = self.task_encoder(input_grids_rep, output_grids_rep)
            task_encoding = task_encoding.unsqueeze(1)

            # Initialize with SOS
            sequences = torch.full(
                (num_samples, 1), SOS_ID, dtype=torch.long, device=device
            )
            log_probs = torch.zeros(num_samples, device=device)

            for step in range(self.config.max_seq_len):
                # Embed current sequence
                token_emb = self.token_embed(sequences)
                token_emb = token_emb + self.pos_encoding[:sequences.shape[1]]

                # Decode
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    sequences.shape[1], device=device
                )
                decoded = self.transformer_decoder(
                    token_emb, task_encoding, tgt_mask=causal_mask
                )

                # Get logits for last position
                logits = self.output_proj(decoded[:, -1])  # (num_samples, vocab)

                # Apply temperature
                logits = logits / temperature

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (num_samples, 1)

                # Update log probs
                token_probs = probs.gather(1, next_token).squeeze(-1)
                log_probs = log_probs + torch.log(token_probs + 1e-10)

                # Append to sequence
                sequences = torch.cat([sequences, next_token], dim=1)

                # Check for EOS
                if (next_token == EOS_ID).all():
                    break

            return sequences[:, 1:], log_probs  # Remove SOS token


    class NeuralSketcherImpl(SketcherModel):
        """
        Neural network-based sketcher implementation.

        Uses a transformer model to generate program sketches.
        """

        def __init__(
            self,
            config: Optional[SketcherConfig] = None,
            model_path: Optional[str] = None,
        ):
            self.config = config or SketcherConfig()
            self.model = SketcherTransformer(self.config)
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
                print(f"Warning: Failed to load model weights: {e}")

        def _grid_to_tensor(self, grid: Grid) -> torch.Tensor:
            """Convert Grid to tensor."""
            return torch.tensor(grid.data, dtype=torch.long, device=self.device)

        def _sequence_to_ast(
            self,
            sequence: torch.Tensor,
            probs: Dict[int, float]
        ) -> Optional[ASTNode]:
            """Convert a primitive sequence to an AST."""
            primitives = []

            for token_id in sequence.tolist():
                if token_id == EOS_ID or token_id == PAD_ID:
                    break
                if token_id < NUM_PRIMITIVES:
                    primitives.append(ID_TO_PRIMITIVE[token_id])

            if not primitives:
                return None

            if len(primitives) == 1:
                return PrimitiveNode(primitives[0])

            # Create composition
            nodes = [PrimitiveNode(name) for name in primitives]
            return ComposeNode(nodes)

        def generate_sketches(
            self,
            task: ARCTask,
            num_sketches: int = 10,
        ) -> List[ProgramSketch]:
            """Generate sketches using neural model."""
            if not task.train:
                return [self._identity_sketch()]

            # Prepare input tensors
            input_grids = [
                self._grid_to_tensor(pair.input).unsqueeze(0)
                for pair in task.train
            ]
            output_grids = [
                self._grid_to_tensor(pair.output).unsqueeze(0)
                for pair in task.train
            ]

            # Generate sequences
            self.model.eval()
            with torch.no_grad():
                sequences, log_probs = self.model.generate(
                    input_grids, output_grids,
                    num_samples=num_sketches * 2,  # Generate extra, filter duplicates
                    temperature=0.8,
                )

            # Convert to sketches
            sketches = []
            seen = set()

            probs = F.softmax(-log_probs, dim=-1)  # Convert log probs to probs

            for seq, log_prob in zip(sequences, log_probs):
                # Get per-primitive probabilities
                prim_probs = {}
                for i, token_id in enumerate(seq.tolist()):
                    if token_id < NUM_PRIMITIVES:
                        name = ID_TO_PRIMITIVE[token_id]
                        prim_probs[name] = 1.0  # Placeholder

                ast = self._sequence_to_ast(seq, prim_probs)
                if ast is None:
                    continue

                ast_str = str(ast)
                if ast_str in seen:
                    continue
                seen.add(ast_str)

                # Compute confidence from log prob
                confidence = float(torch.exp(log_prob).clamp(0, 1))

                sketches.append(ProgramSketch(
                    ast=ast,
                    confidence=confidence,
                    primitive_probs=prim_probs,
                ))

                if len(sketches) >= num_sketches:
                    break

            # Sort by confidence
            sketches.sort(key=lambda s: s.confidence, reverse=True)

            # Fall back to heuristic if no valid sketches
            if not sketches:
                return HeuristicSketcher().generate_sketches(task, num_sketches)

            return sketches

        def _identity_sketch(self) -> ProgramSketch:
            """Create identity sketch."""
            return ProgramSketch(
                ast=PrimitiveNode("identity"),
                confidence=1.0,
                primitive_probs={"identity": 1.0},
            )


# =============================================================================
# Heuristic Fallback (always available)
# =============================================================================

class HeuristicSketcher(SketcherModel):
    """
    Heuristic-based sketcher (no neural network).

    Uses simple heuristics based on input/output analysis.
    """

    def generate_sketches(
        self,
        task: ARCTask,
        num_sketches: int = 10,
    ) -> List[ProgramSketch]:
        """Generate sketches based on heuristics."""
        sketches = []

        # Analyze first training pair for hints
        if not task.train:
            return [self._identity_sketch()]

        pair = task.train[0]
        input_grid = pair.input
        output_grid = pair.output

        # Check for identity
        if input_grid == output_grid:
            sketches.append(self._identity_sketch())

        # Check for simple transformations
        sketches.extend(self._generate_transform_sketches(input_grid, output_grid))

        # Check for dimension changes
        sketches.extend(self._generate_dimension_sketches(input_grid, output_grid))

        # Check for color operations
        sketches.extend(self._generate_color_sketches(input_grid, output_grid))

        # Limit to requested number
        return sketches[:num_sketches]

    def _identity_sketch(self) -> ProgramSketch:
        """Create identity sketch."""
        return ProgramSketch(
            ast=PrimitiveNode("identity"),
            confidence=1.0,
            primitive_probs={"identity": 1.0},
        )

    def _generate_transform_sketches(
        self,
        input_grid: Grid,
        output_grid: Grid,
    ) -> List[ProgramSketch]:
        """Generate geometric transformation sketches."""
        sketches = []

        # Try rotations
        for n in [1, 2, 3]:
            sketches.append(ProgramSketch(
                ast=PrimitiveNode("rotate90", [LiteralNode(n)]),
                confidence=0.5,
                primitive_probs={"rotate90": 0.5},
            ))

        # Try reflections
        sketches.append(ProgramSketch(
            ast=PrimitiveNode("reflect_h"),
            confidence=0.5,
            primitive_probs={"reflect_h": 0.5},
        ))
        sketches.append(ProgramSketch(
            ast=PrimitiveNode("reflect_v"),
            confidence=0.5,
            primitive_probs={"reflect_v": 0.5},
        ))

        # Try transpose
        if input_grid.height == output_grid.width and input_grid.width == output_grid.height:
            sketches.append(ProgramSketch(
                ast=PrimitiveNode("transpose"),
                confidence=0.7,
                primitive_probs={"transpose": 0.7},
            ))

        return sketches

    def _generate_dimension_sketches(
        self,
        input_grid: Grid,
        output_grid: Grid,
    ) -> List[ProgramSketch]:
        """Generate dimension-changing sketches."""
        sketches = []

        # Check for cropping
        if output_grid.height < input_grid.height or output_grid.width < input_grid.width:
            sketches.append(ProgramSketch(
                ast=PrimitiveNode("crop_to_content"),
                confidence=0.6,
                primitive_probs={"crop_to_content": 0.6},
            ))

        # Check for scaling
        if input_grid.height > 0 and input_grid.width > 0:
            h_ratio = output_grid.height / input_grid.height
            w_ratio = output_grid.width / input_grid.width

            if h_ratio == w_ratio and h_ratio == int(h_ratio) and h_ratio > 1:
                factor = int(h_ratio)
                sketches.append(ProgramSketch(
                    ast=PrimitiveNode("scale", [LiteralNode(factor)]),
                    confidence=0.8,
                    primitive_probs={"scale": 0.8},
                ))

        # Check for tiling
        if output_grid.height == input_grid.height and output_grid.width > input_grid.width:
            if output_grid.width % input_grid.width == 0:
                n = output_grid.width // input_grid.width
                sketches.append(ProgramSketch(
                    ast=PrimitiveNode("tile_h", [LiteralNode(n)]),
                    confidence=0.7,
                    primitive_probs={"tile_h": 0.7},
                ))

        if output_grid.width == input_grid.width and output_grid.height > input_grid.height:
            if output_grid.height % input_grid.height == 0:
                n = output_grid.height // input_grid.height
                sketches.append(ProgramSketch(
                    ast=PrimitiveNode("tile_v", [LiteralNode(n)]),
                    confidence=0.7,
                    primitive_probs={"tile_v": 0.7},
                ))

        return sketches

    def _generate_color_sketches(
        self,
        input_grid: Grid,
        output_grid: Grid,
    ) -> List[ProgramSketch]:
        """Generate color-based sketches."""
        sketches = []

        # Check for simple recoloring
        input_palette = input_grid.palette
        output_palette = output_grid.palette

        # If same dimensions and different palettes, suggest recolor
        if input_grid.shape == output_grid.shape:
            if input_palette != output_palette:
                # Suggest recolor for each color change
                for from_color in input_palette - output_palette:
                    for to_color in output_palette - input_palette:
                        sketches.append(ProgramSketch(
                            ast=PrimitiveNode("recolor", [
                                LiteralNode(from_color),
                                LiteralNode(to_color)
                            ]),
                            confidence=0.6,
                            primitive_probs={"recolor": 0.6},
                        ))
                        break  # Only suggest first mapping
                    break

        return sketches


# =============================================================================
# Public API - auto-selects best available implementation
# =============================================================================

def get_sketcher(
    use_neural: bool = True,
    config: Optional[SketcherConfig] = None,
    model_path: Optional[str] = None,
) -> SketcherModel:
    """
    Get the best available sketcher model.

    Args:
        use_neural: Whether to use neural model (if available)
        config: Configuration for neural model
        model_path: Path to pretrained weights

    Returns:
        SketcherModel instance (neural if available and requested, otherwise heuristic)
    """
    if use_neural and TORCH_AVAILABLE:
        return NeuralSketcherImpl(config=config, model_path=model_path)
    return HeuristicSketcher()


# Convenience alias
NeuralSketcher = NeuralSketcherImpl if TORCH_AVAILABLE else HeuristicSketcher
