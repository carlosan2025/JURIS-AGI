"""
Grid tokenization for various representations.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.types import Grid


@dataclass
class TokenizedGrid:
    """A tokenized representation of a grid."""
    tokens: np.ndarray  # Shape depends on tokenization scheme
    vocab_size: int
    special_tokens: Dict[str, int]
    original_shape: tuple


class GridTokenizer:
    """
    Tokenize grids for various purposes.

    Supports multiple tokenization schemes:
    - raw: Raw color values (0-9)
    - position: Include position encoding
    - patch: Patch-based tokenization
    """

    # Special tokens
    PAD_TOKEN = 10
    SEP_TOKEN = 11
    CLS_TOKEN = 12
    MASK_TOKEN = 13

    def __init__(
        self,
        scheme: str = "raw",
        max_height: int = 30,
        max_width: int = 30,
        patch_size: int = 1,
    ):
        self.scheme = scheme
        self.max_height = max_height
        self.max_width = max_width
        self.patch_size = patch_size

        self.special_tokens = {
            "PAD": self.PAD_TOKEN,
            "SEP": self.SEP_TOKEN,
            "CLS": self.CLS_TOKEN,
            "MASK": self.MASK_TOKEN,
        }

        # Vocab: 0-9 colors + special tokens
        self.vocab_size = 14

    def tokenize(self, grid: Grid) -> TokenizedGrid:
        """Tokenize a grid based on the configured scheme."""
        if self.scheme == "raw":
            return self._tokenize_raw(grid)
        elif self.scheme == "position":
            return self._tokenize_with_position(grid)
        elif self.scheme == "patch":
            return self._tokenize_patches(grid)
        else:
            raise ValueError(f"Unknown tokenization scheme: {self.scheme}")

    def _tokenize_raw(self, grid: Grid) -> TokenizedGrid:
        """Simple raw tokenization - flatten grid to 1D."""
        tokens = grid.data.flatten()
        return TokenizedGrid(
            tokens=tokens,
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            original_shape=grid.shape,
        )

    def _tokenize_with_position(self, grid: Grid) -> TokenizedGrid:
        """Tokenize with position encoding."""
        h, w = grid.shape
        # Create (color, row, col) tuples flattened
        tokens = np.zeros((h * w, 3), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                tokens[idx] = [grid[i, j], i, j]
        return TokenizedGrid(
            tokens=tokens,
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            original_shape=grid.shape,
        )

    def _tokenize_patches(self, grid: Grid) -> TokenizedGrid:
        """Patch-based tokenization."""
        h, w = grid.shape
        ph, pw = self.patch_size, self.patch_size

        # Number of patches
        num_patches_h = (h + ph - 1) // ph
        num_patches_w = (w + pw - 1) // pw

        # Pad grid if needed
        padded_h = num_patches_h * ph
        padded_w = num_patches_w * pw

        padded = np.full((padded_h, padded_w), self.PAD_TOKEN, dtype=np.int32)
        padded[:h, :w] = grid.data

        # Extract patches as flattened tokens
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = padded[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                patches.append(patch.flatten())

        tokens = np.array(patches, dtype=np.int32)
        return TokenizedGrid(
            tokens=tokens,
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            original_shape=grid.shape,
        )

    def detokenize(self, tokenized: TokenizedGrid) -> Grid:
        """Convert tokenized representation back to grid."""
        if self.scheme == "raw":
            h, w = tokenized.original_shape
            data = tokenized.tokens.reshape(h, w)
            return Grid(data)
        elif self.scheme == "position":
            h, w = tokenized.original_shape
            data = np.zeros((h, w), dtype=np.int32)
            for color, row, col in tokenized.tokens:
                if 0 <= row < h and 0 <= col < w:
                    data[row, col] = color
            return Grid(data)
        elif self.scheme == "patch":
            h, w = tokenized.original_shape
            ph, pw = self.patch_size, self.patch_size
            num_patches_h = (h + ph - 1) // ph
            num_patches_w = (w + pw - 1) // pw

            padded = np.zeros((num_patches_h * ph, num_patches_w * pw), dtype=np.int32)
            for idx, patch in enumerate(tokenized.tokens):
                i = idx // num_patches_w
                j = idx % num_patches_w
                padded[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = patch.reshape(ph, pw)

            return Grid(padded[:h, :w])
        else:
            raise ValueError(f"Unknown tokenization scheme: {self.scheme}")

    def pad_sequence(
        self,
        tokens: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """Pad a token sequence to max_length."""
        if len(tokens) >= max_length:
            return tokens[:max_length]

        padded = np.full(max_length, self.PAD_TOKEN, dtype=np.int32)
        padded[:len(tokens)] = tokens
        return padded
