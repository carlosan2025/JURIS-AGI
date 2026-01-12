"""
Counterfactual generation for robustness testing.

Generates variations of inputs to test program robustness.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ..core.types import Grid, ARCTask


@dataclass
class Counterfactual:
    """A counterfactual input for testing."""
    original: Grid
    modified: Grid
    modification_type: str
    modification_params: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = "same_transformation"  # or "graceful_failure"


class CounterfactualGenerator(ABC):
    """
    Abstract base for counterfactual generators.

    Generates modified inputs to test program robustness.
    """

    @abstractmethod
    def generate(
        self,
        grid: Grid,
        num_counterfactuals: int = 5,
    ) -> List[Counterfactual]:
        """
        Generate counterfactual inputs.

        Args:
            grid: Original input grid
            num_counterfactuals: Number of counterfactuals to generate

        Returns:
            List of counterfactual inputs
        """
        pass


class GridPerturbationGenerator(CounterfactualGenerator):
    """
    Generates counterfactuals by perturbing grid values.
    """

    def __init__(
        self,
        perturbation_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.perturbation_rate = perturbation_rate
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        grid: Grid,
        num_counterfactuals: int = 5,
    ) -> List[Counterfactual]:
        """Generate perturbed counterfactuals."""
        counterfactuals = []

        for i in range(num_counterfactuals):
            # Choose modification type
            mod_type = self.rng.choice([
                "pixel_noise",
                "color_swap",
                "shift",
                "flip_pixel",
            ])

            if mod_type == "pixel_noise":
                cf = self._pixel_noise(grid)
            elif mod_type == "color_swap":
                cf = self._color_swap(grid)
            elif mod_type == "shift":
                cf = self._shift(grid)
            else:
                cf = self._flip_pixel(grid)

            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def _pixel_noise(self, grid: Grid) -> Counterfactual:
        """Add random pixel noise."""
        modified = grid.copy()
        num_pixels = int(grid.height * grid.width * self.perturbation_rate)
        num_pixels = max(1, num_pixels)

        for _ in range(num_pixels):
            r = self.rng.integers(0, grid.height)
            c = self.rng.integers(0, grid.width)
            new_color = self.rng.integers(0, 10)
            modified[r, c] = new_color

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="pixel_noise",
            modification_params={"num_pixels": num_pixels},
        )

    def _color_swap(self, grid: Grid) -> Optional[Counterfactual]:
        """Swap two colors in the grid."""
        palette = list(grid.palette)
        if len(palette) < 2:
            return None

        c1, c2 = self.rng.choice(palette, size=2, replace=False)

        modified = grid.copy()
        mask1 = modified.data == c1
        mask2 = modified.data == c2
        modified.data[mask1] = c2
        modified.data[mask2] = c1

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="color_swap",
            modification_params={"color1": int(c1), "color2": int(c2)},
        )

    def _shift(self, grid: Grid) -> Counterfactual:
        """Shift grid content by small amount."""
        dr = self.rng.integers(-1, 2)
        dc = self.rng.integers(-1, 2)

        modified = Grid.zeros(grid.height, grid.width)
        for r in range(grid.height):
            for c in range(grid.width):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.height and 0 <= nc < grid.width:
                    modified[nr, nc] = grid[r, c]

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="shift",
            modification_params={"dr": dr, "dc": dc},
        )

    def _flip_pixel(self, grid: Grid) -> Counterfactual:
        """Flip a single pixel."""
        r = self.rng.integers(0, grid.height)
        c = self.rng.integers(0, grid.width)

        modified = grid.copy()
        old_val = modified[r, c]
        new_val = (old_val + 1) % 10
        modified[r, c] = new_val

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="flip_pixel",
            modification_params={"row": r, "col": c, "old": int(old_val), "new": int(new_val)},
        )


class StructuralCounterfactualGenerator(CounterfactualGenerator):
    """
    Generates structurally meaningful counterfactuals.

    Preserves structure while modifying content.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        grid: Grid,
        num_counterfactuals: int = 5,
    ) -> List[Counterfactual]:
        """Generate structural counterfactuals."""
        counterfactuals = []

        # Resize variations
        for scale in [0.5, 2.0]:
            cf = self._resize(grid, scale)
            if cf is not None:
                counterfactuals.append(cf)
                if len(counterfactuals) >= num_counterfactuals:
                    return counterfactuals

        # Padding variations
        for pad in [1, 2]:
            cf = self._pad(grid, pad)
            counterfactuals.append(cf)
            if len(counterfactuals) >= num_counterfactuals:
                return counterfactuals

        # Color remap
        cf = self._remap_colors(grid)
        if cf is not None:
            counterfactuals.append(cf)

        return counterfactuals[:num_counterfactuals]

    def _resize(self, grid: Grid, scale: float) -> Optional[Counterfactual]:
        """Resize grid."""
        new_h = max(1, int(grid.height * scale))
        new_w = max(1, int(grid.width * scale))

        if new_h > 30 or new_w > 30:
            return None

        # Simple nearest-neighbor resize
        modified = Grid.zeros(new_h, new_w)
        for r in range(new_h):
            for c in range(new_w):
                src_r = min(int(r / scale), grid.height - 1)
                src_c = min(int(c / scale), grid.width - 1)
                modified[r, c] = grid[src_r, src_c]

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="resize",
            modification_params={"scale": scale},
            expected_behavior="may_differ",
        )

    def _pad(self, grid: Grid, padding: int) -> Counterfactual:
        """Add padding around grid."""
        new_h = grid.height + 2 * padding
        new_w = grid.width + 2 * padding

        modified = Grid.zeros(new_h, new_w)
        for r in range(grid.height):
            for c in range(grid.width):
                modified[r + padding, c + padding] = grid[r, c]

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="pad",
            modification_params={"padding": padding},
            expected_behavior="may_differ",
        )

    def _remap_colors(self, grid: Grid) -> Optional[Counterfactual]:
        """Remap colors while preserving structure."""
        palette = list(grid.palette - {0})
        if not palette:
            return None

        # Create random permutation of colors
        new_palette = self.rng.permutation(palette)
        color_map = dict(zip(palette, new_palette))

        modified = grid.copy()
        for old, new in color_map.items():
            modified.data[grid.data == old] = new

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="color_remap",
            modification_params={"color_map": {int(k): int(v) for k, v in color_map.items()}},
            expected_behavior="same_structure",
        )


def create_counterfactual_generator(
    strategy: str = "perturbation",
) -> CounterfactualGenerator:
    """Create a counterfactual generator."""
    if strategy == "perturbation":
        return GridPerturbationGenerator()
    elif strategy == "structural":
        return StructuralCounterfactualGenerator()
    else:
        return GridPerturbationGenerator()


# ============================================================================
# Invariant-Preserving Counterfactual Generation
# ============================================================================

@dataclass
class InvariantSpec:
    """Specification of an invariant to preserve."""
    name: str
    preserve: bool = True
    value: Any = None  # Specific value to maintain


class InvariantPreservingGenerator(CounterfactualGenerator):
    """
    Generates counterfactuals that preserve specified invariants.

    This is useful for robustness testing where we want to verify
    that a program handles variations while maintaining key properties.
    """

    def __init__(
        self,
        invariants: Optional[List[InvariantSpec]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize invariant-preserving generator.

        Args:
            invariants: List of invariants to preserve
            seed: Random seed for reproducibility
        """
        self.invariants = invariants or []
        self.rng = np.random.default_rng(seed)

        # Build set of preserved invariant names for quick lookup
        self._preserve_dims = any(
            inv.name in ("dims_preserved", "consistent_output_dims")
            and inv.preserve
            for inv in self.invariants
        )
        self._preserve_palette = any(
            inv.name in ("palette_preserved", "palette_subset")
            and inv.preserve
            for inv in self.invariants
        )

    def generate(
        self,
        grid: Grid,
        num_counterfactuals: int = 5,
    ) -> List[Counterfactual]:
        """Generate counterfactuals while preserving invariants."""
        counterfactuals: List[Counterfactual] = []

        for _ in range(num_counterfactuals):
            cf = self._generate_one(grid)
            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def _generate_one(self, grid: Grid) -> Optional[Counterfactual]:
        """Generate a single counterfactual preserving invariants."""
        # Choose modification type based on what's allowed
        allowed_mods = []

        if not self._preserve_dims:
            allowed_mods.extend(["shift", "pad", "crop"])
        if not self._preserve_palette:
            allowed_mods.extend(["color_swap", "add_color"])

        # These always preserve both dims and palette subset
        allowed_mods.extend(["pixel_flip", "pixel_move", "region_swap"])

        if not allowed_mods:
            return None

        mod_type = self.rng.choice(allowed_mods)

        if mod_type == "pixel_flip":
            return self._pixel_flip_invariant(grid)
        elif mod_type == "pixel_move":
            return self._pixel_move_invariant(grid)
        elif mod_type == "region_swap":
            return self._region_swap_invariant(grid)
        elif mod_type == "color_swap":
            return self._color_swap_invariant(grid)
        elif mod_type == "shift":
            return self._shift_invariant(grid)
        elif mod_type == "pad":
            return self._pad_invariant(grid)
        elif mod_type == "crop":
            return self._crop_invariant(grid)
        elif mod_type == "add_color":
            return self._add_color_invariant(grid)

        return None

    def _pixel_flip_invariant(self, grid: Grid) -> Counterfactual:
        """Flip a pixel to another color in the existing palette."""
        modified = grid.copy()
        palette = list(grid.palette)

        if len(palette) < 2:
            # Just flip to 0 or 1
            palette = [0, 1]

        r = self.rng.integers(0, grid.height)
        c = self.rng.integers(0, grid.width)
        old_val = int(modified[r, c])

        # Choose new color from palette (different from current)
        other_colors = [p for p in palette if p != old_val]
        if other_colors:
            new_val = int(self.rng.choice(other_colors))
        else:
            new_val = (old_val + 1) % 10

        modified[r, c] = new_val

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="pixel_flip_invariant",
            modification_params={
                "row": int(r),
                "col": int(c),
                "old": old_val,
                "new": new_val,
            },
            expected_behavior="same_transformation",
        )

    def _pixel_move_invariant(self, grid: Grid) -> Optional[Counterfactual]:
        """Move a non-zero pixel to a zero location."""
        modified = grid.copy()

        # Find non-zero and zero positions
        nonzero_pos = list(zip(*np.where(grid.data != 0)))
        zero_pos = list(zip(*np.where(grid.data == 0)))

        if not nonzero_pos or not zero_pos:
            return self._pixel_flip_invariant(grid)

        # Pick random source and target
        src = nonzero_pos[self.rng.integers(0, len(nonzero_pos))]
        dst = zero_pos[self.rng.integers(0, len(zero_pos))]

        color = int(modified[src[0], src[1]])
        modified[src[0], src[1]] = 0
        modified[dst[0], dst[1]] = color

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="pixel_move_invariant",
            modification_params={
                "src": (int(src[0]), int(src[1])),
                "dst": (int(dst[0]), int(dst[1])),
                "color": color,
            },
            expected_behavior="same_transformation",
        )

    def _region_swap_invariant(self, grid: Grid) -> Counterfactual:
        """Swap two small regions (preserves dims and palette)."""
        modified = grid.copy()

        # Pick two random 2x2 regions
        if grid.height < 2 or grid.width < 2:
            return self._pixel_flip_invariant(grid)

        r1 = self.rng.integers(0, grid.height - 1)
        c1 = self.rng.integers(0, grid.width - 1)
        r2 = self.rng.integers(0, grid.height - 1)
        c2 = self.rng.integers(0, grid.width - 1)

        # Swap the regions
        region1 = modified.data[r1:r1+2, c1:c1+2].copy()
        region2 = modified.data[r2:r2+2, c2:c2+2].copy()
        modified.data[r1:r1+2, c1:c1+2] = region2
        modified.data[r2:r2+2, c2:c2+2] = region1

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="region_swap_invariant",
            modification_params={
                "region1": (int(r1), int(c1)),
                "region2": (int(r2), int(c2)),
            },
            expected_behavior="same_structure",
        )

    def _color_swap_invariant(self, grid: Grid) -> Optional[Counterfactual]:
        """Swap two colors in the palette."""
        palette = list(grid.palette - {0})
        if len(palette) < 2:
            return None

        c1, c2 = self.rng.choice(palette, size=2, replace=False)

        modified = grid.copy()
        mask1 = modified.data == c1
        mask2 = modified.data == c2
        modified.data[mask1] = c2
        modified.data[mask2] = c1

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="color_swap_invariant",
            modification_params={"color1": int(c1), "color2": int(c2)},
            expected_behavior="same_structure",
        )

    def _shift_invariant(self, grid: Grid) -> Counterfactual:
        """Shift content by 1 pixel (changes dims by wrapping)."""
        dr = self.rng.integers(-1, 2)
        dc = self.rng.integers(-1, 2)

        modified = Grid.zeros(grid.height, grid.width)
        for r in range(grid.height):
            for c in range(grid.width):
                nr = (r + dr) % grid.height
                nc = (c + dc) % grid.width
                modified[nr, nc] = grid[r, c]

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="shift_wrap",
            modification_params={"dr": int(dr), "dc": int(dc)},
            expected_behavior="same_transformation",
        )

    def _pad_invariant(self, grid: Grid) -> Optional[Counterfactual]:
        """Add padding (changes dims)."""
        padding = self.rng.integers(1, 3)
        new_h = grid.height + 2 * padding
        new_w = grid.width + 2 * padding

        if new_h > 30 or new_w > 30:
            return None

        modified = Grid.zeros(new_h, new_w)
        for r in range(grid.height):
            for c in range(grid.width):
                modified[r + padding, c + padding] = grid[r, c]

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="pad",
            modification_params={"padding": int(padding)},
            expected_behavior="may_differ",
        )

    def _crop_invariant(self, grid: Grid) -> Optional[Counterfactual]:
        """Crop to smaller size (changes dims)."""
        if grid.height <= 2 or grid.width <= 2:
            return None

        crop = 1
        new_h = grid.height - 2 * crop
        new_w = grid.width - 2 * crop

        modified = Grid(grid.data[crop:crop+new_h, crop:crop+new_w].copy())

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="crop",
            modification_params={"crop": crop},
            expected_behavior="may_differ",
        )

    def _add_color_invariant(self, grid: Grid) -> Counterfactual:
        """Add a new color not in the original palette."""
        modified = grid.copy()
        palette = grid.palette

        # Find a color not in the palette
        new_color = 1
        for c in range(1, 10):
            if c not in palette:
                new_color = c
                break

        # Replace one random pixel with the new color
        r = self.rng.integers(0, grid.height)
        c = self.rng.integers(0, grid.width)
        old_val = int(modified[r, c])
        modified[r, c] = new_color

        return Counterfactual(
            original=grid,
            modified=modified,
            modification_type="add_color",
            modification_params={
                "row": int(r),
                "col": int(c),
                "old": old_val,
                "new": new_color,
            },
            expected_behavior="may_differ",
        )


def generate_counterfactuals(
    state: Any,
    invariants: Optional[List[Any]] = None,
    k: int = 5,
    grid: Optional[Grid] = None,
) -> List[Counterfactual]:
    """
    Generate k counterfactuals while respecting invariants.

    This is the main entry point for invariant-preserving counterfactual generation.

    Args:
        state: WorldModelState with task features (used for context)
        invariants: List of ProposedInvariant or InvariantSpec to preserve
        k: Number of counterfactuals to generate
        grid: Grid to generate counterfactuals for (if None, uses features from state)

    Returns:
        List of k counterfactuals that preserve the specified invariants
    """
    # Convert ProposedInvariant to InvariantSpec if needed
    inv_specs: List[InvariantSpec] = []
    if invariants:
        for inv in invariants:
            if hasattr(inv, "name") and hasattr(inv, "is_hard"):
                # ProposedInvariant
                inv_specs.append(InvariantSpec(
                    name=inv.name,
                    preserve=inv.is_hard,
                    value=getattr(inv, "value", None),
                ))
            elif isinstance(inv, InvariantSpec):
                inv_specs.append(inv)

    # Create generator
    generator = InvariantPreservingGenerator(invariants=inv_specs)

    # If no grid provided, we can't generate counterfactuals
    if grid is None:
        return []

    return generator.generate(grid, num_counterfactuals=k)
