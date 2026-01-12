"""
Feature computation for grids, objects, and tasks.

Provides comprehensive feature extraction for:
- Individual grids (dimensions, symmetry, density)
- Objects (shape, color, position)
- Tasks (invariants, patterns across examples)
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..core.types import Grid, GridObject, BoundingBox, ARCTask, ARCPair
from .objects import extract_enhanced_objects, compute_object_statistics, EnhancedObject


def compute_grid_features(grid: Grid) -> Dict[str, Any]:
    """
    Compute comprehensive features for a grid.

    Features include dimensions, colors, symmetry, patterns, etc.
    """
    features: Dict[str, Any] = {}

    # Basic dimensions
    features["height"] = grid.height
    features["width"] = grid.width
    features["area"] = grid.height * grid.width
    features["is_square"] = grid.height == grid.width

    # Color features
    features["palette"] = list(grid.palette)
    features["num_colors"] = len(grid.palette)
    features["has_background"] = 0 in grid.palette

    # Color distribution
    unique, counts = np.unique(grid.data, return_counts=True)
    color_dist = dict(zip(unique.tolist(), counts.tolist()))
    features["color_distribution"] = color_dist
    features["dominant_color"] = int(unique[np.argmax(counts)])

    # Symmetry features
    features["h_symmetric"] = _check_horizontal_symmetry(grid)
    features["v_symmetric"] = _check_vertical_symmetry(grid)
    features["d_symmetric"] = _check_diagonal_symmetry(grid) if grid.height == grid.width else False

    # Density features
    non_background = np.sum(grid.data != 0)
    features["fill_density"] = non_background / features["area"]

    # Edge features
    features["has_border"] = _check_has_border(grid)

    return features


def compute_object_features(obj: GridObject) -> Dict[str, Any]:
    """Compute features for a single object."""
    features: Dict[str, Any] = {}

    # Size features
    features["bbox_height"] = obj.bbox.height
    features["bbox_width"] = obj.bbox.width
    features["bbox_area"] = obj.bbox.area
    features["pixel_count"] = obj.pixel_count
    features["fill_ratio"] = obj.pixel_count / obj.bbox.area if obj.bbox.area > 0 else 0

    # Shape features
    features["is_rectangular"] = features["fill_ratio"] == 1.0
    features["is_square"] = (
        obj.bbox.height == obj.bbox.width and features["is_rectangular"]
    )

    # Color features
    features["colors"] = list(obj.colors)
    features["num_colors"] = len(obj.colors)
    features["primary_color"] = obj.primary_color
    features["is_monochrome"] = len(obj.colors) == 1

    # Position features
    features["center_row"] = (obj.bbox.min_row + obj.bbox.max_row) / 2
    features["center_col"] = (obj.bbox.min_col + obj.bbox.max_col) / 2

    # Compute object symmetry
    obj_grid = obj.to_grid()
    features["h_symmetric"] = _check_horizontal_symmetry(obj_grid)
    features["v_symmetric"] = _check_vertical_symmetry(obj_grid)

    return features


def compute_comparative_features(
    input_grid: Grid,
    output_grid: Grid,
) -> Dict[str, Any]:
    """Compute features comparing input and output grids."""
    features: Dict[str, Any] = {}

    # Dimension changes
    features["same_dimensions"] = input_grid.shape == output_grid.shape
    features["height_ratio"] = output_grid.height / input_grid.height if input_grid.height > 0 else 0
    features["width_ratio"] = output_grid.width / input_grid.width if input_grid.width > 0 else 0

    # Color changes
    features["palette_preserved"] = input_grid.palette == output_grid.palette
    features["colors_added"] = list(output_grid.palette - input_grid.palette)
    features["colors_removed"] = list(input_grid.palette - output_grid.palette)

    # If same dimensions, compute pixel changes
    if features["same_dimensions"]:
        diff_mask = input_grid.data != output_grid.data
        features["num_changed_pixels"] = int(np.sum(diff_mask))
        features["change_ratio"] = features["num_changed_pixels"] / input_grid.height / input_grid.width

    return features


def _check_horizontal_symmetry(grid: Grid) -> bool:
    """Check if grid is horizontally symmetric (left-right)."""
    return np.array_equal(grid.data, np.fliplr(grid.data))


def _check_vertical_symmetry(grid: Grid) -> bool:
    """Check if grid is vertically symmetric (top-bottom)."""
    return np.array_equal(grid.data, np.flipud(grid.data))


def _check_diagonal_symmetry(grid: Grid) -> bool:
    """Check if grid is diagonally symmetric (transpose)."""
    if grid.height != grid.width:
        return False
    return np.array_equal(grid.data, grid.data.T)


def _check_has_border(grid: Grid, border_color: int = 0) -> bool:
    """Check if grid has a uniform border."""
    # Check top and bottom rows
    if not (np.all(grid.data[0, :] == border_color) and
            np.all(grid.data[-1, :] == border_color)):
        return False
    # Check left and right columns
    if not (np.all(grid.data[:, 0] == border_color) and
            np.all(grid.data[:, -1] == border_color)):
        return False
    return True


def extract_shape_signature(obj: GridObject) -> Tuple[int, ...]:
    """
    Extract a rotation/translation-invariant shape signature.

    Returns a tuple that can be used for shape comparison.
    """
    # Normalize to top-left corner
    pixels = sorted(obj.pixels)
    if not pixels:
        return ()

    min_r = min(r for r, c, _ in pixels)
    min_c = min(c for r, c, _ in pixels)

    # Normalized positions (ignoring color for shape)
    normalized = tuple(sorted((r - min_r, c - min_c) for r, c, _ in pixels))
    return normalized


def shapes_match(obj1: GridObject, obj2: GridObject) -> bool:
    """Check if two objects have the same shape (ignoring color and position)."""
    return extract_shape_signature(obj1) == extract_shape_signature(obj2)


# ============================================================================
# Task-Level Feature Extraction
# ============================================================================

@dataclass
class TaskFeatures:
    """Comprehensive features extracted from an ARC task."""
    # Dimension features
    input_dims: List[Tuple[int, int]]
    output_dims: List[Tuple[int, int]]
    fixed_output_dims: Optional[Tuple[int, int]]  # None if variable
    dimension_ratio: Optional[Tuple[float, float]]  # (h_ratio, w_ratio) if consistent

    # Palette features
    input_palette: Set[int]
    output_palette: Set[int]
    palette_preserved: bool
    colors_added: Set[int]
    colors_removed: Set[int]
    palette_size_input: int
    palette_size_output: int

    # Symmetry features
    inputs_symmetric_h: List[bool]
    inputs_symmetric_v: List[bool]
    outputs_symmetric_h: List[bool]
    outputs_symmetric_v: List[bool]

    # Object features
    input_object_counts: List[int]
    output_object_counts: List[int]
    object_count_delta: Optional[int]  # None if inconsistent
    avg_input_objects: float
    avg_output_objects: float

    # Bbox statistics
    input_bbox_stats: Dict[str, Any]
    output_bbox_stats: Dict[str, Any]

    # Pattern indicators
    same_dims_all: bool
    scaling_factor: Optional[int]  # None if not uniform scaling
    is_cropping_task: bool
    is_tiling_task: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "fixed_output_dims": self.fixed_output_dims,
            "dimension_ratio": self.dimension_ratio,
            "input_palette": list(self.input_palette),
            "output_palette": list(self.output_palette),
            "palette_preserved": self.palette_preserved,
            "colors_added": list(self.colors_added),
            "colors_removed": list(self.colors_removed),
            "palette_size_input": self.palette_size_input,
            "palette_size_output": self.palette_size_output,
            "input_object_counts": self.input_object_counts,
            "output_object_counts": self.output_object_counts,
            "object_count_delta": self.object_count_delta,
            "same_dims_all": self.same_dims_all,
            "scaling_factor": self.scaling_factor,
            "is_cropping_task": self.is_cropping_task,
            "is_tiling_task": self.is_tiling_task,
        }


def compute_task_features(task: ARCTask) -> TaskFeatures:
    """
    Compute comprehensive features from an ARC task.

    Analyzes all training pairs to extract patterns and invariants.
    """
    if not task.train:
        # Return empty features for empty task
        return _empty_task_features()

    # Collect per-pair data
    input_dims = []
    output_dims = []
    input_palettes: List[Set[int]] = []
    output_palettes: List[Set[int]] = []
    inputs_h_sym = []
    inputs_v_sym = []
    outputs_h_sym = []
    outputs_v_sym = []
    input_obj_counts = []
    output_obj_counts = []
    input_bbox_sizes = []
    output_bbox_sizes = []

    for pair in task.train:
        # Dimensions
        input_dims.append(pair.input.shape)
        output_dims.append(pair.output.shape)

        # Palettes
        input_palettes.append(pair.input.palette)
        output_palettes.append(pair.output.palette)

        # Symmetry
        inputs_h_sym.append(_check_horizontal_symmetry(pair.input))
        inputs_v_sym.append(_check_vertical_symmetry(pair.input))
        outputs_h_sym.append(_check_horizontal_symmetry(pair.output))
        outputs_v_sym.append(_check_vertical_symmetry(pair.output))

        # Object counts
        input_objs = extract_enhanced_objects(pair.input)
        output_objs = extract_enhanced_objects(pair.output)
        input_obj_counts.append(len(input_objs))
        output_obj_counts.append(len(output_objs))

        # Bbox sizes
        for obj in input_objs:
            input_bbox_sizes.append((obj.bbox.height, obj.bbox.width))
        for obj in output_objs:
            output_bbox_sizes.append((obj.bbox.height, obj.bbox.width))

    # Aggregate palettes
    all_input_palette: Set[int] = set()
    all_output_palette: Set[int] = set()
    for p in input_palettes:
        all_input_palette.update(p)
    for p in output_palettes:
        all_output_palette.update(p)

    # Check fixed output dims
    fixed_output_dims = None
    if len(set(output_dims)) == 1:
        fixed_output_dims = output_dims[0]

    # Check dimension ratio consistency
    dimension_ratio = _compute_dimension_ratio(input_dims, output_dims)

    # Check object count delta consistency
    deltas = [o - i for i, o in zip(input_obj_counts, output_obj_counts)]
    object_count_delta = deltas[0] if len(set(deltas)) == 1 else None

    # Check scaling factor
    scaling_factor = _detect_scaling_factor(input_dims, output_dims)

    # Check if cropping task (output smaller than input)
    is_cropping = all(
        o[0] <= i[0] and o[1] <= i[1]
        for i, o in zip(input_dims, output_dims)
    ) and any(o != i for i, o in zip(input_dims, output_dims))

    # Check if tiling task (output larger by integer factor)
    is_tiling = scaling_factor is not None and scaling_factor > 1

    # Bbox statistics
    input_bbox_stats = _compute_bbox_stats(input_bbox_sizes)
    output_bbox_stats = _compute_bbox_stats(output_bbox_sizes)

    return TaskFeatures(
        input_dims=input_dims,
        output_dims=output_dims,
        fixed_output_dims=fixed_output_dims,
        dimension_ratio=dimension_ratio,
        input_palette=all_input_palette,
        output_palette=all_output_palette,
        palette_preserved=all_output_palette.issubset(all_input_palette | {0}),
        colors_added=all_output_palette - all_input_palette,
        colors_removed=all_input_palette - all_output_palette,
        palette_size_input=len(all_input_palette),
        palette_size_output=len(all_output_palette),
        inputs_symmetric_h=inputs_h_sym,
        inputs_symmetric_v=inputs_v_sym,
        outputs_symmetric_h=outputs_h_sym,
        outputs_symmetric_v=outputs_v_sym,
        input_object_counts=input_obj_counts,
        output_object_counts=output_obj_counts,
        object_count_delta=object_count_delta,
        avg_input_objects=sum(input_obj_counts) / len(input_obj_counts) if input_obj_counts else 0,
        avg_output_objects=sum(output_obj_counts) / len(output_obj_counts) if output_obj_counts else 0,
        input_bbox_stats=input_bbox_stats,
        output_bbox_stats=output_bbox_stats,
        same_dims_all=all(i == o for i, o in zip(input_dims, output_dims)),
        scaling_factor=scaling_factor,
        is_cropping_task=is_cropping,
        is_tiling_task=is_tiling,
    )


def _empty_task_features() -> TaskFeatures:
    """Return empty TaskFeatures for edge cases."""
    return TaskFeatures(
        input_dims=[],
        output_dims=[],
        fixed_output_dims=None,
        dimension_ratio=None,
        input_palette=set(),
        output_palette=set(),
        palette_preserved=True,
        colors_added=set(),
        colors_removed=set(),
        palette_size_input=0,
        palette_size_output=0,
        inputs_symmetric_h=[],
        inputs_symmetric_v=[],
        outputs_symmetric_h=[],
        outputs_symmetric_v=[],
        input_object_counts=[],
        output_object_counts=[],
        object_count_delta=None,
        avg_input_objects=0,
        avg_output_objects=0,
        input_bbox_stats={},
        output_bbox_stats={},
        same_dims_all=True,
        scaling_factor=None,
        is_cropping_task=False,
        is_tiling_task=False,
    )


def _compute_dimension_ratio(
    input_dims: List[Tuple[int, int]],
    output_dims: List[Tuple[int, int]],
) -> Optional[Tuple[float, float]]:
    """Compute consistent dimension ratio if it exists."""
    if not input_dims or not output_dims:
        return None

    ratios = []
    for (ih, iw), (oh, ow) in zip(input_dims, output_dims):
        if ih > 0 and iw > 0:
            ratios.append((oh / ih, ow / iw))
        else:
            return None

    if not ratios:
        return None

    # Check consistency
    first = ratios[0]
    if all(abs(r[0] - first[0]) < 0.01 and abs(r[1] - first[1]) < 0.01 for r in ratios):
        return first
    return None


def _detect_scaling_factor(
    input_dims: List[Tuple[int, int]],
    output_dims: List[Tuple[int, int]],
) -> Optional[int]:
    """Detect if output is uniformly scaled version of input."""
    if not input_dims or not output_dims:
        return None

    factors = []
    for (ih, iw), (oh, ow) in zip(input_dims, output_dims):
        if ih > 0 and iw > 0:
            h_factor = oh / ih
            w_factor = ow / iw
            # Check if integer scaling and same for both dimensions
            if (h_factor == w_factor and
                h_factor == int(h_factor) and
                h_factor >= 1):
                factors.append(int(h_factor))
            else:
                return None
        else:
            return None

    if factors and len(set(factors)) == 1:
        return factors[0]
    return None


def _compute_bbox_stats(sizes: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Compute statistics over bounding box sizes."""
    if not sizes:
        return {
            "count": 0,
            "avg_height": 0,
            "avg_width": 0,
            "min_height": 0,
            "max_height": 0,
            "min_width": 0,
            "max_width": 0,
            "all_same_size": True,
        }

    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]

    return {
        "count": len(sizes),
        "avg_height": sum(heights) / len(heights),
        "avg_width": sum(widths) / len(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "min_width": min(widths),
        "max_width": max(widths),
        "all_same_size": len(set(sizes)) == 1,
    }


def compute_pair_features(pair: ARCPair) -> Dict[str, Any]:
    """Compute features for a single input-output pair."""
    input_features = compute_grid_features(pair.input)
    output_features = compute_grid_features(pair.output)
    comparative = compute_comparative_features(pair.input, pair.output)

    # Object-level features
    input_objs = extract_enhanced_objects(pair.input)
    output_objs = extract_enhanced_objects(pair.output)
    input_obj_stats = compute_object_statistics(input_objs)
    output_obj_stats = compute_object_statistics(output_objs)

    return {
        "input": input_features,
        "output": output_features,
        "comparative": comparative,
        "input_objects": input_obj_stats,
        "output_objects": output_obj_stats,
    }
