"""
DSL Primitives for grid manipulation.

Each primitive has:
- A typed signature
- A Python implementation
- Documentation
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional, Tuple
import numpy as np

from ..core.types import Grid, GridObject, BoundingBox, Color
from ..representation.objects import extract_connected_objects, find_bounding_box
from .type_system import (
    DSLType,
    GridType,
    IntType,
    ColorType,
    BoolType,
    ListType,
    ObjectType,
    PointType,
    BBoxType,
    ColorMapType,
    FunctionType,
    GRID,
    INT,
    COLOR,
    BOOL,
    OBJECT,
    POINT,
    BBOX,
    COLOR_MAP,
    list_of,
    function_type,
)


@dataclass
class PrimitiveSpec:
    """Specification of a DSL primitive."""
    name: str
    signature: FunctionType
    implementation: Callable
    doc: str
    cost: int = 1  # For MDL scoring


# Registry of all primitives
PRIMITIVES: Dict[str, PrimitiveSpec] = {}


def register_primitive(
    name: str,
    signature: FunctionType,
    doc: str = "",
    cost: int = 1,
):
    """Decorator to register a primitive."""
    def decorator(fn: Callable) -> Callable:
        PRIMITIVES[name] = PrimitiveSpec(
            name=name,
            signature=signature,
            implementation=fn,
            doc=doc,
            cost=cost,
        )
        return fn
    return decorator


def get_primitive(name: str) -> Optional[PrimitiveSpec]:
    """Get a primitive by name."""
    return PRIMITIVES.get(name)


def list_primitives() -> List[str]:
    """List all registered primitives."""
    return list(PRIMITIVES.keys())


# ============================================================================
# Core Grid Primitives
# ============================================================================

@register_primitive(
    "identity",
    function_type(GRID, ret=GRID),
    "Returns the input grid unchanged.",
    cost=0,
)
def prim_identity(grid: Grid) -> Grid:
    """Identity function - returns input unchanged."""
    return grid.copy()


@register_primitive(
    "crop_to_bbox",
    function_type(GRID, BBOX, ret=GRID),
    "Crop grid to the given bounding box.",
)
def prim_crop_to_bbox(grid: Grid, bbox: BoundingBox) -> Grid:
    """Crop grid to bounding box."""
    cropped = grid.data[
        bbox.min_row:bbox.max_row + 1,
        bbox.min_col:bbox.max_col + 1
    ]
    return Grid(cropped.copy())


@register_primitive(
    "crop_to_content",
    function_type(GRID, ret=GRID),
    "Crop grid to the bounding box of non-background content.",
)
def prim_crop_to_content(grid: Grid, background: int = 0) -> Grid:
    """Crop to content bounding box."""
    non_bg = np.argwhere(grid.data != background)
    if len(non_bg) == 0:
        return Grid.zeros(1, 1)

    min_row, min_col = non_bg.min(axis=0)
    max_row, max_col = non_bg.max(axis=0)

    return Grid(grid.data[min_row:max_row + 1, min_col:max_col + 1].copy())


@register_primitive(
    "extract_objects",
    function_type(GRID, ret=list_of(OBJECT)),
    "Extract connected components as objects.",
)
def prim_extract_objects(grid: Grid, background: int = 0) -> List[GridObject]:
    """Extract connected objects from grid."""
    return extract_connected_objects(grid, background_color=background)


@register_primitive(
    "get_object",
    function_type(list_of(OBJECT), INT, ret=OBJECT),
    "Get object at index from list.",
)
def prim_get_object(objects: List[GridObject], index: int) -> GridObject:
    """Get object by index."""
    return objects[index]


@register_primitive(
    "object_to_grid",
    function_type(OBJECT, ret=GRID),
    "Convert object to grid (bbox-sized).",
)
def prim_object_to_grid(obj: GridObject) -> Grid:
    """Convert object to grid."""
    return obj.to_grid()


@register_primitive(
    "num_objects",
    function_type(list_of(OBJECT), ret=INT),
    "Count number of objects.",
)
def prim_num_objects(objects: List[GridObject]) -> int:
    """Count objects."""
    return len(objects)


# ============================================================================
# Color Primitives
# ============================================================================

@register_primitive(
    "recolor",
    function_type(GRID, COLOR, COLOR, ret=GRID),
    "Replace all pixels of one color with another.",
)
def prim_recolor(grid: Grid, from_color: Color, to_color: Color) -> Grid:
    """Replace one color with another."""
    result = grid.copy()
    result.data[result.data == from_color] = to_color
    return result


@register_primitive(
    "recolor_map",
    function_type(GRID, COLOR_MAP, ret=GRID),
    "Apply a color mapping to the grid.",
)
def prim_recolor_map(grid: Grid, color_map: Dict[int, int]) -> Grid:
    """Apply color mapping."""
    result = grid.copy()
    for from_c, to_c in color_map.items():
        result.data[grid.data == from_c] = to_c
    return result


@register_primitive(
    "fill",
    function_type(GRID, COLOR, ret=GRID),
    "Fill entire grid with a single color.",
)
def prim_fill(grid: Grid, color: Color) -> Grid:
    """Fill grid with color."""
    return Grid.full(grid.height, grid.width, color)


@register_primitive(
    "fill_background",
    function_type(GRID, COLOR, ret=GRID),
    "Fill background (color 0) with specified color.",
)
def prim_fill_background(grid: Grid, color: Color) -> Grid:
    """Fill background with color."""
    result = grid.copy()
    result.data[result.data == 0] = color
    return result


@register_primitive(
    "get_palette",
    function_type(GRID, ret=list_of(COLOR)),
    "Get the set of colors used in the grid.",
)
def prim_get_palette(grid: Grid) -> List[Color]:
    """Get palette as sorted list."""
    return sorted(list(grid.palette))


@register_primitive(
    "dominant_color",
    function_type(GRID, ret=COLOR),
    "Get the most common color in the grid.",
)
def prim_dominant_color(grid: Grid) -> Color:
    """Get most common color."""
    unique, counts = np.unique(grid.data, return_counts=True)
    return int(unique[np.argmax(counts)])


# ============================================================================
# Geometric Transformations
# ============================================================================

@register_primitive(
    "rotate90",
    function_type(GRID, INT, ret=GRID),
    "Rotate grid by 90 degrees clockwise, n times.",
)
def prim_rotate90(grid: Grid, n: int = 1) -> Grid:
    """Rotate 90 degrees clockwise n times."""
    n = n % 4
    if n == 0:
        return grid.copy()
    rotated = np.rot90(grid.data, k=-n)  # Negative for clockwise
    return Grid(rotated)


@register_primitive(
    "reflect_h",
    function_type(GRID, ret=GRID),
    "Reflect grid horizontally (left-right flip).",
)
def prim_reflect_h(grid: Grid) -> Grid:
    """Flip left-right."""
    return Grid(np.fliplr(grid.data))


@register_primitive(
    "reflect_v",
    function_type(GRID, ret=GRID),
    "Reflect grid vertically (top-bottom flip).",
)
def prim_reflect_v(grid: Grid) -> Grid:
    """Flip top-bottom."""
    return Grid(np.flipud(grid.data))


@register_primitive(
    "transpose",
    function_type(GRID, ret=GRID),
    "Transpose grid (swap rows and columns).",
)
def prim_transpose(grid: Grid) -> Grid:
    """Transpose grid."""
    return Grid(grid.data.T)


@register_primitive(
    "translate",
    function_type(GRID, INT, INT, ret=GRID),
    "Translate grid content by (dr, dc), filling empty with background.",
)
def prim_translate(grid: Grid, dr: int, dc: int, background: int = 0) -> Grid:
    """Translate grid content."""
    result = Grid.full(grid.height, grid.width, background)

    for r in range(grid.height):
        for c in range(grid.width):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                result[nr, nc] = grid[r, c]

    return result


# ============================================================================
# Grid Construction and Composition
# ============================================================================

@register_primitive(
    "paste",
    function_type(GRID, GRID, INT, INT, ret=GRID),
    "Paste source grid onto target at position (row, col).",
)
def prim_paste(target: Grid, source: Grid, row: int, col: int) -> Grid:
    """Paste source onto target at position."""
    result = target.copy()

    for r in range(source.height):
        for c in range(source.width):
            tr, tc = row + r, col + c
            if 0 <= tr < result.height and 0 <= tc < result.width:
                if source[r, c] != 0:  # Don't paste background
                    result[tr, tc] = source[r, c]

    return result


@register_primitive(
    "overlay",
    function_type(GRID, GRID, ret=GRID),
    "Overlay source onto target (source non-background overwrites).",
)
def prim_overlay(target: Grid, source: Grid) -> Grid:
    """Overlay source onto target."""
    if target.shape != source.shape:
        raise ValueError("Grids must have same dimensions for overlay")

    result = target.copy()
    mask = source.data != 0
    result.data[mask] = source.data[mask]
    return result


@register_primitive(
    "tile_h",
    function_type(GRID, INT, ret=GRID),
    "Tile grid horizontally n times.",
)
def prim_tile_h(grid: Grid, n: int) -> Grid:
    """Tile horizontally."""
    return Grid(np.tile(grid.data, (1, n)))


@register_primitive(
    "tile_v",
    function_type(GRID, INT, ret=GRID),
    "Tile grid vertically n times.",
)
def prim_tile_v(grid: Grid, n: int) -> Grid:
    """Tile vertically."""
    return Grid(np.tile(grid.data, (n, 1)))


@register_primitive(
    "concat_h",
    function_type(GRID, GRID, ret=GRID),
    "Concatenate two grids horizontally.",
)
def prim_concat_h(grid1: Grid, grid2: Grid) -> Grid:
    """Concatenate horizontally."""
    if grid1.height != grid2.height:
        raise ValueError("Grids must have same height for horizontal concat")
    return Grid(np.hstack([grid1.data, grid2.data]))


@register_primitive(
    "concat_v",
    function_type(GRID, GRID, ret=GRID),
    "Concatenate two grids vertically.",
)
def prim_concat_v(grid1: Grid, grid2: Grid) -> Grid:
    """Concatenate vertically."""
    if grid1.width != grid2.width:
        raise ValueError("Grids must have same width for vertical concat")
    return Grid(np.vstack([grid1.data, grid2.data]))


@register_primitive(
    "resize",
    function_type(GRID, INT, INT, ret=GRID),
    "Resize grid to new dimensions (top-left aligned).",
)
def prim_resize(grid: Grid, new_height: int, new_width: int, background: int = 0) -> Grid:
    """Resize grid."""
    result = Grid.full(new_height, new_width, background)

    copy_h = min(grid.height, new_height)
    copy_w = min(grid.width, new_width)

    result.data[:copy_h, :copy_w] = grid.data[:copy_h, :copy_w]
    return result


@register_primitive(
    "scale",
    function_type(GRID, INT, ret=GRID),
    "Scale grid by integer factor (each pixel becomes factor x factor block).",
)
def prim_scale(grid: Grid, factor: int) -> Grid:
    """Scale grid by factor."""
    return Grid(np.kron(grid.data, np.ones((factor, factor), dtype=np.int32)))


# ============================================================================
# Dimension Queries
# ============================================================================

@register_primitive(
    "height",
    function_type(GRID, ret=INT),
    "Get grid height.",
)
def prim_height(grid: Grid) -> int:
    """Get height."""
    return grid.height


@register_primitive(
    "width",
    function_type(GRID, ret=INT),
    "Get grid width.",
)
def prim_width(grid: Grid) -> int:
    """Get width."""
    return grid.width


@register_primitive(
    "get_pixel",
    function_type(GRID, INT, INT, ret=COLOR),
    "Get pixel color at (row, col).",
)
def prim_get_pixel(grid: Grid, row: int, col: int) -> Color:
    """Get pixel."""
    return int(grid[row, col])


@register_primitive(
    "set_pixel",
    function_type(GRID, INT, INT, COLOR, ret=GRID),
    "Set pixel at (row, col) to color.",
)
def prim_set_pixel(grid: Grid, row: int, col: int, color: Color) -> Grid:
    """Set pixel."""
    result = grid.copy()
    if 0 <= row < result.height and 0 <= col < result.width:
        result[row, col] = color
    return result


# ============================================================================
# Logical / Selection Primitives
# ============================================================================

@register_primitive(
    "mask_color",
    function_type(GRID, COLOR, ret=GRID),
    "Create binary mask where color matches (1) or not (0).",
)
def prim_mask_color(grid: Grid, color: Color) -> Grid:
    """Create mask for color."""
    mask = (grid.data == color).astype(np.int32)
    return Grid(mask)


@register_primitive(
    "invert_mask",
    function_type(GRID, ret=GRID),
    "Invert a binary mask (swap 0 and 1).",
)
def prim_invert_mask(grid: Grid) -> Grid:
    """Invert binary mask."""
    result = grid.copy()
    result.data = 1 - result.data
    return result


@register_primitive(
    "apply_mask",
    function_type(GRID, GRID, COLOR, ret=GRID),
    "Apply mask to grid: where mask is 1, use original; where 0, use fill color.",
)
def prim_apply_mask(grid: Grid, mask: Grid, fill_color: Color) -> Grid:
    """Apply mask to grid."""
    if grid.shape != mask.shape:
        raise ValueError("Grid and mask must have same dimensions")

    result = grid.copy()
    result.data[mask.data == 0] = fill_color
    return result


# ============================================================================
# Object-Level Primitives
# ============================================================================

@register_primitive(
    "largest_object",
    function_type(list_of(OBJECT), ret=OBJECT),
    "Get the largest object by pixel count.",
)
def prim_largest_object(objects: List[GridObject]) -> GridObject:
    """Get largest object."""
    if not objects:
        raise ValueError("No objects to select from")
    return max(objects, key=lambda o: o.pixel_count)


@register_primitive(
    "smallest_object",
    function_type(list_of(OBJECT), ret=OBJECT),
    "Get the smallest object by pixel count.",
)
def prim_smallest_object(objects: List[GridObject]) -> GridObject:
    """Get smallest object."""
    if not objects:
        raise ValueError("No objects to select from")
    return min(objects, key=lambda o: o.pixel_count)


@register_primitive(
    "filter_by_color",
    function_type(list_of(OBJECT), COLOR, ret=list_of(OBJECT)),
    "Filter objects to those with primary color.",
)
def prim_filter_by_color(objects: List[GridObject], color: Color) -> List[GridObject]:
    """Filter by primary color."""
    return [o for o in objects if o.primary_color == color]


@register_primitive(
    "object_bbox",
    function_type(OBJECT, ret=BBOX),
    "Get bounding box of object.",
)
def prim_object_bbox(obj: GridObject) -> BoundingBox:
    """Get object's bounding box."""
    return obj.bbox


@register_primitive(
    "sort_objects_by_size",
    function_type(list_of(OBJECT), ret=list_of(OBJECT)),
    "Sort objects by size (ascending).",
)
def prim_sort_objects_by_size(objects: List[GridObject]) -> List[GridObject]:
    """Sort by size."""
    return sorted(objects, key=lambda o: o.pixel_count)


@register_primitive(
    "sort_objects_by_position",
    function_type(list_of(OBJECT), ret=list_of(OBJECT)),
    "Sort objects by position (top-to-bottom, left-to-right).",
)
def prim_sort_objects_by_position(objects: List[GridObject]) -> List[GridObject]:
    """Sort by position."""
    return sorted(objects, key=lambda o: (o.bbox.min_row, o.bbox.min_col))


# ============================================================================
# Flood Fill and Region Operations
# ============================================================================

@register_primitive(
    "flood_fill",
    function_type(GRID, INT, INT, COLOR, ret=GRID),
    "Flood fill starting at (row, col) with the given color.",
)
def prim_flood_fill(grid: Grid, row: int, col: int, fill_color: Color) -> Grid:
    """
    Flood fill starting at (row, col).

    Uses BFS to fill all connected cells with the same color as the starting cell.
    """
    from collections import deque

    if not (0 <= row < grid.height and 0 <= col < grid.width):
        return grid.copy()

    result = grid.copy()
    target_color = int(grid[row, col])

    if target_color == fill_color:
        return result  # Already the fill color

    visited = set()
    queue = deque([(row, col)])
    visited.add((row, col))

    while queue:
        r, c = queue.popleft()
        result[r, c] = fill_color

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid.height and 0 <= nc < grid.width and
                (nr, nc) not in visited and int(grid[nr, nc]) == target_color):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return result


@register_primitive(
    "fill_mask",
    function_type(GRID, GRID, COLOR, ret=GRID),
    "Fill grid where mask is non-zero with the specified color.",
)
def prim_fill_mask(grid: Grid, mask: Grid, color: Color) -> Grid:
    """
    Fill positions where mask is non-zero with color.

    Unlike apply_mask (which keeps original where mask=1), this fills where mask=1.
    """
    if grid.shape != mask.shape:
        raise ValueError("Grid and mask must have same dimensions")

    result = grid.copy()
    result.data[mask.data != 0] = color
    return result


# ============================================================================
# Tiling Operations
# ============================================================================

@register_primitive(
    "tile_repeat",
    function_type(GRID, INT, INT, ret=GRID),
    "Tile grid by (rows, cols) repetitions.",
)
def prim_tile_repeat(grid: Grid, rows: int, cols: int) -> Grid:
    """Tile grid in both dimensions."""
    return Grid(np.tile(grid.data, (rows, cols)))


@register_primitive(
    "tile_to_size",
    function_type(GRID, INT, INT, ret=GRID),
    "Tile grid to fill at least the specified dimensions.",
)
def prim_tile_to_size(grid: Grid, target_height: int, target_width: int) -> Grid:
    """Tile grid to fill target dimensions, then crop."""
    rows_needed = (target_height + grid.height - 1) // grid.height
    cols_needed = (target_width + grid.width - 1) // grid.width

    tiled = np.tile(grid.data, (rows_needed, cols_needed))
    cropped = tiled[:target_height, :target_width]
    return Grid(cropped)


# ============================================================================
# Pattern Drawing Primitives
# ============================================================================

@register_primitive(
    "draw_line_h",
    function_type(GRID, INT, INT, INT, COLOR, ret=GRID),
    "Draw horizontal line from (row, start_col) to (row, end_col).",
)
def prim_draw_line_h(grid: Grid, row: int, start_col: int, end_col: int, color: Color) -> Grid:
    """Draw horizontal line."""
    result = grid.copy()
    if 0 <= row < grid.height:
        c1, c2 = min(start_col, end_col), max(start_col, end_col)
        c1 = max(0, c1)
        c2 = min(grid.width - 1, c2)
        result.data[row, c1:c2+1] = color
    return result


@register_primitive(
    "draw_line_v",
    function_type(GRID, INT, INT, INT, COLOR, ret=GRID),
    "Draw vertical line from (start_row, col) to (end_row, col).",
)
def prim_draw_line_v(grid: Grid, col: int, start_row: int, end_row: int, color: Color) -> Grid:
    """Draw vertical line."""
    result = grid.copy()
    if 0 <= col < grid.width:
        r1, r2 = min(start_row, end_row), max(start_row, end_row)
        r1 = max(0, r1)
        r2 = min(grid.height - 1, r2)
        result.data[r1:r2+1, col] = color
    return result


@register_primitive(
    "draw_rect",
    function_type(GRID, INT, INT, INT, INT, COLOR, ret=GRID),
    "Draw filled rectangle from (r1, c1) to (r2, c2).",
)
def prim_draw_rect(grid: Grid, r1: int, c1: int, r2: int, c2: int, color: Color) -> Grid:
    """Draw filled rectangle."""
    result = grid.copy()
    r1, r2 = max(0, min(r1, r2)), min(grid.height - 1, max(r1, r2))
    c1, c2 = max(0, min(c1, c2)), min(grid.width - 1, max(c1, c2))
    result.data[r1:r2+1, c1:c2+1] = color
    return result


@register_primitive(
    "draw_rect_outline",
    function_type(GRID, INT, INT, INT, INT, COLOR, ret=GRID),
    "Draw rectangle outline from (r1, c1) to (r2, c2).",
)
def prim_draw_rect_outline(grid: Grid, r1: int, c1: int, r2: int, c2: int, color: Color) -> Grid:
    """Draw rectangle outline."""
    result = grid.copy()
    r1, r2 = max(0, min(r1, r2)), min(grid.height - 1, max(r1, r2))
    c1, c2 = max(0, min(c1, c2)), min(grid.width - 1, max(c1, c2))

    # Top and bottom edges
    result.data[r1, c1:c2+1] = color
    result.data[r2, c1:c2+1] = color
    # Left and right edges
    result.data[r1:r2+1, c1] = color
    result.data[r1:r2+1, c2] = color

    return result


# ============================================================================
# Grid Analysis Primitives
# ============================================================================

@register_primitive(
    "find_color_positions",
    function_type(GRID, COLOR, ret=list_of(POINT)),
    "Find all positions of a specific color.",
)
def prim_find_color_positions(grid: Grid, color: Color) -> List[Tuple[int, int]]:
    """Find all positions with given color."""
    positions = np.argwhere(grid.data == color)
    return [(int(r), int(c)) for r, c in positions]


@register_primitive(
    "count_color",
    function_type(GRID, COLOR, ret=INT),
    "Count occurrences of a color.",
)
def prim_count_color(grid: Grid, color: Color) -> int:
    """Count color occurrences."""
    return int(np.sum(grid.data == color))


@register_primitive(
    "has_color",
    function_type(GRID, COLOR, ret=BOOL),
    "Check if grid contains a specific color.",
)
def prim_has_color(grid: Grid, color: Color) -> bool:
    """Check if color exists."""
    return color in grid.palette


@register_primitive(
    "unique_colors",
    function_type(GRID, ret=INT),
    "Count unique colors in grid.",
)
def prim_unique_colors(grid: Grid) -> int:
    """Count unique colors."""
    return len(grid.palette)
