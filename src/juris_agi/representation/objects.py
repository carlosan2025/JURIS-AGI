"""
Object extraction from grids using connected components.

Enhanced with per-object attributes: bbox, mask, color histogram, area, centroid.
"""

from typing import List, Set, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from ..core.types import Grid, GridObject, BoundingBox, Color


@dataclass
class EnhancedObject:
    """
    Enhanced object representation with detailed attributes.

    Provides richer information than basic GridObject for constraint checking.
    """
    object_id: int
    bbox: BoundingBox
    mask: np.ndarray  # Binary mask within bbox (height x width)
    colors: Set[int]  # Set of colors in the object
    primary_color: Optional[int]  # Most common color
    color_histogram: Dict[int, int]  # Color -> count mapping
    pixel_count: int  # Total non-background pixels
    area: int  # Same as pixel_count
    centroid: Tuple[float, float]  # (row, col) center of mass
    fill_ratio: float  # pixel_count / bbox.area
    is_rectangular: bool  # True if fill_ratio == 1.0
    is_monochrome: bool  # True if only one non-background color
    perimeter: int  # Number of edge pixels
    compactness: float  # 4*pi*area / perimeter^2 (1.0 for circle)

    # Original pixels for compatibility
    pixels: Set[Tuple[int, int, int]]  # (local_row, local_col, color)

    def to_grid(self) -> Grid:
        """Convert object back to a Grid (bbox-sized)."""
        grid = Grid.zeros(self.bbox.height, self.bbox.width)
        for r, c, color in self.pixels:
            grid[r, c] = color
        return grid

    def to_grid_object(self) -> GridObject:
        """Convert to basic GridObject for compatibility."""
        return GridObject(
            pixels=frozenset(self.pixels),
            bbox=self.bbox,
            object_id=self.object_id,
        )


def find_bounding_box(pixels: Set[Tuple[int, int]]) -> BoundingBox:
    """Find the bounding box of a set of pixel coordinates."""
    if not pixels:
        return BoundingBox(0, 0, 0, 0)

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]

    return BoundingBox(
        min_row=min(rows),
        min_col=min(cols),
        max_row=max(rows),
        max_col=max(cols),
    )


def extract_connected_objects(
    grid: Grid,
    background_color: int = 0,
    connectivity: int = 4,
) -> List[GridObject]:
    """
    Extract connected components from a grid as GridObjects.

    Args:
        grid: The input grid
        background_color: Color to treat as background (not part of objects)
        connectivity: 4 for von Neumann neighborhood, 8 for Moore neighborhood

    Returns:
        List of GridObject instances
    """
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    objects: List[GridObject] = []
    object_id = 0

    # Define neighbor offsets
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connectivity
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1),
        ]

    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r, start_c]:
                continue
            if grid[start_r, start_c] == background_color:
                visited[start_r, start_c] = True
                continue

            # BFS to find connected component
            component_pixels: Set[Tuple[int, int]] = set()
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True

            while queue:
                r, c = queue.popleft()
                component_pixels.add((r, c))

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if not visited[nr, nc] and grid[nr, nc] != background_color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if component_pixels:
                bbox = find_bounding_box(component_pixels)

                # Convert to local coordinates with colors
                local_pixels = frozenset(
                    (r - bbox.min_row, c - bbox.min_col, int(grid[r, c]))
                    for r, c in component_pixels
                )

                objects.append(GridObject(
                    pixels=local_pixels,
                    bbox=bbox,
                    object_id=object_id,
                ))
                object_id += 1

    return objects


def extract_objects_by_color(
    grid: Grid,
    target_color: Optional[Color] = None,
    background_color: int = 0,
) -> List[GridObject]:
    """
    Extract objects of a specific color (or all colors if not specified).

    Each color forms separate objects based on connectivity.
    """
    if target_color is not None:
        # Mask to only the target color
        mask = Grid(np.where(grid.data == target_color, target_color, 0))
        return extract_connected_objects(mask, background_color=0)

    # Extract objects for each non-background color
    all_objects: List[GridObject] = []
    unique_colors = set(np.unique(grid.data).tolist()) - {background_color}

    object_id = 0
    for color in unique_colors:
        mask = Grid(np.where(grid.data == color, color, 0))
        color_objects = extract_connected_objects(mask, background_color=0)
        for obj in color_objects:
            all_objects.append(GridObject(
                pixels=obj.pixels,
                bbox=obj.bbox,
                object_id=object_id,
            ))
            object_id += 1

    return all_objects


def extract_single_object(grid: Grid, background_color: int = 0) -> Optional[GridObject]:
    """
    Extract a single object from a grid (assumes only one connected component).

    Returns None if no non-background pixels found.
    """
    objects = extract_connected_objects(grid, background_color)
    if not objects:
        return None
    if len(objects) == 1:
        return objects[0]
    # If multiple, merge them into one
    all_pixels: Set[Tuple[int, int, Color]] = set()
    for obj in objects:
        for lr, lc, color in obj.pixels:
            gr = lr + obj.bbox.min_row
            gc = lc + obj.bbox.min_col
            all_pixels.add((gr, gc, color))

    global_coords = {(r, c) for r, c, _ in all_pixels}
    bbox = find_bounding_box(global_coords)

    local_pixels = frozenset(
        (r - bbox.min_row, c - bbox.min_col, color)
        for r, c, color in all_pixels
    )

    return GridObject(pixels=local_pixels, bbox=bbox, object_id=0)


def object_to_grid(obj: GridObject, background_color: int = 0) -> Grid:
    """Convert a GridObject back to a Grid."""
    grid = Grid.full(obj.bbox.height, obj.bbox.width, background_color)
    for r, c, color in obj.pixels:
        grid[r, c] = color
    return grid


def extract_enhanced_objects(
    grid: Grid,
    background_color: int = 0,
    connectivity: int = 4,
) -> List[EnhancedObject]:
    """
    Extract connected components as EnhancedObjects with rich attributes.

    Args:
        grid: The input grid
        background_color: Color to treat as background
        connectivity: 4 or 8 connectivity

    Returns:
        List of EnhancedObject with detailed attributes
    """
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    objects: List[EnhancedObject] = []
    object_id = 0

    # Define neighbor offsets
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1),
        ]

    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r, start_c]:
                continue
            if grid[start_r, start_c] == background_color:
                visited[start_r, start_c] = True
                continue

            # BFS to find connected component
            component_pixels: Set[Tuple[int, int]] = set()
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True

            while queue:
                r, c = queue.popleft()
                component_pixels.add((r, c))

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if not visited[nr, nc] and grid[nr, nc] != background_color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if component_pixels:
                enhanced_obj = _build_enhanced_object(
                    grid, component_pixels, object_id, background_color
                )
                objects.append(enhanced_obj)
                object_id += 1

    return objects


def _build_enhanced_object(
    grid: Grid,
    global_pixels: Set[Tuple[int, int]],
    object_id: int,
    background_color: int = 0,
) -> EnhancedObject:
    """Build an EnhancedObject from a set of global pixel coordinates."""
    # Compute bounding box
    bbox = find_bounding_box(global_pixels)

    # Build mask and local pixels
    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    local_pixels: Set[Tuple[int, int, int]] = set()
    color_histogram: Dict[int, int] = {}

    sum_r, sum_c = 0.0, 0.0
    perimeter = 0

    for gr, gc in global_pixels:
        lr = gr - bbox.min_row
        lc = gc - bbox.min_col
        color = int(grid[gr, gc])

        mask[lr, lc] = 1
        local_pixels.add((lr, lc, color))

        # Color histogram
        color_histogram[color] = color_histogram.get(color, 0) + 1

        # Centroid accumulation
        sum_r += gr
        sum_c += gc

        # Perimeter: count pixels with at least one background neighbor
        has_bg_neighbor = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < grid.height and 0 <= nc < grid.width):
                has_bg_neighbor = True
            elif grid[nr, nc] == background_color:
                has_bg_neighbor = True
        if has_bg_neighbor:
            perimeter += 1

    pixel_count = len(global_pixels)
    colors = set(color_histogram.keys())

    # Primary color
    primary_color = max(color_histogram, key=color_histogram.get) if color_histogram else None

    # Centroid
    centroid = (sum_r / pixel_count, sum_c / pixel_count) if pixel_count > 0 else (0.0, 0.0)

    # Fill ratio and rectangularity
    fill_ratio = pixel_count / bbox.area if bbox.area > 0 else 0.0
    is_rectangular = abs(fill_ratio - 1.0) < 1e-6

    # Compactness (4*pi*area / perimeter^2)
    if perimeter > 0:
        compactness = (4 * 3.14159 * pixel_count) / (perimeter * perimeter)
    else:
        compactness = 0.0

    return EnhancedObject(
        object_id=object_id,
        bbox=bbox,
        mask=mask,
        colors=colors,
        primary_color=primary_color,
        color_histogram=color_histogram,
        pixel_count=pixel_count,
        area=pixel_count,
        centroid=centroid,
        fill_ratio=fill_ratio,
        is_rectangular=is_rectangular,
        is_monochrome=len(colors) == 1,
        perimeter=perimeter,
        compactness=compactness,
        pixels=local_pixels,
    )


def compute_object_statistics(objects: List[EnhancedObject]) -> Dict[str, Any]:
    """
    Compute aggregate statistics over a list of objects.

    Useful for task-level feature extraction.
    """
    if not objects:
        return {
            "count": 0,
            "total_area": 0,
            "avg_area": 0.0,
            "min_area": 0,
            "max_area": 0,
            "avg_fill_ratio": 0.0,
            "colors_used": set(),
            "all_monochrome": True,
            "all_rectangular": True,
            "bbox_sizes": [],
        }

    areas = [obj.area for obj in objects]
    fill_ratios = [obj.fill_ratio for obj in objects]
    all_colors: Set[int] = set()
    for obj in objects:
        all_colors.update(obj.colors)

    return {
        "count": len(objects),
        "total_area": sum(areas),
        "avg_area": sum(areas) / len(areas),
        "min_area": min(areas),
        "max_area": max(areas),
        "avg_fill_ratio": sum(fill_ratios) / len(fill_ratios),
        "colors_used": all_colors,
        "all_monochrome": all(obj.is_monochrome for obj in objects),
        "all_rectangular": all(obj.is_rectangular for obj in objects),
        "bbox_sizes": [(obj.bbox.height, obj.bbox.width) for obj in objects],
    }


def find_largest_object(objects: List[EnhancedObject]) -> Optional[EnhancedObject]:
    """Find the object with the largest area."""
    if not objects:
        return None
    return max(objects, key=lambda obj: obj.area)


def find_objects_by_color(
    objects: List[EnhancedObject],
    color: int,
) -> List[EnhancedObject]:
    """Filter objects that contain a specific color."""
    return [obj for obj in objects if color in obj.colors]


def find_objects_by_size(
    objects: List[EnhancedObject],
    min_area: int = 0,
    max_area: Optional[int] = None,
) -> List[EnhancedObject]:
    """Filter objects by area range."""
    result = [obj for obj in objects if obj.area >= min_area]
    if max_area is not None:
        result = [obj for obj in result if obj.area <= max_area]
    return result


def objects_overlap(obj1: EnhancedObject, obj2: EnhancedObject) -> bool:
    """Check if two objects' bounding boxes overlap."""
    # Check if bboxes don't overlap (no overlap if any of these is true)
    if obj1.bbox.max_row < obj2.bbox.min_row:
        return False
    if obj2.bbox.max_row < obj1.bbox.min_row:
        return False
    if obj1.bbox.max_col < obj2.bbox.min_col:
        return False
    if obj2.bbox.max_col < obj1.bbox.min_col:
        return False
    return True


def compute_object_relations(
    objects: List[EnhancedObject],
) -> List[Dict[str, Any]]:
    """
    Compute pairwise relations between objects.

    Returns list of relations with spatial relationships.
    """
    relations = []

    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue

            # Compute relative position
            dr = obj2.centroid[0] - obj1.centroid[0]
            dc = obj2.centroid[1] - obj1.centroid[1]

            # Directional relationship
            if abs(dr) > abs(dc):
                direction = "below" if dr > 0 else "above"
            else:
                direction = "right" if dc > 0 else "left"

            relation = {
                "obj1_id": obj1.object_id,
                "obj2_id": obj2.object_id,
                "direction": direction,
                "distance": (dr * dr + dc * dc) ** 0.5,
                "overlapping": objects_overlap(obj1, obj2),
                "same_color": obj1.primary_color == obj2.primary_color,
                "same_size": abs(obj1.area - obj2.area) <= 1,
                "same_shape": obj1.mask.shape == obj2.mask.shape and np.array_equal(obj1.mask, obj2.mask),
            }
            relations.append(relation)

    return relations
