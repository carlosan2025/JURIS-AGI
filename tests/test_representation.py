"""Tests for representation module (objects, relations, features)."""

import pytest
import numpy as np

from juris_agi.core.types import Grid, BoundingBox
from juris_agi.representation.objects import (
    extract_connected_objects,
    extract_objects_by_color,
    find_bounding_box,
    extract_single_object,
)
from juris_agi.representation.features import (
    compute_grid_features,
    compute_object_features,
    compute_comparative_features,
    shapes_match,
)


class TestObjectExtraction:
    """Test object extraction from grids."""

    def test_extract_single_object(self):
        """Extract single connected component."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        objects = extract_connected_objects(grid, background_color=0)

        assert len(objects) == 1
        assert objects[0].pixel_count == 1
        assert objects[0].primary_color == 1

    def test_extract_multiple_objects(self):
        """Extract multiple disconnected objects."""
        grid = Grid.from_list([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        objects = extract_connected_objects(grid, background_color=0)

        assert len(objects) == 4

    def test_extract_connected_component(self):
        """L-shaped object should be single component."""
        grid = Grid.from_list([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ])
        objects = extract_connected_objects(grid, background_color=0)

        assert len(objects) == 1
        assert objects[0].pixel_count == 5

    def test_extract_8_connectivity(self):
        """Test 8-connectivity (diagonal connection)."""
        grid = Grid.from_list([
            [1, 0],
            [0, 1],
        ])

        # 4-connectivity: 2 objects
        objects_4 = extract_connected_objects(grid, connectivity=4)
        assert len(objects_4) == 2

        # 8-connectivity: 1 object
        objects_8 = extract_connected_objects(grid, connectivity=8)
        assert len(objects_8) == 1

    def test_extract_objects_by_color(self):
        """Extract objects filtered by color."""
        grid = Grid.from_list([
            [1, 0, 2],
            [1, 0, 2],
            [0, 0, 0],
        ])

        red_objects = extract_objects_by_color(grid, target_color=1)
        assert len(red_objects) == 1
        assert 1 in red_objects[0].colors

        blue_objects = extract_objects_by_color(grid, target_color=2)
        assert len(blue_objects) == 1

    def test_object_to_grid(self):
        """Convert object back to grid."""
        original = Grid.from_list([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        objects = extract_connected_objects(original)
        assert len(objects) == 1

        obj_grid = objects[0].to_grid()
        assert obj_grid.shape == (3, 3)

    def test_empty_grid(self):
        """Extract objects from empty grid."""
        grid = Grid.zeros(3, 3)
        objects = extract_connected_objects(grid)
        assert len(objects) == 0

    def test_find_bounding_box(self):
        """Test bounding box computation."""
        pixels = {(0, 0), (0, 2), (2, 1)}
        bbox = find_bounding_box(pixels)

        assert bbox.min_row == 0
        assert bbox.max_row == 2
        assert bbox.min_col == 0
        assert bbox.max_col == 2
        assert bbox.height == 3
        assert bbox.width == 3


class TestGridFeatures:
    """Test grid feature computation."""

    def test_basic_features(self):
        """Test basic grid features."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        features = compute_grid_features(grid)

        assert features["height"] == 2
        assert features["width"] == 3
        assert features["area"] == 6
        assert features["is_square"] == False
        assert features["num_colors"] == 6

    def test_symmetry_features(self):
        """Test symmetry detection."""
        # Horizontally symmetric
        h_sym = Grid.from_list([
            [1, 2, 1],
            [3, 4, 3],
        ])
        h_features = compute_grid_features(h_sym)
        assert h_features["h_symmetric"] == True
        assert h_features["v_symmetric"] == False

        # Vertically symmetric
        v_sym = Grid.from_list([
            [1, 2],
            [3, 4],
            [3, 4],
            [1, 2],
        ])
        v_features = compute_grid_features(v_sym)
        assert v_features["v_symmetric"] == True

    def test_fill_density(self):
        """Test fill density calculation."""
        sparse = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        features = compute_grid_features(sparse)
        assert abs(features["fill_density"] - 1/9) < 0.01

        dense = Grid.from_list([
            [1, 1],
            [1, 1],
        ])
        features = compute_grid_features(dense)
        assert features["fill_density"] == 1.0


class TestObjectFeatures:
    """Test object feature computation."""

    def test_rectangular_object(self):
        """Test features of rectangular object."""
        grid = Grid.from_list([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        objects = extract_connected_objects(grid)
        features = compute_object_features(objects[0])

        assert features["bbox_height"] == 2
        assert features["bbox_width"] == 2
        assert features["pixel_count"] == 4
        assert features["fill_ratio"] == 1.0
        assert features["is_rectangular"] == True
        assert features["is_square"] == True

    def test_non_rectangular_object(self):
        """Test features of L-shaped object."""
        grid = Grid.from_list([
            [1, 0],
            [1, 0],
            [1, 1],
        ])
        objects = extract_connected_objects(grid)
        features = compute_object_features(objects[0])

        assert features["pixel_count"] == 4
        assert features["fill_ratio"] < 1.0
        assert features["is_rectangular"] == False


class TestComparativeFeatures:
    """Test comparative features between grids."""

    def test_same_dimensions(self):
        """Test same dimension detection."""
        inp = Grid.from_list([[1, 2], [3, 4]])
        out = Grid.from_list([[5, 6], [7, 8]])

        features = compute_comparative_features(inp, out)
        assert features["same_dimensions"] == True
        assert features["height_ratio"] == 1.0
        assert features["width_ratio"] == 1.0

    def test_different_dimensions(self):
        """Test different dimension detection."""
        inp = Grid.from_list([[1, 2]])
        out = Grid.from_list([[1, 2], [3, 4]])

        features = compute_comparative_features(inp, out)
        assert features["same_dimensions"] == False
        assert features["height_ratio"] == 2.0

    def test_palette_changes(self):
        """Test palette change detection."""
        inp = Grid.from_list([[1, 2, 3]])
        out = Grid.from_list([[1, 2, 4]])

        features = compute_comparative_features(inp, out)
        assert features["palette_preserved"] == False
        assert 4 in features["colors_added"]
        assert 3 in features["colors_removed"]


class TestShapeMatching:
    """Test shape matching utilities."""

    def test_identical_shapes(self):
        """Identical shapes should match."""
        grid1 = Grid.from_list([
            [1, 0],
            [1, 1],
        ])
        grid2 = Grid.from_list([
            [2, 0],
            [2, 2],
        ])

        obj1 = extract_connected_objects(grid1)[0]
        obj2 = extract_connected_objects(grid2)[0]

        assert shapes_match(obj1, obj2)

    def test_different_shapes(self):
        """Different shapes should not match."""
        grid1 = Grid.from_list([
            [1, 1],
            [0, 0],
        ])
        grid2 = Grid.from_list([
            [1, 0],
            [1, 0],
        ])

        obj1 = extract_connected_objects(grid1)[0]
        obj2 = extract_connected_objects(grid2)[0]

        assert not shapes_match(obj1, obj2)
