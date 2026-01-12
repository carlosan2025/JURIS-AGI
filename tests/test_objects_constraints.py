"""Tests for enhanced object extraction and constraint helpers."""

import pytest
import numpy as np

from juris_agi.core.types import Grid, BoundingBox, ARCTask, ARCPair
from juris_agi.core.metrics import (
    DimensionConstraint,
    PaletteConstraint,
    ObjectCountConstraint,
    ConstraintSet,
    extract_constraints_from_task,
    fast_dimension_check,
    fast_palette_check,
)
from juris_agi.representation.objects import (
    extract_connected_objects,
    extract_enhanced_objects,
    compute_object_statistics,
    find_largest_object,
    find_objects_by_color,
    find_objects_by_size,
    objects_overlap,
    compute_object_relations,
    EnhancedObject,
)
from juris_agi.representation.features import (
    compute_task_features,
    compute_pair_features,
    TaskFeatures,
)


class TestEnhancedObjects:
    """Tests for enhanced object extraction."""

    def test_enhanced_object_attributes(self):
        """EnhancedObject should have all required attributes."""
        grid = Grid.from_list([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)

        assert len(objects) == 1
        obj = objects[0]

        # Check all attributes exist
        assert hasattr(obj, 'object_id')
        assert hasattr(obj, 'bbox')
        assert hasattr(obj, 'mask')
        assert hasattr(obj, 'colors')
        assert hasattr(obj, 'primary_color')
        assert hasattr(obj, 'color_histogram')
        assert hasattr(obj, 'pixel_count')
        assert hasattr(obj, 'area')
        assert hasattr(obj, 'centroid')
        assert hasattr(obj, 'fill_ratio')
        assert hasattr(obj, 'is_rectangular')
        assert hasattr(obj, 'is_monochrome')
        assert hasattr(obj, 'perimeter')
        assert hasattr(obj, 'compactness')

    def test_enhanced_object_values(self):
        """EnhancedObject should have correct values."""
        grid = Grid.from_list([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)
        obj = objects[0]

        # Check values
        assert obj.pixel_count == 4
        assert obj.area == 4
        assert obj.bbox.height == 2
        assert obj.bbox.width == 2
        assert obj.fill_ratio == 1.0
        assert obj.is_rectangular is True
        assert obj.is_monochrome is True
        assert obj.primary_color == 1
        assert obj.colors == {1}
        assert obj.color_histogram == {1: 4}

    def test_enhanced_object_mask(self):
        """Mask should correctly represent object shape."""
        grid = Grid.from_list([
            [1, 0],
            [1, 1],
        ])
        objects = extract_enhanced_objects(grid)
        obj = objects[0]

        expected_mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        assert np.array_equal(obj.mask, expected_mask)

    def test_enhanced_object_centroid(self):
        """Centroid should be correctly computed."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)
        obj = objects[0]

        # Single pixel at (1, 1)
        assert obj.centroid == (1.0, 1.0)

    def test_enhanced_object_perimeter(self):
        """Perimeter should count edge pixels."""
        # Single pixel - all edges exposed
        grid1 = Grid.from_list([[0, 1, 0]])
        obj1 = extract_enhanced_objects(grid1)[0]
        assert obj1.perimeter == 1  # Single pixel is all perimeter

        # 2x2 square - all pixels are on edge
        grid2 = Grid.from_list([
            [1, 1],
            [1, 1],
        ])
        obj2 = extract_enhanced_objects(grid2)[0]
        assert obj2.perimeter == 4  # All 4 pixels touch background

    def test_enhanced_object_non_rectangular(self):
        """Non-rectangular objects should have fill_ratio < 1."""
        grid = Grid.from_list([
            [1, 0],
            [1, 1],
        ])
        objects = extract_enhanced_objects(grid)
        obj = objects[0]

        assert obj.pixel_count == 3
        assert obj.bbox.height == 2
        assert obj.bbox.width == 2
        assert obj.fill_ratio == 0.75  # 3/4
        assert obj.is_rectangular is False

    def test_enhanced_object_multicolor(self):
        """Multicolor objects should have is_monochrome=False."""
        grid = Grid.from_list([
            [1, 2],
            [2, 1],
        ])
        # With 8-connectivity, this is one object
        objects = extract_enhanced_objects(grid, connectivity=8)
        obj = objects[0]

        assert obj.is_monochrome is False
        assert obj.colors == {1, 2}
        assert len(obj.color_histogram) == 2

    def test_enhanced_object_to_grid(self):
        """to_grid should recreate the object correctly."""
        grid = Grid.from_list([
            [0, 1, 1],
            [0, 1, 0],
        ])
        objects = extract_enhanced_objects(grid)
        obj = objects[0]

        obj_grid = obj.to_grid()
        # Should be bbox-sized (2x2)
        assert obj_grid.shape == (2, 2)
        assert obj_grid[0, 0] == 1
        assert obj_grid[0, 1] == 1
        assert obj_grid[1, 0] == 1
        assert obj_grid[1, 1] == 0


class TestObjectStatistics:
    """Tests for object statistics computation."""

    def test_compute_statistics_empty(self):
        """Statistics for empty list should have defaults."""
        stats = compute_object_statistics([])

        assert stats["count"] == 0
        assert stats["total_area"] == 0
        assert stats["all_monochrome"] is True

    def test_compute_statistics_multiple(self):
        """Statistics should aggregate correctly."""
        grid = Grid.from_list([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ])
        objects = extract_enhanced_objects(grid)
        stats = compute_object_statistics(objects)

        assert stats["count"] == 4
        assert stats["total_area"] == 4
        assert stats["avg_area"] == 1.0
        assert stats["all_monochrome"] is True
        assert stats["all_rectangular"] is True

    def test_find_largest_object(self):
        """Should find the largest object by area."""
        grid = Grid.from_list([
            [1, 1, 0, 2],
            [1, 1, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)
        largest = find_largest_object(objects)

        assert largest is not None
        assert largest.area == 4
        assert largest.primary_color == 1

    def test_find_objects_by_color(self):
        """Should filter objects by color."""
        grid = Grid.from_list([
            [1, 0, 2],
            [1, 0, 2],
        ])
        objects = extract_enhanced_objects(grid)

        red_objs = find_objects_by_color(objects, 1)
        assert len(red_objs) == 1
        assert 1 in red_objs[0].colors

        blue_objs = find_objects_by_color(objects, 2)
        assert len(blue_objs) == 1

        green_objs = find_objects_by_color(objects, 3)
        assert len(green_objs) == 0

    def test_find_objects_by_size(self):
        """Should filter objects by area range."""
        grid = Grid.from_list([
            [1, 1, 0, 2],
            [1, 1, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)

        large = find_objects_by_size(objects, min_area=3)
        assert len(large) == 1
        assert large[0].area == 4

        small = find_objects_by_size(objects, max_area=2)
        assert len(small) == 1
        assert small[0].area == 1


class TestObjectRelations:
    """Tests for object relations computation."""

    def test_objects_overlap(self):
        """Should detect overlapping bboxes."""
        grid = Grid.from_list([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
        ])
        objects = extract_enhanced_objects(grid)

        # Should have 2 separate objects (separated by 0 column)
        assert len(objects) == 2
        # Adjacent objects don't overlap (there's a gap)
        assert not objects_overlap(objects[0], objects[1])

    def test_compute_relations(self):
        """Should compute relations between objects."""
        grid = Grid.from_list([
            [1, 0, 0, 2],
            [0, 0, 0, 0],
        ])
        objects = extract_enhanced_objects(grid)
        relations = compute_object_relations(objects)

        assert len(relations) == 1  # One pair
        assert relations[0]["direction"] == "right"
        assert not relations[0]["overlapping"]
        assert not relations[0]["same_color"]


class TestDimensionConstraint:
    """Tests for dimension constraints."""

    def test_fixed_dims_satisfied(self):
        """Should pass when dimensions match."""
        constraint = DimensionConstraint(fixed_dims=(3, 3))
        grid = Grid.zeros(3, 3)

        assert constraint.check(grid) is True
        assert constraint.violation_severity(grid) == 0.0

    def test_fixed_dims_violated(self):
        """Should fail when dimensions don't match."""
        constraint = DimensionConstraint(fixed_dims=(3, 3))
        grid = Grid.zeros(2, 2)

        assert constraint.check(grid) is False
        assert constraint.violation_severity(grid) > 0.0

    def test_ratio_constraint(self):
        """Should check dimension ratio."""
        constraint = DimensionConstraint(h_ratio=2.0, w_ratio=2.0)
        input_grid = Grid.zeros(2, 3)
        output_ok = Grid.zeros(4, 6)
        output_bad = Grid.zeros(3, 4)

        assert constraint.check(output_ok, input_grid) is True
        assert constraint.check(output_bad, input_grid) is False

    def test_bounds_constraint(self):
        """Should check size bounds."""
        constraint = DimensionConstraint(max_height=50, max_width=50)

        small = Grid.zeros(10, 10)
        large = Grid.zeros(100, 100)

        assert constraint.check(small) is True
        assert constraint.check(large) is False


class TestPaletteConstraint:
    """Tests for palette constraints."""

    def test_allowed_colors_satisfied(self):
        """Should pass when palette is subset of allowed."""
        constraint = PaletteConstraint(allowed_colors={0, 1, 2, 3})
        grid = Grid.from_list([[1, 2], [0, 1]])

        assert constraint.check(grid) is True

    def test_allowed_colors_violated(self):
        """Should fail when extra colors present."""
        constraint = PaletteConstraint(allowed_colors={0, 1})
        grid = Grid.from_list([[1, 2], [0, 1]])  # Has color 2

        assert constraint.check(grid) is False

    def test_required_colors(self):
        """Should check required colors."""
        constraint = PaletteConstraint(required_colors={1, 2})
        grid_ok = Grid.from_list([[1, 2]])
        grid_bad = Grid.from_list([[1, 1]])

        assert constraint.check(grid_ok) is True
        assert constraint.check(grid_bad) is False

    def test_forbidden_colors(self):
        """Should check forbidden colors."""
        constraint = PaletteConstraint(forbidden_colors={5, 6, 7})
        grid_ok = Grid.from_list([[1, 2]])
        grid_bad = Grid.from_list([[1, 5]])

        assert constraint.check(grid_ok) is True
        assert constraint.check(grid_bad) is False


class TestObjectCountConstraint:
    """Tests for object count constraints."""

    def test_expected_count(self):
        """Should check expected object count."""
        constraint = ObjectCountConstraint(expected_count=2)

        grid_2 = Grid.from_list([[1, 0, 2]])
        grid_1 = Grid.from_list([[1, 1, 1]])

        ok, conf = constraint.check(grid_2)
        assert ok is True

        bad, conf = constraint.check(grid_1)
        assert bad is False

    def test_count_delta(self):
        """Should check object count delta from input."""
        constraint = ObjectCountConstraint(count_delta=1)  # Expect +1 object

        input_grid = Grid.from_list([[1]])  # 1 object
        output_2 = Grid.from_list([[1, 0, 2]])  # 2 objects
        output_1 = Grid.from_list([[1, 1]])  # 1 object

        ok, _ = constraint.check(output_2, input_grid)
        assert ok is True

        bad, _ = constraint.check(output_1, input_grid)
        assert bad is False


class TestConstraintSet:
    """Tests for combined constraint sets."""

    def test_check_all(self):
        """Should check all constraints together."""
        constraint_set = ConstraintSet(
            dimension=DimensionConstraint(fixed_dims=(2, 2)),
            palette=PaletteConstraint(allowed_colors={0, 1, 2}),
        )

        good_grid = Grid.from_list([[1, 2], [0, 1]])
        bad_dims = Grid.from_list([[1, 2, 3]])
        bad_palette = Grid.from_list([[1, 5], [0, 1]])

        ok, details = constraint_set.check_all(good_grid)
        assert ok is True
        assert details["dimension_ok"] is True
        assert details["palette_ok"] is True

        ok, details = constraint_set.check_all(bad_dims)
        assert ok is False
        assert details["dimension_ok"] is False

        ok, details = constraint_set.check_all(bad_palette)
        assert ok is False
        assert details["palette_ok"] is False

    def test_pruning_score(self):
        """Pruning score should reflect constraint satisfaction."""
        constraint_set = ConstraintSet(
            dimension=DimensionConstraint(fixed_dims=(2, 2)),
            palette=PaletteConstraint(allowed_colors={0, 1}),
        )

        good = Grid.from_list([[1, 0], [0, 1]])
        bad = Grid.from_list([[1, 2, 3]])

        good_score = constraint_set.compute_pruning_score(good)
        bad_score = constraint_set.compute_pruning_score(bad)

        assert good_score > bad_score
        assert good_score == 1.0  # Fully satisfied


class TestTaskFeatures:
    """Tests for task-level feature extraction."""

    def test_compute_task_features(self):
        """Should extract features from task."""
        task = ARCTask(
            task_id="test",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2]]),
                    output=Grid.from_list([[1, 2]]),
                ),
                ARCPair(
                    input=Grid.from_list([[3, 4, 5]]),
                    output=Grid.from_list([[3, 4, 5]]),
                ),
            ],
            test=[],
        )

        features = compute_task_features(task)

        assert isinstance(features, TaskFeatures)
        assert len(features.input_dims) == 2
        assert features.same_dims_all is True  # Input == output for each pair
        assert features.palette_preserved is True

    def test_fixed_output_dims_detection(self):
        """Should detect fixed output dimensions."""
        task = ARCTask(
            task_id="test",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2, 3]]),
                    output=Grid.from_list([[1]]),
                ),
                ARCPair(
                    input=Grid.from_list([[4, 5]]),
                    output=Grid.from_list([[2]]),
                ),
            ],
            test=[],
        )

        features = compute_task_features(task)

        assert features.fixed_output_dims == (1, 1)

    def test_scaling_factor_detection(self):
        """Should detect scaling factor."""
        task = ARCTask(
            task_id="test",
            train=[
                ARCPair(
                    input=Grid.from_list([[1]]),
                    output=Grid.from_list([[1, 1], [1, 1]]),
                ),
                ARCPair(
                    input=Grid.from_list([[2, 3]]),
                    output=Grid.from_list([[2, 3, 2, 3], [2, 3, 2, 3]]),
                ),
            ],
            test=[],
        )

        features = compute_task_features(task)

        assert features.scaling_factor == 2
        assert features.is_tiling_task is True


class TestExtractConstraintsFromTask:
    """Tests for constraint extraction from tasks."""

    def test_extract_constraints(self):
        """Should extract reasonable constraints from task."""
        task = ARCTask(
            task_id="test",
            train=[
                ARCPair(
                    input=Grid.from_list([[1, 2], [3, 4]]),
                    output=Grid.from_list([[1, 2], [3, 4]]),
                ),
            ],
            test=[],
        )

        constraints = extract_constraints_from_task(task)

        assert constraints.dimension is not None
        assert constraints.palette is not None
        # Object count may or may not be set


class TestFastChecks:
    """Tests for fast constraint checks."""

    def test_fast_dimension_check(self):
        """Fast dimension check should work correctly."""
        grid = Grid.zeros(3, 4)

        assert fast_dimension_check(grid, (3, 4)) is True
        assert fast_dimension_check(grid, (4, 3)) is False

    def test_fast_dimension_check_tolerance(self):
        """Fast dimension check with tolerance."""
        grid = Grid.zeros(10, 10)

        # Within 10% tolerance of expected dims
        # For expected (11, 11), tolerance is 0.1 * 11 = 1.1, so |10-11| = 1 <= 1.1 OK
        assert fast_dimension_check(grid, (11, 11), tolerance=0.1) is True

        # For expected (9, 9), tolerance is 0.1 * 9 = 0.9, so |10-9| = 1 > 0.9 NOT OK
        # Use higher tolerance
        assert fast_dimension_check(grid, (9, 9), tolerance=0.15) is True

        # Outside tolerance - expected 15, tolerance = 1.5, diff = 5 > 1.5
        assert fast_dimension_check(grid, (15, 15), tolerance=0.1) is False

    def test_fast_palette_check(self):
        """Fast palette check should work correctly."""
        grid = Grid.from_list([[1, 2], [0, 1]])

        assert fast_palette_check(grid, {0, 1, 2, 3}) is True
        assert fast_palette_check(grid, {0, 1}) is False  # Missing 2
