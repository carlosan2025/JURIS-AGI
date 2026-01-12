"""Tests for DSL primitives."""

import pytest
import numpy as np

from juris_agi.core.types import Grid, BoundingBox
from juris_agi.dsl.primitives import (
    PRIMITIVES,
    get_primitive,
    list_primitives,
    prim_identity,
    prim_rotate90,
    prim_reflect_h,
    prim_reflect_v,
    prim_transpose,
    prim_crop_to_content,
    prim_paste,
    prim_scale,
    prim_tile_h,
    prim_tile_v,
    prim_tile_repeat,
    prim_tile_to_size,
    prim_flood_fill,
    prim_fill_mask,
    prim_recolor,
    prim_overlay,
    prim_mask_color,
    prim_apply_mask,
    prim_draw_line_h,
    prim_draw_line_v,
    prim_draw_rect,
    prim_draw_rect_outline,
    prim_find_color_positions,
    prim_count_color,
    prim_has_color,
    prim_unique_colors,
)


class TestPrimitiveRegistry:
    """Test primitive registration and lookup."""

    def test_primitives_registered(self):
        """Verify key primitives are registered."""
        prims = list_primitives()
        assert "identity" in prims
        assert "rotate90" in prims
        assert "reflect_h" in prims
        assert "flood_fill" in prims
        assert "tile_repeat" in prims
        assert "fill_mask" in prims

    def test_get_primitive(self):
        """Test getting primitive by name."""
        spec = get_primitive("identity")
        assert spec is not None
        assert spec.name == "identity"
        assert spec.cost == 0  # identity has zero cost

    def test_get_unknown_primitive(self):
        """Unknown primitive returns None."""
        spec = get_primitive("nonexistent_primitive")
        assert spec is None


class TestIdentityPrimitive:
    """Test identity primitive."""

    def test_identity_returns_copy(self):
        """Identity returns a copy of input."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        result = prim_identity(grid)
        assert result == grid
        # Verify it's a copy, not the same object
        result[0, 0] = 9
        assert grid[0, 0] == 1


class TestRotatePrimitive:
    """Test rotation primitive."""

    def test_rotate90_once(self):
        """Rotate 90 degrees clockwise."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_rotate90(grid, 1)
        expected = Grid.from_list([
            [3, 1],
            [4, 2],
        ])
        assert result == expected

    def test_rotate90_twice(self):
        """Rotate 180 degrees."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_rotate90(grid, 2)
        expected = Grid.from_list([
            [4, 3],
            [2, 1],
        ])
        assert result == expected

    def test_rotate90_four_times(self):
        """Rotate 360 degrees returns original."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_rotate90(grid, 4)
        assert result == grid

    def test_rotate90_non_square(self):
        """Rotate non-square grid."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_rotate90(grid, 1)
        assert result.shape == (3, 2)


class TestReflectPrimitives:
    """Test reflection primitives."""

    def test_reflect_h(self):
        """Horizontal reflection (left-right flip)."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_reflect_h(grid)
        expected = Grid.from_list([
            [3, 2, 1],
            [6, 5, 4],
        ])
        assert result == expected

    def test_reflect_v(self):
        """Vertical reflection (top-bottom flip)."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_reflect_v(grid)
        expected = Grid.from_list([
            [4, 5, 6],
            [1, 2, 3],
        ])
        assert result == expected

    def test_reflect_h_twice(self):
        """Double horizontal reflection returns original."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        result = prim_reflect_h(prim_reflect_h(grid))
        assert result == grid


class TestTransposePrimitive:
    """Test transpose primitive."""

    def test_transpose_square(self):
        """Transpose square grid."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_transpose(grid)
        expected = Grid.from_list([
            [1, 3],
            [2, 4],
        ])
        assert result == expected

    def test_transpose_rectangular(self):
        """Transpose rectangular grid."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_transpose(grid)
        assert result.shape == (3, 2)


class TestCropPrimitive:
    """Test crop to content primitive."""

    def test_crop_to_content(self):
        """Crop removes background borders."""
        grid = Grid.from_list([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ])
        result = prim_crop_to_content(grid)
        expected = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        assert result == expected

    def test_crop_no_background(self):
        """Crop with no background returns same."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_crop_to_content(grid)
        assert result == grid

    def test_crop_empty_grid(self):
        """Crop empty grid returns minimal grid."""
        grid = Grid.zeros(3, 3)
        result = prim_crop_to_content(grid)
        assert result.shape == (1, 1)


class TestPastePrimitive:
    """Test paste primitive."""

    def test_paste_basic(self):
        """Paste source onto target."""
        target = Grid.zeros(4, 4)
        source = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_paste(target, source, 1, 1)
        expected = Grid.from_list([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ])
        assert result == expected

    def test_paste_with_transparency(self):
        """Background (0) in source doesn't overwrite target."""
        target = Grid.full(3, 3, 5)
        source = Grid.from_list([
            [1, 0],
            [0, 2],
        ])
        result = prim_paste(target, source, 0, 0)
        expected = Grid.from_list([
            [1, 5, 5],
            [5, 2, 5],
            [5, 5, 5],
        ])
        assert result == expected

    def test_paste_clipping(self):
        """Paste clips to target boundaries."""
        target = Grid.zeros(3, 3)
        source = Grid.full(2, 2, 1)
        result = prim_paste(target, source, 2, 2)
        # Only one pixel should be pasted
        assert int(result[2, 2]) == 1
        assert int(result[1, 1]) == 0


class TestTilingPrimitives:
    """Test tiling primitives."""

    def test_tile_h(self):
        """Tile horizontally."""
        grid = Grid.from_list([[1, 2]])
        result = prim_tile_h(grid, 3)
        expected = Grid.from_list([[1, 2, 1, 2, 1, 2]])
        assert result == expected

    def test_tile_v(self):
        """Tile vertically."""
        grid = Grid.from_list([[1], [2]])
        result = prim_tile_v(grid, 3)
        expected = Grid.from_list([[1], [2], [1], [2], [1], [2]])
        assert result == expected

    def test_tile_repeat_2d(self):
        """Tile in both dimensions."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_tile_repeat(grid, 2, 3)
        assert result.shape == (4, 6)
        # Check pattern repeats
        assert int(result[0, 0]) == 1
        assert int(result[0, 2]) == 1
        assert int(result[0, 4]) == 1
        assert int(result[2, 0]) == 1

    def test_tile_to_size(self):
        """Tile to fill target size."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        result = prim_tile_to_size(grid, 5, 7)
        assert result.shape == (5, 7)
        # Check corners
        assert int(result[0, 0]) == 1
        assert int(result[4, 6]) == 1  # Wrapped pattern


class TestScalePrimitive:
    """Test scale primitive."""

    def test_scale_2x(self):
        """Scale by factor 2."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_scale(grid, 2)
        expected = Grid.from_list([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ])
        assert result == expected

    def test_scale_3x(self):
        """Scale by factor 3."""
        grid = Grid.from_list([[1, 2]])
        result = prim_scale(grid, 3)
        assert result.shape == (3, 6)


class TestFloodFillPrimitive:
    """Test flood fill primitive."""

    def test_flood_fill_basic(self):
        """Basic flood fill from center."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = prim_flood_fill(grid, 1, 1, 5)
        # All connected background should be filled
        assert int(result[1, 1]) == 5
        assert int(result[0, 0]) == 5
        assert int(result[2, 2]) == 5

    def test_flood_fill_bounded(self):
        """Flood fill respects boundaries."""
        grid = Grid.from_list([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        result = prim_flood_fill(grid, 1, 1, 5)
        # Only center should be filled
        assert int(result[1, 1]) == 5
        assert int(result[0, 0]) == 1
        assert int(result[0, 1]) == 1

    def test_flood_fill_connected_region(self):
        """Flood fill fills entire connected region."""
        grid = Grid.from_list([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        result = prim_flood_fill(grid, 0, 0, 2)
        # Left region should be filled
        assert int(result[0, 0]) == 2
        assert int(result[1, 1]) == 2
        # Right region should remain
        assert int(result[0, 3]) == 0
        assert int(result[3, 3]) == 0

    def test_flood_fill_same_color(self):
        """Flood fill with same color returns unchanged."""
        grid = Grid.full(3, 3, 5)
        result = prim_flood_fill(grid, 1, 1, 5)
        assert result == grid

    def test_flood_fill_out_of_bounds(self):
        """Flood fill at invalid position returns unchanged."""
        grid = Grid.zeros(3, 3)
        result = prim_flood_fill(grid, 10, 10, 5)
        assert result == grid


class TestFillMaskPrimitive:
    """Test fill_mask primitive."""

    def test_fill_mask_basic(self):
        """Fill where mask is non-zero."""
        grid = Grid.zeros(3, 3)
        mask = Grid.from_list([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ])
        result = prim_fill_mask(grid, mask, 5)
        expected = Grid.from_list([
            [5, 0, 5],
            [0, 5, 0],
            [5, 0, 5],
        ])
        assert result == expected

    def test_fill_mask_preserves_existing(self):
        """Fill mask replaces, doesn't blend."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        mask = Grid.from_list([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        result = prim_fill_mask(grid, mask, 0)
        expected = Grid.from_list([
            [0, 2, 3],
            [4, 0, 6],
            [7, 8, 0],
        ])
        assert result == expected

    def test_fill_mask_dimension_mismatch(self):
        """Fill mask raises on dimension mismatch."""
        grid = Grid.zeros(3, 3)
        mask = Grid.zeros(2, 2)
        with pytest.raises(ValueError):
            prim_fill_mask(grid, mask, 5)


class TestRecolorPrimitive:
    """Test recolor primitive."""

    def test_recolor_basic(self):
        """Replace one color with another."""
        grid = Grid.from_list([
            [1, 2, 1],
            [2, 1, 2],
        ])
        result = prim_recolor(grid, 1, 5)
        expected = Grid.from_list([
            [5, 2, 5],
            [2, 5, 2],
        ])
        assert result == expected

    def test_recolor_no_match(self):
        """Recolor with no matching pixels."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        result = prim_recolor(grid, 9, 5)
        assert result == grid


class TestOverlayPrimitive:
    """Test overlay primitive."""

    def test_overlay_basic(self):
        """Overlay source onto target."""
        target = Grid.full(3, 3, 1)
        source = Grid.from_list([
            [0, 2, 0],
            [2, 2, 2],
            [0, 2, 0],
        ])
        result = prim_overlay(target, source)
        expected = Grid.from_list([
            [1, 2, 1],
            [2, 2, 2],
            [1, 2, 1],
        ])
        assert result == expected

    def test_overlay_dimension_mismatch(self):
        """Overlay raises on dimension mismatch."""
        target = Grid.zeros(3, 3)
        source = Grid.zeros(2, 2)
        with pytest.raises(ValueError):
            prim_overlay(target, source)


class TestMaskPrimitives:
    """Test mask-related primitives."""

    def test_mask_color(self):
        """Create mask for specific color."""
        grid = Grid.from_list([
            [1, 2, 1],
            [2, 1, 2],
        ])
        result = prim_mask_color(grid, 1)
        expected = Grid.from_list([
            [1, 0, 1],
            [0, 1, 0],
        ])
        assert result == expected

    def test_apply_mask(self):
        """Apply mask to grid."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        mask = Grid.from_list([
            [1, 0, 1],
            [0, 1, 0],
        ])
        result = prim_apply_mask(grid, mask, 0)
        expected = Grid.from_list([
            [1, 0, 3],
            [0, 5, 0],
        ])
        assert result == expected


class TestDrawingPrimitives:
    """Test drawing primitives."""

    def test_draw_line_h(self):
        """Draw horizontal line."""
        grid = Grid.zeros(3, 5)
        result = prim_draw_line_h(grid, 1, 1, 3, 5)
        expected = Grid.from_list([
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ])
        assert result == expected

    def test_draw_line_v(self):
        """Draw vertical line."""
        grid = Grid.zeros(5, 3)
        result = prim_draw_line_v(grid, 1, 1, 3, 5)
        expected = Grid.from_list([
            [0, 0, 0],
            [0, 5, 0],
            [0, 5, 0],
            [0, 5, 0],
            [0, 0, 0],
        ])
        assert result == expected

    def test_draw_rect_filled(self):
        """Draw filled rectangle."""
        grid = Grid.zeros(5, 5)
        result = prim_draw_rect(grid, 1, 1, 3, 3, 5)
        expected = Grid.from_list([
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ])
        assert result == expected

    def test_draw_rect_outline(self):
        """Draw rectangle outline."""
        grid = Grid.zeros(5, 5)
        result = prim_draw_rect_outline(grid, 1, 1, 3, 3, 5)
        expected = Grid.from_list([
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ])
        assert result == expected


class TestAnalysisPrimitives:
    """Test analysis primitives."""

    def test_find_color_positions(self):
        """Find all positions of a color."""
        grid = Grid.from_list([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ])
        positions = prim_find_color_positions(grid, 1)
        assert len(positions) == 4
        assert (0, 0) in positions
        assert (0, 2) in positions
        assert (2, 0) in positions
        assert (2, 2) in positions

    def test_count_color(self):
        """Count occurrences of a color."""
        grid = Grid.from_list([
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ])
        assert prim_count_color(grid, 1) == 5
        assert prim_count_color(grid, 2) == 4
        assert prim_count_color(grid, 3) == 0

    def test_has_color(self):
        """Check if color exists."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        assert prim_has_color(grid, 1) == True
        assert prim_has_color(grid, 5) == False

    def test_unique_colors(self):
        """Count unique colors."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        assert prim_unique_colors(grid) == 6
