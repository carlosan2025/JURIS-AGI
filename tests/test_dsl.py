"""Tests for DSL primitives and interpreter."""

import pytest
import numpy as np

from juris_agi.core.types import Grid
from juris_agi.dsl.ast import (
    PrimitiveNode,
    ComposeNode,
    LiteralNode,
    LambdaNode,
    VariableNode,
)
from juris_agi.dsl.interpreter import DSLInterpreter, make_program, run_on_grid
from juris_agi.dsl.primitives import (
    prim_identity,
    prim_rotate90,
    prim_reflect_h,
    prim_reflect_v,
    prim_transpose,
    prim_crop_to_content,
    prim_scale,
    prim_recolor,
    prim_fill,
    prim_translate,
    prim_tile_h,
    prim_tile_v,
    PRIMITIVES,
)
from juris_agi.dsl.prettyprint import ast_to_source


class TestGridPrimitives:
    """Test grid manipulation primitives."""

    def test_identity(self):
        """Identity should return same grid."""
        grid = Grid.from_list([[1, 2], [3, 4]])
        result = prim_identity(grid)
        assert result == grid
        # Should be a copy, not same object
        result[0, 0] = 9
        assert grid[0, 0] == 1

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

    def test_rotate90_four_times_is_identity(self):
        """Four 90-degree rotations should return original."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_rotate90(grid, 4)
        assert result == grid

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
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        result = prim_reflect_v(grid)
        expected = Grid.from_list([
            [5, 6],
            [3, 4],
            [1, 2],
        ])
        assert result == expected

    def test_transpose(self):
        """Transpose grid."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        result = prim_transpose(grid)
        expected = Grid.from_list([
            [1, 4],
            [2, 5],
            [3, 6],
        ])
        assert result == expected

    def test_crop_to_content(self):
        """Crop to non-background content."""
        grid = Grid.from_list([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        result = prim_crop_to_content(grid)
        expected = Grid.from_list([
            [1, 1],
            [1, 1],
        ])
        assert result == expected

    def test_crop_single_pixel(self):
        """Crop to single non-background pixel."""
        grid = Grid.from_list([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ])
        result = prim_crop_to_content(grid)
        expected = Grid.from_list([[5]])
        assert result == expected

    def test_scale_2x(self):
        """Scale grid by 2x."""
        grid = Grid.from_list([
            [1, 2],
        ])
        result = prim_scale(grid, 2)
        expected = Grid.from_list([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ])
        assert result == expected

    def test_scale_3x(self):
        """Scale grid by 3x."""
        grid = Grid.from_list([[1]])
        result = prim_scale(grid, 3)
        expected = Grid.from_list([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        assert result == expected


class TestColorPrimitives:
    """Test color manipulation primitives."""

    def test_recolor(self):
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

    def test_fill(self):
        """Fill entire grid with color."""
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])
        result = prim_fill(grid, 7)
        expected = Grid.from_list([
            [7, 7],
            [7, 7],
        ])
        assert result == expected


class TestTilingPrimitives:
    """Test tiling primitives."""

    def test_tile_h(self):
        """Tile horizontally."""
        grid = Grid.from_list([
            [1, 2],
        ])
        result = prim_tile_h(grid, 3)
        expected = Grid.from_list([
            [1, 2, 1, 2, 1, 2],
        ])
        assert result == expected

    def test_tile_v(self):
        """Tile vertically."""
        grid = Grid.from_list([
            [1],
            [2],
        ])
        result = prim_tile_v(grid, 2)
        expected = Grid.from_list([
            [1],
            [2],
            [1],
            [2],
        ])
        assert result == expected


class TestInterpreter:
    """Test DSL interpreter."""

    def test_interpret_identity(self):
        """Interpret identity primitive."""
        ast = PrimitiveNode("identity")
        grid = Grid.from_list([[1, 2], [3, 4]])

        program = make_program(ast)
        result = program(grid)

        assert result == grid

    def test_interpret_primitive_with_args(self):
        """Interpret primitive with arguments."""
        ast = PrimitiveNode("rotate90", [LiteralNode(2)])
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])

        result = run_on_grid(ast, grid)
        expected = Grid.from_list([
            [4, 3],
            [2, 1],
        ])

        assert result == expected

    def test_interpret_composition(self):
        """Interpret composition of primitives."""
        # Compose: reflect_h >> reflect_v
        ast = ComposeNode([
            PrimitiveNode("reflect_h"),
            PrimitiveNode("reflect_v"),
        ])
        grid = Grid.from_list([
            [1, 2],
            [3, 4],
        ])

        result = run_on_grid(ast, grid)

        # reflect_h: [[2,1],[4,3]], then reflect_v: [[4,3],[2,1]]
        expected = Grid.from_list([
            [4, 3],
            [2, 1],
        ])

        assert result == expected

    def test_compose_is_associative(self):
        """Composition should be associative."""
        grid = Grid.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        # (reflect_h >> reflect_v) >> transpose
        ast1 = ComposeNode([
            ComposeNode([
                PrimitiveNode("reflect_h"),
                PrimitiveNode("reflect_v"),
            ]),
            PrimitiveNode("transpose"),
        ])

        # reflect_h >> (reflect_v >> transpose)
        ast2 = ComposeNode([
            PrimitiveNode("reflect_h"),
            ComposeNode([
                PrimitiveNode("reflect_v"),
                PrimitiveNode("transpose"),
            ]),
        ])

        # Both should give same result
        result1 = run_on_grid(ast1, grid)
        result2 = run_on_grid(ast2, grid)

        assert result1 == result2


class TestPrettyPrint:
    """Test AST pretty printing."""

    def test_simple_primitive(self):
        """Pretty print simple primitive."""
        ast = PrimitiveNode("identity")
        source = ast_to_source(ast)
        assert source == "identity"

    def test_primitive_with_args(self):
        """Pretty print primitive with arguments."""
        ast = PrimitiveNode("rotate90", [LiteralNode(2)])
        source = ast_to_source(ast)
        assert source == "rotate90(2)"

    def test_composition(self):
        """Pretty print composition."""
        ast = ComposeNode([
            PrimitiveNode("reflect_h"),
            PrimitiveNode("reflect_v"),
        ])
        source = ast_to_source(ast)
        assert source == "reflect_h >> reflect_v"


class TestPrimitivesRegistry:
    """Test primitive registry."""

    def test_all_primitives_registered(self):
        """All expected primitives should be registered."""
        expected = [
            "identity", "crop_to_content", "rotate90",
            "reflect_h", "reflect_v", "transpose",
            "recolor", "fill", "scale",
        ]
        for name in expected:
            assert name in PRIMITIVES, f"Primitive {name} not registered"

    def test_primitives_have_signatures(self):
        """All primitives should have type signatures."""
        for name, spec in PRIMITIVES.items():
            assert spec.signature is not None, f"{name} has no signature"
            assert spec.implementation is not None, f"{name} has no implementation"
