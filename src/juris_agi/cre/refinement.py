"""
Refinement engine for near-miss solutions.

Uses symbolic diffs to propose local edits when synthesis gets close.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable, Tuple

from ..core.types import Grid, ARCTask
from ..dsl.ast import (
    ASTNode,
    PrimitiveNode,
    ComposeNode,
    LiteralNode,
    walk_ast,
    transform_ast,
)
from ..dsl.primitives import PRIMITIVES
from ..dsl.interpreter import make_program
from ..dsl.prettyprint import ast_to_source
from .critic_symbolic import SymbolicCritic, SymbolicDiff, CriticResult


class EditType(Enum):
    """Types of edits to programs."""
    SWAP_PRIMITIVE = auto()
    TWEAK_ARG = auto()
    INSERT_PRIMITIVE = auto()
    REMOVE_PRIMITIVE = auto()
    WRAP_PRIMITIVE = auto()
    SWAP_ORDER = auto()  # Swap order of operations in composition
    ADJUST_RECOLOR = auto()  # Adjust recolor_map mapping


@dataclass
class EditOperation:
    """A single edit operation on a program."""
    edit_type: EditType
    location: int  # Index in AST walk order
    original: Optional[str] = None
    replacement: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable description of the edit."""
        if self.edit_type == EditType.SWAP_PRIMITIVE:
            return f"Swap {self.original} with {self.replacement}"
        elif self.edit_type == EditType.TWEAK_ARG:
            return f"Adjust argument at position {self.location}"
        elif self.edit_type == EditType.INSERT_PRIMITIVE:
            return f"Insert {self.replacement} at position {self.location}"
        elif self.edit_type == EditType.REMOVE_PRIMITIVE:
            return f"Remove {self.original} at position {self.location}"
        elif self.edit_type == EditType.WRAP_PRIMITIVE:
            return f"Wrap with {self.replacement}"
        elif self.edit_type == EditType.SWAP_ORDER:
            return f"Swap operation order at {self.location}"
        elif self.edit_type == EditType.ADJUST_RECOLOR:
            return f"Adjust recolor mapping: {self.details}"
        return f"Edit: {self.edit_type.name}"


@dataclass
class RefinementResult:
    """Result from refinement attempt."""
    success: bool
    improved: bool
    original_score: float
    new_score: float
    original_program: str
    refined_program: Optional[str] = None
    refined_ast: Optional[ASTNode] = None
    edits_applied: List[EditOperation] = field(default_factory=list)
    iterations: int = 0


class RefinementEngine:
    """
    Refinement engine that improves near-miss programs.

    Uses symbolic diffs to guide local edits.
    """

    def __init__(
        self,
        max_iterations: int = 20,
        max_edits_per_iteration: int = 10,
    ):
        self.max_iterations = max_iterations
        self.max_edits_per_iteration = max_edits_per_iteration
        self.critic = SymbolicCritic()

        # Edit generators
        self.edit_generators: List[Callable[[ASTNode, List[Dict]], List[Tuple[EditOperation, ASTNode]]]] = [
            self._generate_swap_edits,
            self._generate_arg_tweaks,
            self._generate_insert_edits,
            self._generate_removal_edits,
            self._generate_order_swap_edits,
            self._generate_translate_tweaks,
            self._generate_recolor_tweaks,
        ]

    def refine(
        self,
        program: ASTNode,
        task: ARCTask,
        initial_critique: Optional[CriticResult] = None,
    ) -> RefinementResult:
        """
        Attempt to refine a near-miss program.

        Args:
            program: The program to refine
            task: The task to solve
            initial_critique: Pre-computed critique (optional)

        Returns:
            RefinementResult with refined program if successful
        """
        if initial_critique is None:
            initial_critique = self.critic.evaluate(program, task)

        if initial_critique.is_certified:
            # Already correct, no refinement needed
            return RefinementResult(
                success=True,
                improved=False,
                original_score=100.0,
                new_score=100.0,
                original_program=ast_to_source(program),
                refined_ast=program,
                refined_program=ast_to_source(program),
            )

        original_score = self._compute_score(initial_critique)
        best_score = original_score
        best_ast = program
        best_edits: List[EditOperation] = []

        current_ast = program
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            # Get refinement hints from critic
            hints = self.critic.compute_refinement_hints(initial_critique.diffs)

            # Generate candidate edits
            candidates = self._generate_edit_candidates(current_ast, hints)

            if not candidates:
                break

            # Evaluate candidates
            improved = False
            for edit, edited_ast in candidates[:self.max_edits_per_iteration]:
                critique = self.critic.evaluate(edited_ast, task)
                score = self._compute_score(critique)

                if critique.is_certified:
                    # Found exact solution!
                    return RefinementResult(
                        success=True,
                        improved=True,
                        original_score=original_score,
                        new_score=100.0,
                        original_program=ast_to_source(program),
                        refined_program=ast_to_source(edited_ast),
                        refined_ast=edited_ast,
                        edits_applied=best_edits + [edit],
                        iterations=iterations,
                    )

                if score > best_score:
                    best_score = score
                    best_ast = edited_ast
                    best_edits.append(edit)
                    current_ast = edited_ast
                    initial_critique = critique
                    improved = True
                    break

            if not improved:
                break

        return RefinementResult(
            success=best_score >= 100.0,
            improved=best_score > original_score,
            original_score=original_score,
            new_score=best_score,
            original_program=ast_to_source(program),
            refined_program=ast_to_source(best_ast),
            refined_ast=best_ast,
            edits_applied=best_edits,
            iterations=iterations,
        )

    def _compute_score(self, critique: CriticResult) -> float:
        """Compute score from critique."""
        if critique.exact_match_all:
            return 100.0

        total_accuracy = sum(
            r.get("pixel_accuracy", 0.0)
            for r in critique.pair_results
            if "pixel_accuracy" in r
        )
        num_pairs = len(critique.pair_results)
        if num_pairs == 0:
            return 0.0

        avg_accuracy = total_accuracy / num_pairs
        return avg_accuracy * 50.0

    def _generate_edit_candidates(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate candidate edits based on hints."""
        candidates = []

        for generator in self.edit_generators:
            new_candidates = generator(ast, hints)
            candidates.extend(new_candidates)

        return candidates

    def _generate_swap_edits(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that swap one primitive for another."""
        candidates = []

        # Find all primitive nodes
        nodes = walk_ast(ast)
        prim_nodes = [
            (i, n) for i, n in enumerate(nodes)
            if isinstance(n, PrimitiveNode)
        ]

        # Swap candidates based on primitive compatibility
        swap_groups = {
            "rotate90": ["reflect_h", "reflect_v", "transpose"],
            "reflect_h": ["reflect_v", "rotate90", "transpose"],
            "reflect_v": ["reflect_h", "rotate90", "transpose"],
            "crop_to_content": ["identity"],
            "identity": ["crop_to_content"],
        }

        for idx, node in prim_nodes:
            if node.name in swap_groups:
                for replacement in swap_groups[node.name]:
                    edit = EditOperation(
                        edit_type=EditType.SWAP_PRIMITIVE,
                        location=idx,
                        original=node.name,
                        replacement=replacement,
                    )

                    # Create edited AST
                    def make_swap(n: ASTNode, target_idx: int = idx, repl: str = replacement) -> Optional[ASTNode]:
                        if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                            return PrimitiveNode(repl, n.args)
                        return None

                    edited = transform_ast(ast, make_swap)
                    candidates.append((edit, edited))

        return candidates

    def _generate_arg_tweaks(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that tweak primitive arguments."""
        candidates = []

        nodes = walk_ast(ast)
        for idx, node in enumerate(nodes):
            if not isinstance(node, PrimitiveNode):
                continue

            # Tweak rotation arguments
            if node.name == "rotate90" and node.args:
                for new_val in [1, 2, 3]:
                    if isinstance(node.args[0], LiteralNode):
                        if node.args[0].value == new_val:
                            continue

                    edit = EditOperation(
                        edit_type=EditType.TWEAK_ARG,
                        location=idx,
                        details={"new_value": new_val},
                    )

                    def make_tweak(n: ASTNode, target_idx: int = idx, val: int = new_val) -> Optional[ASTNode]:
                        if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                            return PrimitiveNode(n.name, [LiteralNode(val)])
                        return None

                    edited = transform_ast(ast, make_tweak)
                    candidates.append((edit, edited))

            # Tweak scale arguments
            if node.name == "scale" and node.args:
                for new_val in [2, 3, 4]:
                    if isinstance(node.args[0], LiteralNode):
                        if node.args[0].value == new_val:
                            continue

                    edit = EditOperation(
                        edit_type=EditType.TWEAK_ARG,
                        location=idx,
                        details={"new_value": new_val},
                    )

                    def make_scale_tweak(n: ASTNode, target_idx: int = idx, val: int = new_val) -> Optional[ASTNode]:
                        if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                            return PrimitiveNode(n.name, [LiteralNode(val)])
                        return None

                    edited = transform_ast(ast, make_scale_tweak)
                    candidates.append((edit, edited))

        return candidates

    def _generate_insert_edits(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that insert new primitives."""
        candidates = []

        # Primitives to try inserting
        insert_prims = [
            PrimitiveNode("crop_to_content"),
            PrimitiveNode("reflect_h"),
            PrimitiveNode("reflect_v"),
            PrimitiveNode("transpose"),
        ]
        for n in [1, 2, 3]:
            insert_prims.append(PrimitiveNode("rotate90", [LiteralNode(n)]))

        for prim in insert_prims:
            # Insert at beginning
            if isinstance(ast, ComposeNode):
                new_ops = [prim] + ast.operations
                edited = ComposeNode(new_ops)
            else:
                edited = ComposeNode([prim, ast])

            edit = EditOperation(
                edit_type=EditType.INSERT_PRIMITIVE,
                location=0,
                replacement=prim.name,
            )
            candidates.append((edit, edited))

            # Insert at end
            if isinstance(ast, ComposeNode):
                new_ops = ast.operations + [prim]
                edited = ComposeNode(new_ops)
            else:
                edited = ComposeNode([ast, prim])

            edit = EditOperation(
                edit_type=EditType.INSERT_PRIMITIVE,
                location=-1,
                replacement=prim.name,
            )
            candidates.append((edit, edited))

        return candidates

    def _generate_removal_edits(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that remove primitives."""
        candidates = []

        if not isinstance(ast, ComposeNode):
            return candidates

        if len(ast.operations) <= 1:
            return candidates

        for i, op in enumerate(ast.operations):
            if not isinstance(op, PrimitiveNode):
                continue

            # Skip removing identity (it's a no-op anyway)
            if op.name == "identity":
                continue

            edit = EditOperation(
                edit_type=EditType.REMOVE_PRIMITIVE,
                location=i,
                original=op.name,
            )

            new_ops = ast.operations[:i] + ast.operations[i+1:]
            if len(new_ops) == 1:
                edited = new_ops[0]
            else:
                edited = ComposeNode(new_ops)

            candidates.append((edit, edited))

        return candidates

    def _generate_order_swap_edits(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that swap order of operations in compositions."""
        candidates = []

        if not isinstance(ast, ComposeNode):
            return candidates

        if len(ast.operations) < 2:
            return candidates

        # Try swapping adjacent pairs
        for i in range(len(ast.operations) - 1):
            op1 = ast.operations[i]
            op2 = ast.operations[i + 1]

            # Create swapped version
            new_ops = list(ast.operations)
            new_ops[i] = op2
            new_ops[i + 1] = op1

            edit = EditOperation(
                edit_type=EditType.SWAP_ORDER,
                location=i,
                original=f"{getattr(op1, 'name', 'op')},{getattr(op2, 'name', 'op')}",
                replacement=f"{getattr(op2, 'name', 'op')},{getattr(op1, 'name', 'op')}",
            )

            edited = ComposeNode(new_ops)
            candidates.append((edit, edited))

        return candidates

    def _generate_translate_tweaks(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that tweak translate(dx, dy) arguments."""
        candidates = []

        nodes = walk_ast(ast)
        for idx, node in enumerate(nodes):
            if not isinstance(node, PrimitiveNode):
                continue

            if node.name != "translate":
                continue

            # Get current dx, dy values
            current_dx = 0
            current_dy = 0
            if len(node.args) >= 2:
                if isinstance(node.args[0], LiteralNode):
                    current_dx = node.args[0].value
                if isinstance(node.args[1], LiteralNode):
                    current_dy = node.args[1].value

            # Try small adjustments to dx and dy
            deltas = [-2, -1, 1, 2]
            for d_dx in deltas:
                for d_dy in deltas:
                    new_dx = current_dx + d_dx
                    new_dy = current_dy + d_dy

                    # Skip if same as current
                    if new_dx == current_dx and new_dy == current_dy:
                        continue

                    edit = EditOperation(
                        edit_type=EditType.TWEAK_ARG,
                        location=idx,
                        original=f"translate({current_dx}, {current_dy})",
                        replacement=f"translate({new_dx}, {new_dy})",
                        details={"new_dx": new_dx, "new_dy": new_dy},
                    )

                    def make_translate_tweak(
                        n: ASTNode,
                        target_idx: int = idx,
                        dx: int = new_dx,
                        dy: int = new_dy,
                    ) -> Optional[ASTNode]:
                        if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                            return PrimitiveNode(n.name, [LiteralNode(dx), LiteralNode(dy)])
                        return None

                    edited = transform_ast(ast, make_translate_tweak)
                    candidates.append((edit, edited))

        return candidates

    def _generate_recolor_tweaks(
        self,
        ast: ASTNode,
        hints: List[Dict[str, Any]],
    ) -> List[Tuple[EditOperation, ASTNode]]:
        """Generate edits that adjust recolor_map mappings."""
        candidates = []

        nodes = walk_ast(ast)
        for idx, node in enumerate(nodes):
            if not isinstance(node, PrimitiveNode):
                continue

            if node.name != "recolor_map":
                continue

            # Get current color mapping
            current_map = {}
            if node.args and isinstance(node.args[0], LiteralNode):
                current_map = node.args[0].value if isinstance(node.args[0].value, dict) else {}

            # Use hints to suggest color changes
            extra_colors = set()
            missing_colors = set()
            for hint in hints:
                if "extra_colors" in hint:
                    extra_colors.update(hint["extra_colors"])
                if "missing_colors" in hint:
                    missing_colors.update(hint["missing_colors"])

            # Try mapping extra colors to missing colors
            if extra_colors and missing_colors:
                for extra in extra_colors:
                    for missing in missing_colors:
                        new_map = dict(current_map)
                        new_map[extra] = missing

                        edit = EditOperation(
                            edit_type=EditType.ADJUST_RECOLOR,
                            location=idx,
                            details={"from": extra, "to": missing, "new_map": new_map},
                        )

                        def make_recolor_tweak(
                            n: ASTNode,
                            target_idx: int = idx,
                            color_map: dict = new_map,
                        ) -> Optional[ASTNode]:
                            if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                                return PrimitiveNode(n.name, [LiteralNode(color_map)])
                            return None

                        edited = transform_ast(ast, make_recolor_tweak)
                        candidates.append((edit, edited))

            # Also try common color swaps (0-9 ARC palette)
            if not candidates:
                # Try swapping each color in the map with adjacent colors
                for src_color, dst_color in current_map.items():
                    for new_dst in [dst_color - 1, dst_color + 1]:
                        if 0 <= new_dst <= 9 and new_dst != dst_color:
                            new_map = dict(current_map)
                            new_map[src_color] = new_dst

                            edit = EditOperation(
                                edit_type=EditType.ADJUST_RECOLOR,
                                location=idx,
                                details={"adjusted": src_color, "from": dst_color, "to": new_dst},
                            )

                            def make_adj_tweak(
                                n: ASTNode,
                                target_idx: int = idx,
                                color_map: dict = new_map,
                            ) -> Optional[ASTNode]:
                                if walk_ast(ast).index(n) == target_idx and isinstance(n, PrimitiveNode):
                                    return PrimitiveNode(n.name, [LiteralNode(color_map)])
                                return None

                            edited = transform_ast(ast, make_adj_tweak)
                            candidates.append((edit, edited))

        return candidates
