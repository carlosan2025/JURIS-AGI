"""
Program synthesizer using beam search over DSL programs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set, TYPE_CHECKING
import heapq
from copy import deepcopy

from ..core.types import Grid, ARCTask, ARCPair
from ..core.metrics import compute_exact_match, compute_pixel_accuracy, evaluate_on_pairs
from ..dsl.ast import ASTNode, PrimitiveNode, ComposeNode, LiteralNode
from ..dsl.interpreter import DSLInterpreter, make_program
from ..dsl.primitives import PRIMITIVES, PrimitiveSpec
from ..dsl.prettyprint import ast_to_source
from .refinement import RefinementEngine
from ..core.metrics import (
    ConstraintSet,
    extract_constraints_from_task,
    DimensionConstraint,
    PaletteConstraint,
    ObjectCountConstraint,
)

if TYPE_CHECKING:
    from ..mal.retrieval import MacroStore
    from ..core.trace import JSONLTraceWriter, SolveTrace


@dataclass
class SynthesisConfig:
    """Configuration for synthesis."""
    max_depth: int = 4
    beam_width: int = 50
    max_iterations: int = 1000
    timeout_seconds: float = 30.0
    use_dimension_pruning: bool = True
    use_palette_pruning: bool = True
    use_object_count_pruning: bool = False  # Soft constraint, disabled by default
    use_constraint_set: bool = True  # Use new ConstraintSet system
    min_pixel_accuracy: float = 0.0
    # Near-miss and refinement settings
    enable_refinement: bool = True
    near_miss_threshold: float = 30.0  # Score threshold to consider near-miss
    top_k_near_miss: int = 5  # Keep top-k near-misses for refinement
    max_refinement_iterations: int = 20
    # MAL integration settings
    use_mal: bool = False  # Enable MAL macro retrieval
    mal_top_k: int = 5  # Number of macros to retrieve
    write_traces: bool = False  # Enable trace writing
    trace_dir: str = "traces"  # Directory for trace files
    # WME integration settings
    use_wme: bool = False  # Enable WME advisory scoring
    wme_robustness_weight: float = 0.1  # Weight for robustness in soft scoring
    wme_length_weight: float = 0.05  # Weight for program length penalty


@dataclass
class SynthesisResult:
    """Result from synthesis."""
    success: bool
    program: Optional[ASTNode] = None
    program_source: Optional[str] = None
    score: float = 0.0
    iterations: int = 0
    nodes_explored: int = 0
    candidates_pruned: int = 0
    error: Optional[str] = None
    # Near-miss and refinement info
    near_misses: List[Tuple[ASTNode, float]] = field(default_factory=list)
    refinement_applied: bool = False
    refinement_improved: bool = False
    # MAL info
    macros_retrieved: List[str] = field(default_factory=list)
    macros_used: List[str] = field(default_factory=list)
    synthesis_time_ms: float = 0.0
    # WME info
    robustness_score: float = 0.0
    hard_veto_applied: bool = False  # True if a candidate was rejected by hard constraint
    soft_score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class Candidate:
    """A candidate program during search."""
    ast: ASTNode
    score: float
    depth: int
    train_results: Optional[Dict[str, Any]] = None

    def __lt__(self, other: "Candidate") -> bool:
        # Higher score is better, so we invert for min-heap
        return self.score > other.score


class Synthesizer(ABC):
    """Abstract base class for synthesizers."""

    @abstractmethod
    def synthesize(
        self,
        task: ARCTask,
        config: Optional[SynthesisConfig] = None,
    ) -> SynthesisResult:
        """
        Synthesize a program for the given task.

        Args:
            task: The ARC task to solve
            config: Synthesis configuration

        Returns:
            SynthesisResult with the found program (if any)
        """
        pass


class BeamSearchSynthesizer(Synthesizer):
    """
    Beam search synthesizer over DSL programs.

    Uses constraint-based pruning:
    - Output dimension match
    - Palette subset
    - Exact match on training pairs

    Also keeps track of near-miss programs and applies refinement.
    """

    def __init__(
        self,
        config: Optional[SynthesisConfig] = None,
        macro_store: Optional["MacroStore"] = None,
        trace_writer: Optional["JSONLTraceWriter"] = None,
    ):
        self.config = config or SynthesisConfig()
        self.interpreter = DSLInterpreter()

        # Refinement engine for near-misses
        self.refinement_engine = RefinementEngine(
            max_iterations=self.config.max_refinement_iterations,
        )

        # Select primitives for synthesis (Grid -> Grid primarily)
        self.grid_primitives = self._select_grid_primitives()

        # Track near-miss candidates
        self.near_misses: List[Tuple[ASTNode, float]] = []

        # MAL integration (optional)
        self.macro_store = macro_store
        self.trace_writer = trace_writer
        self._retrieved_macros: List[str] = []
        self._used_macros: List[str] = []

    def _select_grid_primitives(self) -> List[PrimitiveSpec]:
        """Select primitives that are useful for synthesis."""
        # Focus on Grid -> Grid and simple transformations
        useful_prims = [
            # Core transformations
            "identity",
            "crop_to_content",
            "rotate90",
            "reflect_h",
            "reflect_v",
            "transpose",
            # Scaling and tiling
            "scale",
            "tile_h",
            "tile_v",
            "tile_repeat",
            # Color operations
            "fill_background",
            # Mask operations
            "invert_mask",
        ]
        return [PRIMITIVES[name] for name in useful_prims if name in PRIMITIVES]

    def synthesize(
        self,
        task: ARCTask,
        config: Optional[SynthesisConfig] = None,
    ) -> SynthesisResult:
        """Run beam search synthesis."""
        import time
        start_time = time.time()

        cfg = config or self.config

        # Reset state for this synthesis run
        self.near_misses = []
        self._retrieved_macros = []
        self._used_macros = []

        # Create trace if enabled
        trace = None
        if cfg.write_traces:
            from ..core.trace import create_trace_from_task
            trace = create_trace_from_task(task, task.task_id)

        # MAL: Retrieve relevant macros if enabled
        if cfg.use_mal and self.macro_store is not None:
            self._retrieve_macros_for_task(task, cfg.mal_top_k)
            if trace:
                trace.log("mal_retrieval", "cre", macros_retrieved=self._retrieved_macros)

        # Extract constraints from training pairs
        constraints = self._extract_constraints(task.train)

        # Extract advanced constraint set if enabled
        constraint_set: Optional[ConstraintSet] = None
        if cfg.use_constraint_set:
            try:
                constraint_set = self._extract_constraint_set(task)
            except Exception:
                constraint_set = None  # Fallback to legacy if extraction fails

        # Initialize beam with identity and basic primitives
        beam: List[Candidate] = []
        initial_candidates = self._generate_initial_candidates()

        for ast in initial_candidates:
            score, train_results = self._evaluate_candidate(ast, task.train)
            if train_results and train_results.get("all_exact_match"):
                # Found perfect solution immediately
                result = SynthesisResult(
                    success=True,
                    program=ast,
                    program_source=ast_to_source(ast),
                    score=score,
                    iterations=1,
                    nodes_explored=1,
                    macros_retrieved=self._retrieved_macros,
                    macros_used=self._used_macros,
                    synthesis_time_ms=(time.time() - start_time) * 1000,
                )
                self._finalize_trace(trace, result, cfg)
                return result

            # Track near-misses
            if score >= cfg.near_miss_threshold:
                self._add_near_miss(ast, score, cfg.top_k_near_miss)

            beam.append(Candidate(
                ast=ast,
                score=score,
                depth=ast.depth(),
                train_results=train_results,
            ))

        # Sort beam by score (descending)
        beam.sort(key=lambda c: c.score, reverse=True)
        beam = beam[:cfg.beam_width]

        iterations = 0
        nodes_explored = len(beam)
        candidates_pruned = 0

        while iterations < cfg.max_iterations and beam:
            iterations += 1
            new_candidates: List[Candidate] = []

            for candidate in beam:
                if candidate.depth >= cfg.max_depth:
                    continue

                # Expand candidate
                expansions = self._expand_candidate(candidate.ast)

                for expanded_ast in expansions:
                    nodes_explored += 1

                    # Prune based on constraints
                    if self._should_prune(expanded_ast, task.train, cfg, constraints, constraint_set):
                        candidates_pruned += 1
                        continue

                    score, train_results = self._evaluate_candidate(
                        expanded_ast, task.train
                    )

                    if train_results and train_results.get("all_exact_match"):
                        # Found perfect solution!
                        result = SynthesisResult(
                            success=True,
                            program=expanded_ast,
                            program_source=ast_to_source(expanded_ast),
                            score=score,
                            iterations=iterations,
                            nodes_explored=nodes_explored,
                            candidates_pruned=candidates_pruned,
                            near_misses=self.near_misses,
                            macros_retrieved=self._retrieved_macros,
                            macros_used=self._used_macros,
                            synthesis_time_ms=(time.time() - start_time) * 1000,
                        )
                        self._finalize_trace(trace, result, cfg)
                        return result

                    # Track near-misses
                    if score >= cfg.near_miss_threshold:
                        self._add_near_miss(expanded_ast, score, cfg.top_k_near_miss)

                    if score > cfg.min_pixel_accuracy:
                        new_candidates.append(Candidate(
                            ast=expanded_ast,
                            score=score,
                            depth=expanded_ast.depth(),
                            train_results=train_results,
                        ))

            # Merge and select top beam_width candidates
            all_candidates = beam + new_candidates
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beam = all_candidates[:cfg.beam_width]

            # Early termination if no progress
            if not new_candidates:
                break

        # No exact solution found - try refinement on near-misses
        if cfg.enable_refinement and self.near_misses:
            refined_result = self._try_refinement(
                task, iterations, nodes_explored, candidates_pruned, start_time
            )
            if refined_result is not None:
                self._finalize_trace(trace, refined_result, cfg)
                return refined_result

        # Return best candidate found (even if not perfect)
        if beam:
            best = beam[0]
            result = SynthesisResult(
                success=False,
                program=best.ast,
                program_source=ast_to_source(best.ast),
                score=best.score,
                iterations=iterations,
                nodes_explored=nodes_explored,
                candidates_pruned=candidates_pruned,
                error="No exact solution found",
                near_misses=self.near_misses,
                refinement_applied=cfg.enable_refinement and len(self.near_misses) > 0,
                macros_retrieved=self._retrieved_macros,
                macros_used=self._used_macros,
                synthesis_time_ms=(time.time() - start_time) * 1000,
            )
            self._finalize_trace(trace, result, cfg)
            return result

        result = SynthesisResult(
            success=False,
            iterations=iterations,
            nodes_explored=nodes_explored,
            candidates_pruned=candidates_pruned,
            error="Search exhausted without finding solution",
            near_misses=self.near_misses,
            macros_retrieved=self._retrieved_macros,
            macros_used=self._used_macros,
            synthesis_time_ms=(time.time() - start_time) * 1000,
        )
        self._finalize_trace(trace, result, cfg)
        return result

    def _add_near_miss(self, ast: ASTNode, score: float, top_k: int) -> None:
        """Add a near-miss candidate, keeping only top-k."""
        self.near_misses.append((ast, score))
        # Keep sorted by score (descending)
        self.near_misses.sort(key=lambda x: x[1], reverse=True)
        # Keep only top-k
        self.near_misses = self.near_misses[:top_k]

    def _try_refinement(
        self,
        task: ARCTask,
        iterations: int,
        nodes_explored: int,
        candidates_pruned: int,
        start_time: float,
    ) -> Optional[SynthesisResult]:
        """Try refining near-miss programs to find exact solution."""
        import time
        for ast, score in self.near_misses:
            try:
                result = self.refinement_engine.refine(ast, task)
                if result.success:
                    return SynthesisResult(
                        success=True,
                        program=result.refined_ast,
                        program_source=result.refined_program,
                        score=100.0,
                        iterations=iterations,
                        nodes_explored=nodes_explored,
                        candidates_pruned=candidates_pruned,
                        near_misses=self.near_misses,
                        refinement_applied=True,
                        refinement_improved=True,
                        macros_retrieved=self._retrieved_macros,
                        macros_used=self._used_macros,
                        synthesis_time_ms=(time.time() - start_time) * 1000,
                    )
            except Exception:
                continue
        return None

    def _retrieve_macros_for_task(self, task: ARCTask, top_k: int) -> None:
        """Retrieve macros from MAL for seeding beam search."""
        if self.macro_store is None:
            return

        try:
            from ..mal.retrieval import retrieve_macros
            results = retrieve_macros(task, self.macro_store, top_k)
            self._retrieved_macros = [m.code for m, _ in results]
        except Exception:
            self._retrieved_macros = []

    def _finalize_trace(
        self,
        trace: Optional["SolveTrace"],
        result: SynthesisResult,
        cfg: SynthesisConfig,
    ) -> None:
        """Finalize and write trace to disk."""
        if trace is None or not cfg.write_traces:
            return

        try:
            # Update trace with result info
            trace.finalize(
                success=result.success,
                program=result.program_source,
            )
            trace.final_metrics = {
                "score": result.score,
                "iterations": result.iterations,
                "nodes_explored": result.nodes_explored,
                "candidates_pruned": result.candidates_pruned,
                "synthesis_time_ms": result.synthesis_time_ms,
            }

            # Log macros info
            if result.macros_retrieved:
                trace.log(
                    "mal_macros",
                    "cre",
                    retrieved=result.macros_retrieved,
                    used=result.macros_used,
                )

            # Write trace
            if self.trace_writer is not None:
                self.trace_writer.write_trace(trace)
            else:
                from ..core.trace import JSONLTraceWriter
                writer = JSONLTraceWriter(cfg.trace_dir)
                writer.write_trace(trace)

        except Exception:
            pass  # Don't fail synthesis due to trace writing errors

    def _generate_initial_candidates(self) -> List[ASTNode]:
        """Generate initial candidate programs."""
        candidates = []

        # Identity
        candidates.append(PrimitiveNode("identity"))

        # All zero-argument Grid->Grid primitives
        for prim in self.grid_primitives:
            if prim.name != "identity":
                candidates.append(PrimitiveNode(prim.name))

        # Rotate with different arguments
        for n in [1, 2, 3]:
            candidates.append(PrimitiveNode("rotate90", [LiteralNode(n)]))

        # Scale with small factors
        for factor in [2, 3]:
            if "scale" in PRIMITIVES:
                candidates.append(PrimitiveNode("scale", [LiteralNode(factor)]))

        # Tile operations with common repeat counts
        for n in [2, 3]:
            if "tile_h" in PRIMITIVES:
                candidates.append(PrimitiveNode("tile_h", [LiteralNode(n)]))
            if "tile_v" in PRIMITIVES:
                candidates.append(PrimitiveNode("tile_v", [LiteralNode(n)]))

        # tile_repeat with common 2D patterns
        if "tile_repeat" in PRIMITIVES:
            for rows in [2, 3]:
                for cols in [2, 3]:
                    candidates.append(PrimitiveNode("tile_repeat", [LiteralNode(rows), LiteralNode(cols)]))

        # Fill background with common ARC colors
        if "fill_background" in PRIMITIVES:
            for color in [1, 2, 3, 4, 5]:  # Common ARC non-black colors
                candidates.append(PrimitiveNode("fill_background", [LiteralNode(color)]))

        return candidates

    def _expand_candidate(self, ast: ASTNode) -> List[ASTNode]:
        """Expand a candidate by composing with more primitives."""
        expansions = []

        # Compose with each primitive
        for prim in self.grid_primitives:
            # Prepend primitive
            if isinstance(ast, ComposeNode):
                new_ops = [PrimitiveNode(prim.name)] + ast.operations
            else:
                new_ops = [PrimitiveNode(prim.name), ast]
            expansions.append(ComposeNode(new_ops))

            # Append primitive
            if isinstance(ast, ComposeNode):
                new_ops = ast.operations + [PrimitiveNode(prim.name)]
            else:
                new_ops = [ast, PrimitiveNode(prim.name)]
            expansions.append(ComposeNode(new_ops))

        # Add rotations with different arguments
        for n in [1, 2, 3]:
            rot = PrimitiveNode("rotate90", [LiteralNode(n)])
            if isinstance(ast, ComposeNode):
                expansions.append(ComposeNode(ast.operations + [rot]))
            else:
                expansions.append(ComposeNode([ast, rot]))

        return expansions

    def _evaluate_candidate(
        self,
        ast: ASTNode,
        train_pairs: List[ARCPair],
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Evaluate a candidate on training pairs."""
        try:
            program = make_program(ast)
            results = evaluate_on_pairs(program, train_pairs)

            # Score: prioritize exact match, then pixel accuracy
            if results["all_exact_match"]:
                score = 100.0
            else:
                score = results["average_pixel_accuracy"] * 50.0

            # Penalize program length (MDL)
            score -= ast.size() * 0.1

            return score, results

        except Exception as e:
            return -1.0, {"error": str(e)}

    def _apply_selection_scoring(
        self,
        ast: ASTNode,
        base_score: float,
        task: ARCTask,
        cfg: SynthesisConfig,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply soft scoring for program selection.

        Combines base score with:
        - Program length penalty (shorter is better)
        - Robustness score (higher is better)

        This is the "soft" part of selection - advisory, not a veto.

        Args:
            ast: The program AST
            base_score: Score from training pair evaluation
            task: The task for robustness checking
            cfg: Synthesis configuration

        Returns:
            Tuple of (adjusted_score, score_breakdown)
        """
        breakdown: Dict[str, float] = {"base": base_score}

        if not cfg.use_wme:
            return base_score, breakdown

        # Length penalty (shorter programs preferred)
        length_penalty = ast.size() * cfg.wme_length_weight
        breakdown["length_penalty"] = -length_penalty

        # Robustness score (if enabled and we have training data)
        robustness_bonus = 0.0
        if cfg.wme_robustness_weight > 0 and task.train:
            try:
                from ..wme.robustness import quick_robustness_check
                # Only check robustness for high-scoring candidates to save time
                if base_score >= cfg.near_miss_threshold:
                    robustness = quick_robustness_check(
                        ast, task.train[0].input, num_counterfactuals=3
                    )
                    robustness_bonus = robustness * cfg.wme_robustness_weight * 10
                    breakdown["robustness_bonus"] = robustness_bonus
            except Exception:
                pass  # Don't fail on robustness check errors

        adjusted_score = base_score - length_penalty + robustness_bonus
        breakdown["final"] = adjusted_score

        return adjusted_score, breakdown

    def _apply_hard_veto(
        self,
        ast: ASTNode,
        task: ARCTask,
        train_results: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Apply hard veto for candidate rejection.

        Hard veto means the candidate is absolutely rejected - it cannot
        be part of the final solution. This is different from soft scoring.

        Hard veto conditions:
        1. Program fails on ANY training pair (crashes/exceptions)
        2. Program produces wrong dimensions on ANY training pair
        3. Program produces empty output

        Args:
            ast: The program AST
            task: The task being solved
            train_results: Results from training pair evaluation

        Returns:
            Tuple of (should_veto, reason)
        """
        if train_results is None:
            return True, "No evaluation results"

        if "error" in train_results:
            return True, f"Execution error: {train_results['error']}"

        # Check dimension match on all training pairs
        try:
            program = make_program(ast)
            for i, pair in enumerate(task.train):
                output = program(pair.input)

                # Empty output is a hard veto
                if output is None or output.height == 0 or output.width == 0:
                    return True, f"Empty output on train pair {i}"

                # Dimension mismatch is a hard veto
                if output.shape != pair.output.shape:
                    return True, f"Dimension mismatch on train pair {i}: got {output.shape}, expected {pair.output.shape}"

        except Exception as e:
            return True, f"Execution failed: {e}"

        return False, ""

    def select_best_candidate(
        self,
        candidates: List["Candidate"],
        task: ARCTask,
        cfg: SynthesisConfig,
    ) -> Optional["Candidate"]:
        """
        Select the best candidate using hard veto and soft scoring.

        Selection process:
        1. Apply hard veto to filter out invalid candidates
        2. Apply soft scoring to rank remaining candidates
        3. Return highest-scoring candidate

        Args:
            candidates: List of candidates to select from
            task: The task being solved
            cfg: Synthesis configuration

        Returns:
            Best candidate or None if all were vetoed
        """
        valid_candidates: List[Tuple["Candidate", float, Dict[str, float]]] = []

        for candidate in candidates:
            # Step 1: Hard veto
            should_veto, reason = self._apply_hard_veto(
                candidate.ast, task, candidate.train_results
            )
            if should_veto:
                continue  # Reject this candidate

            # Step 2: Soft scoring
            adjusted_score, breakdown = self._apply_selection_scoring(
                candidate.ast, candidate.score, task, cfg
            )

            valid_candidates.append((candidate, adjusted_score, breakdown))

        if not valid_candidates:
            return None

        # Step 3: Select best by adjusted score
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_score, best_breakdown = valid_candidates[0]

        # Update candidate score with adjusted value
        best_candidate.score = best_score

        return best_candidate

    def _extract_constraints(
        self,
        train_pairs: List[ARCPair],
    ) -> Dict[str, Any]:
        """Extract constraints from training pairs."""
        constraints: Dict[str, Any] = {
            "output_dims": [],
            "output_palettes": [],
            "dim_ratios": [],
            "constraint_set": None,  # New ConstraintSet object
        }

        for pair in train_pairs:
            constraints["output_dims"].append(pair.output.shape)
            constraints["output_palettes"].append(pair.output.palette)

            # Dimension ratios
            h_ratio = pair.output.height / pair.input.height if pair.input.height > 0 else 0
            w_ratio = pair.output.width / pair.input.width if pair.input.width > 0 else 0
            constraints["dim_ratios"].append((h_ratio, w_ratio))

        return constraints

    def _extract_constraint_set(self, task: ARCTask) -> ConstraintSet:
        """Extract a ConstraintSet from the task for advanced pruning."""
        return extract_constraints_from_task(task)

    def _should_prune(
        self,
        ast: ASTNode,
        train_pairs: List[ARCPair],
        cfg: SynthesisConfig,
        constraints: Dict[str, Any],
        constraint_set: Optional[ConstraintSet] = None,
    ) -> bool:
        """Check if candidate should be pruned based on constraints."""
        try:
            program = make_program(ast)

            for i, pair in enumerate(train_pairs):
                output = program(pair.input)

                # Use new ConstraintSet if available and enabled
                if cfg.use_constraint_set and constraint_set is not None:
                    satisfied, details = constraint_set.check_all(output, pair.input)

                    # Hard constraints must be satisfied
                    if not details["dimension_ok"] and cfg.use_dimension_pruning:
                        return True
                    if not details["palette_ok"] and cfg.use_palette_pruning:
                        return True

                    # Soft object count constraint (only prune if very wrong)
                    if cfg.use_object_count_pruning and not details.get("object_count_ok", True):
                        severity = details.get("object_count_severity", 0.0)
                        if severity > 0.8:  # Only prune if severely violated
                            return True
                else:
                    # Fallback to legacy constraint checking
                    # Dimension pruning
                    if cfg.use_dimension_pruning:
                        expected_dims = constraints["output_dims"][i]
                        if output.shape != expected_dims:
                            # Check if dimensions are reasonable
                            if output.height > 100 or output.width > 100:
                                return True
                            if output.height == 0 or output.width == 0:
                                return True

                    # Palette pruning
                    if cfg.use_palette_pruning:
                        expected_palette = constraints["output_palettes"][i]
                        # Allow if output palette is subset of expected
                        if not output.palette.issubset(expected_palette | {0}):
                            # Extra colors not in expected output
                            return True

            return False

        except Exception:
            return True


class EnumerativeSynthesizer(Synthesizer):
    """
    Exhaustive enumeration synthesizer (for small search spaces).

    Enumerates all programs up to a given depth.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.interpreter = DSLInterpreter()

    def synthesize(
        self,
        task: ARCTask,
        config: Optional[SynthesisConfig] = None,
    ) -> SynthesisResult:
        """Enumerate programs until finding a solution."""
        cfg = config or SynthesisConfig(max_depth=self.max_depth)

        nodes_explored = 0
        for depth in range(1, cfg.max_depth + 1):
            for ast in self._enumerate_programs(depth):
                nodes_explored += 1

                try:
                    program = make_program(ast)
                    results = evaluate_on_pairs(program, task.train)

                    if results["all_exact_match"]:
                        return SynthesisResult(
                            success=True,
                            program=ast,
                            program_source=ast_to_source(ast),
                            score=100.0,
                            nodes_explored=nodes_explored,
                        )
                except Exception:
                    continue

        return SynthesisResult(
            success=False,
            nodes_explored=nodes_explored,
            error="Enumeration exhausted",
        )

    def _enumerate_programs(self, depth: int) -> List[ASTNode]:
        """Enumerate all programs of given depth."""
        if depth == 1:
            # Base: single primitives
            programs = [PrimitiveNode("identity")]
            for name in ["crop_to_content", "rotate90", "reflect_h", "reflect_v", "transpose"]:
                if name in PRIMITIVES:
                    programs.append(PrimitiveNode(name))
            # Rotations with args
            for n in [1, 2, 3]:
                programs.append(PrimitiveNode("rotate90", [LiteralNode(n)]))
            return programs

        else:
            # Compose programs of smaller depths
            programs = []
            smaller = self._enumerate_programs(depth - 1)
            base_prims = [
                PrimitiveNode("identity"),
                PrimitiveNode("crop_to_content"),
                PrimitiveNode("reflect_h"),
                PrimitiveNode("reflect_v"),
                PrimitiveNode("transpose"),
            ]
            for n in [1, 2, 3]:
                base_prims.append(PrimitiveNode("rotate90", [LiteralNode(n)]))

            for prog in smaller:
                for prim in base_prims:
                    programs.append(ComposeNode([prog, prim]))
                    programs.append(ComposeNode([prim, prog]))

            return programs
