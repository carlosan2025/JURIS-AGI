"""
Meta-Controller: routes tasks and coordinates experts.

The meta-controller orchestrates:
- CRE (Certified Reasoning Expert)
- WME (World Model Expert)
- MAL (Memory & Abstraction Library)

JURISDICTION: Symbolic critic has veto power. WME/MAL are advisory only.

Regime Detection:
- ARC_DISCRETE: Standard ARC-style tasks with clear input-output patterns
- UNCERTAIN: Tasks with high epistemic/aleatoric uncertainty

Budget Allocation:
- Priors phase: WME analysis
- Synthesis phase: CRE beam search
- Refinement phase: Near-miss refinement
- Robustness phase: WME robustness checking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto

from ..core.types import ARCTask, Grid, AuditTrace, SolverResult, SymbolicDiffSummary
from ..core.trace import SolveTrace, TraceWriter
from ..core.metrics import evaluate_on_pairs

from ..dsl.ast import ASTNode, PrimitiveNode
from ..dsl.interpreter import make_program
from ..dsl.prettyprint import ast_to_source

from ..cre.synthesizer import BeamSearchSynthesizer, SynthesisConfig, SynthesisResult
import time
from ..cre.critic_symbolic import SymbolicCritic, CriticResult
from ..cre.refinement import RefinementEngine

from ..wme.world_model import HeuristicWorldModel, WorldModelState
from ..wme.robustness import RobustnessChecker

from ..mal.retrieval import InMemoryStore, create_memory_from_solution
from ..mal.macro_induction import MacroLibrary
from ..mal.gating import GatingMechanism, GatingMode


# ============================================================================
# Regime Detection
# ============================================================================

class TaskRegime(Enum):
    """Task regime classification."""
    ARC_DISCRETE = auto()  # Standard ARC with clear patterns
    UNCERTAIN = auto()  # High uncertainty, may require more exploration


@dataclass
class RegimeDecision:
    """Result of regime determination."""
    regime: TaskRegime
    confidence: float
    features: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.name,
            "confidence": self.confidence,
            "features": self.features,
            "rationale": self.rationale,
        }


def determine_regime(task: ARCTask) -> RegimeDecision:
    """
    Determine the task regime based on task features.

    ARC_DISCRETE: Tasks that fit the standard ARC paradigm
    - Consistent dimension relationships across examples
    - Clear palette patterns
    - Low variance in features

    UNCERTAIN: Tasks with high uncertainty
    - Inconsistent patterns across examples
    - Many possible interpretations
    - High feature variance

    Args:
        task: The ARC task to classify

    Returns:
        RegimeDecision with regime classification
    """
    features: Dict[str, Any] = {}

    if not task.train:
        return RegimeDecision(
            regime=TaskRegime.UNCERTAIN,
            confidence=0.3,
            features={},
            rationale="No training examples",
        )

    # Feature 1: Dimension consistency
    dim_ratios = []
    for pair in task.train:
        if pair.input.height > 0 and pair.input.width > 0:
            h_ratio = pair.output.height / pair.input.height
            w_ratio = pair.output.width / pair.input.width
            dim_ratios.append((h_ratio, w_ratio))

    if dim_ratios:
        h_ratios = [r[0] for r in dim_ratios]
        w_ratios = [r[1] for r in dim_ratios]

        # Variance in dimension ratios
        h_variance = _compute_variance(h_ratios)
        w_variance = _compute_variance(w_ratios)
        features["dim_ratio_variance"] = (h_variance + w_variance) / 2

        # All same dimensions?
        features["same_dims"] = all(
            pair.input.shape == pair.output.shape for pair in task.train
        )
    else:
        features["dim_ratio_variance"] = 1.0
        features["same_dims"] = False

    # Feature 2: Palette consistency
    input_palettes = [pair.input.palette for pair in task.train]
    output_palettes = [pair.output.palette for pair in task.train]

    # Do all palettes match?
    features["consistent_input_palette"] = len(set(map(frozenset, input_palettes))) == 1
    features["consistent_output_palette"] = len(set(map(frozenset, output_palettes))) == 1

    # Palette preserved?
    features["palette_preserved"] = all(
        pair.input.palette == pair.output.palette for pair in task.train
    )

    # Feature 3: Number of training examples
    features["num_train"] = len(task.train)
    features["few_examples"] = len(task.train) < 3

    # Feature 4: Grid complexity
    avg_input_cells = sum(p.input.height * p.input.width for p in task.train) / len(task.train)
    avg_output_cells = sum(p.output.height * p.output.width for p in task.train) / len(task.train)
    features["avg_input_cells"] = avg_input_cells
    features["avg_output_cells"] = avg_output_cells
    features["high_complexity"] = avg_input_cells > 200 or avg_output_cells > 200

    # Compute uncertainty score
    uncertainty_score = 0.0

    # High variance in dimensions suggests uncertainty
    if features["dim_ratio_variance"] > 0.1:
        uncertainty_score += 0.3

    # Inconsistent palettes suggest uncertainty
    if not features["consistent_input_palette"]:
        uncertainty_score += 0.15
    if not features["consistent_output_palette"]:
        uncertainty_score += 0.15

    # Few examples increases uncertainty
    if features["few_examples"]:
        uncertainty_score += 0.2

    # High complexity increases uncertainty
    if features["high_complexity"]:
        uncertainty_score += 0.2

    features["uncertainty_score"] = uncertainty_score

    # Determine regime
    if uncertainty_score > 0.5:
        regime = TaskRegime.UNCERTAIN
        rationale = "High uncertainty due to: "
        reasons = []
        if features["dim_ratio_variance"] > 0.1:
            reasons.append("inconsistent dimensions")
        if not features["consistent_input_palette"]:
            reasons.append("varying input palettes")
        if features["few_examples"]:
            reasons.append("few training examples")
        if features["high_complexity"]:
            reasons.append("high grid complexity")
        rationale += ", ".join(reasons) if reasons else "multiple factors"
    else:
        regime = TaskRegime.ARC_DISCRETE
        rationale = "Standard ARC task with consistent patterns"

    confidence = 1.0 - uncertainty_score if regime == TaskRegime.ARC_DISCRETE else uncertainty_score

    return RegimeDecision(
        regime=regime,
        confidence=confidence,
        features=features,
        rationale=rationale,
    )


def _compute_variance(values: List[float]) -> float:
    """Compute variance of a list of values."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance


@dataclass
class ControllerConfig:
    """Configuration for the meta-controller."""
    # Synthesis settings
    max_synthesis_depth: int = 4
    beam_width: int = 50
    max_synthesis_iterations: int = 1000

    # Refinement settings
    max_refinement_iterations: int = 20
    enable_refinement: bool = True

    # Expert settings
    enable_wme: bool = True
    enable_mal: bool = True

    # Robustness settings
    compute_robustness: bool = True
    min_robustness_score: float = 0.3

    # Budget settings
    max_time_seconds: float = 60.0


@dataclass
class SolveAttempt:
    """Record of a single solve attempt."""
    method: str
    success: bool
    program: Optional[ASTNode] = None
    program_source: Optional[str] = None
    iterations: int = 0
    error: Optional[str] = None


class MetaController:
    """
    Meta-controller that orchestrates all experts.

    Routes tasks, manages budgets, and ensures jurisdiction.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()

        # Initialize experts
        self.synthesizer = BeamSearchSynthesizer(SynthesisConfig(
            max_depth=self.config.max_synthesis_depth,
            beam_width=self.config.beam_width,
            max_iterations=self.config.max_synthesis_iterations,
        ))
        self.critic = SymbolicCritic()
        self.refinement_engine = RefinementEngine(
            max_iterations=self.config.max_refinement_iterations,
        )

        # Advisory experts (NO JURISDICTION)
        self.world_model = HeuristicWorldModel()
        self.robustness_checker = RobustnessChecker()

        # Memory & Abstraction
        self.memory_store = InMemoryStore()
        self.macro_library = MacroLibrary()
        self.gating = GatingMechanism()

    def solve(self, task: ARCTask) -> SolverResult:
        """
        Solve an ARC task.

        Orchestrates experts and returns certified solution.
        """
        start_time = time.time()
        trace = SolveTrace.start(task.task_id)
        attempts: List[SolveAttempt] = []
        synth_result: Optional[SynthesisResult] = None

        trace.log("start", "controller", task_id=task.task_id)

        # Step 1: Analyze task with WME (advisory)
        wme_state = None
        if self.config.enable_wme:
            trace.log("wme_analysis", "wme")
            wme_state = self.world_model.analyze_task(task)
            trace.log("wme_complete", "wme", hypotheses=len(wme_state.hypotheses))

        # Step 2: Check memory for similar solutions
        gating_decision = None
        if self.config.enable_mal:
            trace.log("memory_retrieval", "mal")
            retrieved = self.memory_store.retrieve(task, top_k=5)
            gating_decision = self.gating.decide(task, retrieved)
            trace.log(
                "gating_decision",
                "mal",
                mode=gating_decision.mode.name,
                confidence=gating_decision.confidence,
            )

            # Try memory-based solutions first
            if gating_decision.mode == GatingMode.USE_MEMORY:
                for result in gating_decision.retrieved_solutions:
                    mem_program = result.memory.program
                    critique = self.critic.evaluate(mem_program, task)

                    if critique.is_certified:
                        trace.log("memory_hit", "mal", success=True)
                        return self._create_result(
                            task, mem_program, critique, trace, attempts, "memory",
                            start_time=start_time,
                        )

        # Step 3: Run synthesis (CRE)
        trace.log("synthesis_start", "cre")

        synthesis_config = SynthesisConfig(
            max_depth=self.config.max_synthesis_depth,
            beam_width=self.config.beam_width,
            max_iterations=self.config.max_synthesis_iterations,
        )

        # If WME has strong hypotheses, guide synthesis
        if wme_state and wme_state.confidence > 0.7:
            # Could add hypothesis-guided synthesis here
            pass

        synth_result = self.synthesizer.synthesize(task, synthesis_config)

        trace.log(
            "synthesis_complete",
            "cre",
            success=synth_result.success,
            iterations=synth_result.iterations,
            nodes_explored=synth_result.nodes_explored,
        )

        attempts.append(SolveAttempt(
            method="beam_search",
            success=synth_result.success,
            program=synth_result.program,
            program_source=synth_result.program_source,
            iterations=synth_result.iterations,
            error=synth_result.error,
        ))

        if synth_result.success and synth_result.program:
            # Verify with symbolic critic (JURISDICTION)
            critique = self.critic.evaluate(synth_result.program, task)

            if critique.is_certified:
                return self._create_result(
                    task, synth_result.program, critique, trace, attempts, "synthesis",
                    synth_result=synth_result, start_time=start_time,
                )

            # Not certified - try refinement
            if self.config.enable_refinement:
                trace.log("refinement_start", "cre")
                refined = self.refinement_engine.refine(
                    synth_result.program, task, critique
                )
                trace.log(
                    "refinement_complete",
                    "cre",
                    success=refined.success,
                    improved=refined.improved,
                    iterations=refined.iterations,
                )

                if refined.success and refined.refined_ast:
                    refined_critique = self.critic.evaluate(refined.refined_ast, task)
                    if refined_critique.is_certified:
                        return self._create_result(
                            task, refined.refined_ast, refined_critique, trace, attempts, "refinement",
                            synth_result=synth_result, start_time=start_time,
                            refinement_edits=[e.describe() for e in refined.edits_applied],
                        )

        # Step 4: If synthesis didn't find solution, try near-miss refinement
        if synth_result.program and not synth_result.success:
            if self.config.enable_refinement:
                critique = self.critic.evaluate(synth_result.program, task)
                refined = self.refinement_engine.refine(synth_result.program, task, critique)

                if refined.success and refined.refined_ast:
                    refined_critique = self.critic.evaluate(refined.refined_ast, task)
                    if refined_critique.is_certified:
                        return self._create_result(
                            task, refined.refined_ast, refined_critique, trace, attempts, "refinement",
                            synth_result=synth_result, start_time=start_time,
                            refinement_edits=[e.describe() for e in refined.edits_applied],
                        )

        # Step 5: Failed - return best effort
        trace.log("solve_failed", "controller")
        trace.finalize(success=False)

        best_program = synth_result.program if synth_result and synth_result.program else PrimitiveNode("identity")
        best_source = ast_to_source(best_program)
        runtime = time.time() - start_time

        # Get critique for diffs
        best_critique = self.critic.evaluate(best_program, task)
        symbolic_diffs = self._build_diff_summaries(best_critique)

        return SolverResult(
            task_id=task.task_id,
            success=False,
            predictions=self._generate_predictions(best_program, task),
            audit_trace=AuditTrace(
                task_id=task.task_id,
                program_source=best_source,
                program_ast=best_program,
                constraints_violated=["exact_match"],
                synthesis_iterations=synth_result.iterations if synth_result else 0,
                search_nodes_explored=synth_result.nodes_explored if synth_result else 0,
                # Extended fields
                program_depth=best_program.depth() if hasattr(best_program, 'depth') else 0,
                program_size=best_program.size() if hasattr(best_program, 'size') else 0,
                expansions_generated=synth_result.nodes_explored if synth_result else 0,
                candidates_pruned=synth_result.candidates_pruned if synth_result else 0,
                runtime_seconds=runtime,
                symbolic_diffs=symbolic_diffs,
                near_miss_count=len(synth_result.near_misses) if synth_result else 0,
            ),
            error_message="Could not find certified solution",
        )

    def _create_result(
        self,
        task: ARCTask,
        program: ASTNode,
        critique: CriticResult,
        trace: SolveTrace,
        attempts: List[SolveAttempt],
        method: str,
        synth_result: Optional[SynthesisResult] = None,
        start_time: Optional[float] = None,
        refinement_edits: Optional[List[str]] = None,
    ) -> SolverResult:
        """Create a successful solver result."""
        program_source = ast_to_source(program)

        # Compute robustness
        robustness_score = 0.0
        if self.config.compute_robustness:
            robustness_result = self.robustness_checker.check_robustness(program, task)
            robustness_score = robustness_result.overall_score

        # Store in memory
        if self.config.enable_mal:
            memory = create_memory_from_solution(
                task, program, success=True, robustness_score=robustness_score
            )
            self.memory_store.store(memory)
            self.macro_library.add_program(program, task_context=task.task_id)

        # Generate predictions
        predictions = self._generate_predictions(program, task)

        # Build symbolic diff summaries
        symbolic_diffs = self._build_diff_summaries(critique)

        # Compute runtime
        runtime = time.time() - start_time if start_time else 0.0

        trace.final_program = program_source
        trace.finalize(success=True, program=program_source)

        return SolverResult(
            task_id=task.task_id,
            success=True,
            predictions=predictions,
            audit_trace=AuditTrace(
                task_id=task.task_id,
                program_source=program_source,
                program_ast=program,
                constraints_satisfied=critique.invariants_satisfied,
                constraints_violated=critique.invariants_violated,
                robustness_score=robustness_score,
                # Extended fields
                program_depth=program.depth() if hasattr(program, 'depth') else 0,
                program_size=program.size() if hasattr(program, 'size') else 0,
                synthesis_iterations=synth_result.iterations if synth_result else 0,
                search_nodes_explored=synth_result.nodes_explored if synth_result else 0,
                expansions_generated=synth_result.nodes_explored if synth_result else 0,
                candidates_pruned=synth_result.candidates_pruned if synth_result else 0,
                runtime_seconds=runtime,
                symbolic_diffs=symbolic_diffs,
                near_miss_count=len(synth_result.near_misses) if synth_result else 0,
                refinement_applied=method == "refinement",
                refinement_improved=method == "refinement",
                refinement_edits=refinement_edits or [],
            ),
        )

    def _build_diff_summaries(self, critique: CriticResult) -> List[SymbolicDiffSummary]:
        """Build symbolic diff summaries from critique."""
        summaries = []
        for i, diff in enumerate(critique.diffs):
            pair_result = critique.pair_results[i] if i < len(critique.pair_results) else {}
            summary = SymbolicDiffSummary(
                pair_index=i,
                pair_type=pair_result.get("pair_type", "train"),
                dimension_match=diff.dimension_match,
                exact_match=diff.exact_match,
                pixel_accuracy=diff.pixel_accuracy,
                num_errors=diff.num_errors,
                severity=diff.severity,
                extra_colors=list(diff.extra_colors),
                missing_colors=list(diff.missing_colors),
            )
            summaries.append(summary)
        return summaries

    def _generate_predictions(
        self,
        program: ASTNode,
        task: ARCTask,
    ) -> List[Grid]:
        """Generate predictions for test inputs."""
        predictions = []
        program_fn = make_program(program)

        for pair in task.test:
            try:
                output = program_fn(pair.input)
                predictions.append(output)
            except Exception:
                # Fallback to empty grid
                predictions.append(Grid.zeros(1, 1))

        return predictions

    def solve_batch(
        self,
        tasks: List[ARCTask],
    ) -> List[SolverResult]:
        """Solve multiple tasks."""
        return [self.solve(task) for task in tasks]
