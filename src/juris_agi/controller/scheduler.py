"""
Expert scheduler for budget management.

Manages computational budgets across experts using uncertainty estimates.

Phase-based allocation:
- PRIORS: WME analysis, MAL retrieval
- SYNTHESIS: Beam search program generation
- REFINEMENT: Near-miss refinement
- ROBUSTNESS: Counterfactual testing
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple
import time


class ExpertType(Enum):
    """Types of experts in the system."""
    CRE = auto()  # Certified Reasoning Expert
    WME = auto()  # World Model Expert
    MAL = auto()  # Memory & Abstraction Library
    SYNTHESIZER = auto()
    CRITIC = auto()
    REFINEMENT = auto()


class SolvePhase(Enum):
    """Phases of the solve process."""
    PRIORS = auto()      # WME analysis, hypothesis generation
    SYNTHESIS = auto()   # Main program synthesis
    REFINEMENT = auto()  # Near-miss refinement
    ROBUSTNESS = auto()  # Robustness checking


@dataclass
class Budget:
    """Budget allocation for an expert."""
    expert: ExpertType
    time_limit: float  # seconds
    iteration_limit: int
    priority: float = 1.0  # Higher = more important
    time_used: float = 0.0
    iterations_used: int = 0

    @property
    def time_remaining(self) -> float:
        return max(0, self.time_limit - self.time_used)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.iteration_limit - self.iterations_used)

    @property
    def exhausted(self) -> bool:
        return self.time_remaining <= 0 or self.iterations_remaining <= 0


@dataclass
class ScheduleDecision:
    """Decision about which expert to run next."""
    expert: ExpertType
    budget: Budget
    rationale: str
    uncertainty_estimate: float


@dataclass
class UncertaintyEstimate:
    """Estimate of epistemic vs aleatoric uncertainty."""
    epistemic: float  # Reducible through more computation
    aleatoric: float  # Irreducible randomness
    total: float

    @classmethod
    def compute(
        cls,
        num_candidates: int,
        best_score: float,
        variance: float,
    ) -> "UncertaintyEstimate":
        """
        Compute uncertainty estimate.

        Args:
            num_candidates: Number of candidate solutions found
            best_score: Best score so far (0-1)
            variance: Variance in scores
        """
        # Epistemic: reducible by more search
        # Higher when few candidates, low score
        epistemic = (1 - best_score) * max(0, 1 - num_candidates / 100)

        # Aleatoric: inherent randomness
        # Higher variance suggests more aleatoric uncertainty
        aleatoric = min(1.0, variance)

        total = min(1.0, epistemic + aleatoric)

        return cls(epistemic=epistemic, aleatoric=aleatoric, total=total)


class ExpertScheduler:
    """
    Schedules expert execution based on uncertainty.

    Uses epistemic vs aleatoric uncertainty to decide:
    - High epistemic: more synthesis/search might help
    - High aleatoric: try different approaches
    """

    def __init__(
        self,
        total_time_budget: float = 60.0,
        total_iteration_budget: int = 10000,
    ):
        self.total_time_budget = total_time_budget
        self.total_iteration_budget = total_iteration_budget

        # Default budget allocation
        self.budgets: Dict[ExpertType, Budget] = {
            ExpertType.CRE: Budget(
                expert=ExpertType.CRE,
                time_limit=total_time_budget * 0.6,
                iteration_limit=int(total_iteration_budget * 0.7),
                priority=1.0,
            ),
            ExpertType.WME: Budget(
                expert=ExpertType.WME,
                time_limit=total_time_budget * 0.1,
                iteration_limit=int(total_iteration_budget * 0.05),
                priority=0.8,
            ),
            ExpertType.MAL: Budget(
                expert=ExpertType.MAL,
                time_limit=total_time_budget * 0.1,
                iteration_limit=int(total_iteration_budget * 0.05),
                priority=0.7,
            ),
            ExpertType.SYNTHESIZER: Budget(
                expert=ExpertType.SYNTHESIZER,
                time_limit=total_time_budget * 0.4,
                iteration_limit=int(total_iteration_budget * 0.5),
                priority=1.0,
            ),
            ExpertType.REFINEMENT: Budget(
                expert=ExpertType.REFINEMENT,
                time_limit=total_time_budget * 0.2,
                iteration_limit=int(total_iteration_budget * 0.2),
                priority=0.9,
            ),
        }

        self.start_time = time.time()
        self.history: List[Dict[str, Any]] = []

    def get_next_expert(
        self,
        current_state: Dict[str, Any],
    ) -> ScheduleDecision:
        """
        Decide which expert to run next.

        Args:
            current_state: Current solve state including:
                - num_candidates: Number of candidates found
                - best_score: Best score so far
                - variance: Score variance
                - experts_tried: Set of experts already tried

        Returns:
            ScheduleDecision with recommended expert
        """
        # Compute uncertainty
        uncertainty = UncertaintyEstimate.compute(
            num_candidates=current_state.get("num_candidates", 0),
            best_score=current_state.get("best_score", 0.0),
            variance=current_state.get("variance", 0.5),
        )

        experts_tried = current_state.get("experts_tried", set())

        # Decision logic based on uncertainty
        if uncertainty.epistemic > 0.5:
            # High epistemic uncertainty - more search might help
            if not self.budgets[ExpertType.SYNTHESIZER].exhausted:
                return ScheduleDecision(
                    expert=ExpertType.SYNTHESIZER,
                    budget=self.budgets[ExpertType.SYNTHESIZER],
                    rationale="High epistemic uncertainty - continue synthesis",
                    uncertainty_estimate=uncertainty.epistemic,
                )

        if uncertainty.aleatoric > 0.5:
            # High aleatoric - try different approaches
            if ExpertType.WME not in experts_tried and not self.budgets[ExpertType.WME].exhausted:
                return ScheduleDecision(
                    expert=ExpertType.WME,
                    budget=self.budgets[ExpertType.WME],
                    rationale="High aleatoric uncertainty - consult world model",
                    uncertainty_estimate=uncertainty.aleatoric,
                )

            if ExpertType.MAL not in experts_tried and not self.budgets[ExpertType.MAL].exhausted:
                return ScheduleDecision(
                    expert=ExpertType.MAL,
                    budget=self.budgets[ExpertType.MAL],
                    rationale="High aleatoric uncertainty - check memory",
                    uncertainty_estimate=uncertainty.aleatoric,
                )

        # Default: prioritize by remaining budget
        available = [
            (expert, budget)
            for expert, budget in self.budgets.items()
            if not budget.exhausted
        ]

        if not available:
            # All budgets exhausted
            return ScheduleDecision(
                expert=ExpertType.CRE,
                budget=self.budgets[ExpertType.CRE],
                rationale="All budgets exhausted",
                uncertainty_estimate=uncertainty.total,
            )

        # Sort by priority * remaining budget
        available.sort(
            key=lambda x: x[1].priority * x[1].time_remaining,
            reverse=True,
        )

        best_expert, best_budget = available[0]
        return ScheduleDecision(
            expert=best_expert,
            budget=best_budget,
            rationale=f"Best available expert: {best_expert.name}",
            uncertainty_estimate=uncertainty.total,
        )

    def record_usage(
        self,
        expert: ExpertType,
        time_used: float,
        iterations_used: int,
    ) -> None:
        """Record resource usage by an expert."""
        if expert in self.budgets:
            self.budgets[expert].time_used += time_used
            self.budgets[expert].iterations_used += iterations_used

        self.history.append({
            "expert": expert.name,
            "time_used": time_used,
            "iterations_used": iterations_used,
            "timestamp": time.time() - self.start_time,
        })

    def get_remaining_time(self) -> float:
        """Get total remaining time budget."""
        elapsed = time.time() - self.start_time
        return max(0, self.total_time_budget - elapsed)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of budget usage."""
        return {
            "total_time_budget": self.total_time_budget,
            "elapsed_time": time.time() - self.start_time,
            "remaining_time": self.get_remaining_time(),
            "budgets": {
                expert.name: {
                    "time_limit": budget.time_limit,
                    "time_used": budget.time_used,
                    "iterations_used": budget.iterations_used,
                    "exhausted": budget.exhausted,
                }
                for expert, budget in self.budgets.items()
            },
            "history_length": len(self.history),
        }


# ============================================================================
# Phase-Based Budget Allocation
# ============================================================================

@dataclass
class PhaseBudget:
    """Budget allocation for a solve phase."""
    phase: SolvePhase
    time_limit: float  # seconds
    iteration_limit: int
    time_used: float = 0.0
    iterations_used: int = 0

    @property
    def time_remaining(self) -> float:
        return max(0, self.time_limit - self.time_used)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.iteration_limit - self.iterations_used)

    @property
    def exhausted(self) -> bool:
        return self.time_remaining <= 0 or self.iterations_remaining <= 0

    @property
    def utilization(self) -> float:
        """Fraction of budget used."""
        if self.time_limit <= 0:
            return 1.0
        return self.time_used / self.time_limit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.name,
            "time_limit": self.time_limit,
            "time_used": self.time_used,
            "time_remaining": self.time_remaining,
            "iteration_limit": self.iteration_limit,
            "iterations_used": self.iterations_used,
            "iterations_remaining": self.iterations_remaining,
            "exhausted": self.exhausted,
            "utilization": self.utilization,
        }


@dataclass
class PhaseAllocation:
    """Complete phase budget allocation plan."""
    total_time: float
    total_iterations: int
    phase_budgets: Dict[SolvePhase, PhaseBudget]
    rationale: str

    def get_phase_budget(self, phase: SolvePhase) -> PhaseBudget:
        """Get budget for a specific phase."""
        return self.phase_budgets.get(phase, PhaseBudget(
            phase=phase, time_limit=0, iteration_limit=0
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time": self.total_time,
            "total_iterations": self.total_iterations,
            "rationale": self.rationale,
            "phases": {
                phase.name: budget.to_dict()
                for phase, budget in self.phase_budgets.items()
            },
        }


def allocate_budget(
    total_time: float,
    total_iterations: int,
    regime: Optional[str] = None,
    task_features: Optional[Dict[str, Any]] = None,
) -> PhaseAllocation:
    """
    Allocate budget across solve phases.

    Default allocation:
    - PRIORS: 10% (WME analysis, MAL retrieval)
    - SYNTHESIS: 60% (main beam search)
    - REFINEMENT: 20% (near-miss refinement)
    - ROBUSTNESS: 10% (counterfactual testing)

    Adjustments based on regime:
    - UNCERTAIN regime: More time to PRIORS and REFINEMENT
    - ARC_DISCRETE regime: More time to SYNTHESIS

    Args:
        total_time: Total time budget in seconds
        total_iterations: Total iteration budget
        regime: Task regime (ARC_DISCRETE or UNCERTAIN)
        task_features: Additional task features for tuning

    Returns:
        PhaseAllocation with per-phase budgets
    """
    # Default allocation percentages
    priors_pct = 0.10
    synthesis_pct = 0.60
    refinement_pct = 0.20
    robustness_pct = 0.10

    rationale = "Default allocation"

    # Adjust based on regime
    if regime == "UNCERTAIN":
        # More exploration needed
        priors_pct = 0.15
        synthesis_pct = 0.45
        refinement_pct = 0.30
        robustness_pct = 0.10
        rationale = "Uncertain regime: increased priors and refinement"

    elif regime == "ARC_DISCRETE":
        # Standard ARC task - more synthesis
        priors_pct = 0.05
        synthesis_pct = 0.70
        refinement_pct = 0.15
        robustness_pct = 0.10
        rationale = "ARC_DISCRETE regime: increased synthesis"

    # Adjust based on task features
    if task_features:
        if task_features.get("high_complexity", False):
            # Complex tasks need more synthesis
            synthesis_pct += 0.10
            priors_pct -= 0.05
            robustness_pct -= 0.05
            rationale += " + high complexity adjustment"

        if task_features.get("few_examples", False):
            # Few examples - more priors analysis
            priors_pct += 0.05
            synthesis_pct -= 0.05
            rationale += " + few examples adjustment"

    # Create phase budgets
    phase_budgets = {
        SolvePhase.PRIORS: PhaseBudget(
            phase=SolvePhase.PRIORS,
            time_limit=total_time * priors_pct,
            iteration_limit=int(total_iterations * priors_pct),
        ),
        SolvePhase.SYNTHESIS: PhaseBudget(
            phase=SolvePhase.SYNTHESIS,
            time_limit=total_time * synthesis_pct,
            iteration_limit=int(total_iterations * synthesis_pct),
        ),
        SolvePhase.REFINEMENT: PhaseBudget(
            phase=SolvePhase.REFINEMENT,
            time_limit=total_time * refinement_pct,
            iteration_limit=int(total_iterations * refinement_pct),
        ),
        SolvePhase.ROBUSTNESS: PhaseBudget(
            phase=SolvePhase.ROBUSTNESS,
            time_limit=total_time * robustness_pct,
            iteration_limit=int(total_iterations * robustness_pct),
        ),
    }

    return PhaseAllocation(
        total_time=total_time,
        total_iterations=total_iterations,
        phase_budgets=phase_budgets,
        rationale=rationale,
    )


class PhaseScheduler:
    """
    Phase-based scheduler for solve process.

    Manages budget allocation and tracking across solve phases.
    """

    def __init__(
        self,
        total_time: float = 60.0,
        total_iterations: int = 10000,
        regime: Optional[str] = None,
        task_features: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize phase scheduler.

        Args:
            total_time: Total time budget in seconds
            total_iterations: Total iteration budget
            regime: Task regime for allocation tuning
            task_features: Task features for allocation tuning
        """
        self.allocation = allocate_budget(
            total_time=total_time,
            total_iterations=total_iterations,
            regime=regime,
            task_features=task_features,
        )

        self.start_time = time.time()
        self.current_phase: Optional[SolvePhase] = None
        self.phase_start_time: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    def start_phase(self, phase: SolvePhase) -> PhaseBudget:
        """
        Start a solve phase.

        Args:
            phase: The phase to start

        Returns:
            Budget for this phase
        """
        # End previous phase if any
        if self.current_phase is not None:
            self.end_phase()

        self.current_phase = phase
        self.phase_start_time = time.time()

        self.history.append({
            "event": "phase_start",
            "phase": phase.name,
            "timestamp": time.time() - self.start_time,
        })

        return self.allocation.get_phase_budget(phase)

    def end_phase(
        self,
        iterations_used: int = 0,
        success: bool = False,
    ) -> None:
        """
        End the current phase.

        Args:
            iterations_used: Iterations consumed in this phase
            success: Whether phase achieved its goal
        """
        if self.current_phase is None:
            return

        phase_budget = self.allocation.get_phase_budget(self.current_phase)
        time_used = time.time() - (self.phase_start_time or time.time())

        phase_budget.time_used = time_used
        phase_budget.iterations_used = iterations_used

        self.history.append({
            "event": "phase_end",
            "phase": self.current_phase.name,
            "timestamp": time.time() - self.start_time,
            "time_used": time_used,
            "iterations_used": iterations_used,
            "success": success,
        })

        self.current_phase = None
        self.phase_start_time = None

    def get_remaining_time(self, phase: Optional[SolvePhase] = None) -> float:
        """
        Get remaining time for a phase or overall.

        Args:
            phase: Specific phase (or None for overall)

        Returns:
            Remaining time in seconds
        """
        if phase is not None:
            return self.allocation.get_phase_budget(phase).time_remaining

        # Overall remaining
        elapsed = time.time() - self.start_time
        return max(0, self.allocation.total_time - elapsed)

    def is_phase_exhausted(self, phase: SolvePhase) -> bool:
        """Check if a phase's budget is exhausted."""
        return self.allocation.get_phase_budget(phase).exhausted

    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of all phase budgets."""
        return {
            "total_time": self.allocation.total_time,
            "elapsed": time.time() - self.start_time,
            "remaining": self.get_remaining_time(),
            "allocation_rationale": self.allocation.rationale,
            "phases": {
                phase.name: self.allocation.get_phase_budget(phase).to_dict()
                for phase in SolvePhase
            },
            "current_phase": self.current_phase.name if self.current_phase else None,
            "history_length": len(self.history),
        }

    def should_continue(self) -> Tuple[bool, str]:
        """
        Check if solving should continue.

        Returns:
            Tuple of (should_continue, reason)
        """
        # Check overall time
        if self.get_remaining_time() <= 0:
            return False, "Total time budget exhausted"

        # Check if all phases exhausted
        all_exhausted = all(
            self.allocation.get_phase_budget(phase).exhausted
            for phase in SolvePhase
        )
        if all_exhausted:
            return False, "All phase budgets exhausted"

        return True, "Budget available"
