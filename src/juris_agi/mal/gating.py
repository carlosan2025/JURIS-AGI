"""
Gating mechanism for controlling information flow between experts.

Decides when to use memory/macros vs. fresh synthesis.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional

from ..core.types import ARCTask
from .retrieval import RetrievalResult


class GatingMode(Enum):
    """Modes for the gating mechanism."""
    USE_MEMORY = auto()      # Use retrieved solution directly
    ADAPT_MEMORY = auto()    # Adapt retrieved solution
    FRESH_SYNTHESIS = auto() # Start fresh synthesis
    HYBRID = auto()          # Combine memory and synthesis


@dataclass
class GatingDecision:
    """Decision from the gating mechanism."""
    mode: GatingMode
    confidence: float
    retrieved_solutions: List[RetrievalResult]
    suggested_primitives: List[str]
    rationale: str


class GatingMechanism:
    """
    Decides how to use memory and synthesis.

    Controls the flow of information between:
    - Memory (retrieved solutions)
    - Macros (learned patterns)
    - Fresh synthesis
    """

    def __init__(
        self,
        memory_threshold: float = 0.8,
        adapt_threshold: float = 0.5,
    ):
        """
        Initialize gating mechanism.

        Args:
            memory_threshold: Similarity above which to use memory directly
            adapt_threshold: Similarity above which to adapt memory
        """
        self.memory_threshold = memory_threshold
        self.adapt_threshold = adapt_threshold

    def decide(
        self,
        task: ARCTask,
        retrieved: List[RetrievalResult],
        task_features: Optional[Dict[str, Any]] = None,
    ) -> GatingDecision:
        """
        Make gating decision for a task.

        Args:
            task: The task to solve
            retrieved: Retrieved similar solutions
            task_features: Optional precomputed features

        Returns:
            GatingDecision with recommended mode
        """
        if not retrieved:
            return GatingDecision(
                mode=GatingMode.FRESH_SYNTHESIS,
                confidence=0.5,
                retrieved_solutions=[],
                suggested_primitives=[],
                rationale="No similar solutions found in memory",
            )

        # Check best match
        best_match = retrieved[0]

        if best_match.similarity >= self.memory_threshold and best_match.memory.success:
            return GatingDecision(
                mode=GatingMode.USE_MEMORY,
                confidence=best_match.similarity,
                retrieved_solutions=[best_match],
                suggested_primitives=self._extract_primitives(best_match),
                rationale=f"High similarity ({best_match.similarity:.2f}) to successful solution",
            )

        if best_match.similarity >= self.adapt_threshold:
            return GatingDecision(
                mode=GatingMode.ADAPT_MEMORY,
                confidence=best_match.similarity * 0.8,
                retrieved_solutions=retrieved[:3],  # Top 3 for adaptation
                suggested_primitives=self._extract_primitives_from_multiple(retrieved[:3]),
                rationale=f"Moderate similarity ({best_match.similarity:.2f}), recommend adaptation",
            )

        # Low similarity but some matches exist
        if retrieved and any(r.memory.success for r in retrieved):
            return GatingDecision(
                mode=GatingMode.HYBRID,
                confidence=0.4,
                retrieved_solutions=retrieved[:3],
                suggested_primitives=self._extract_primitives_from_multiple(retrieved[:3]),
                rationale="Low similarity but some successful solutions found",
            )

        return GatingDecision(
            mode=GatingMode.FRESH_SYNTHESIS,
            confidence=0.5,
            retrieved_solutions=retrieved,
            suggested_primitives=[],
            rationale="No sufficiently similar successful solutions",
        )

    def _extract_primitives(self, result: RetrievalResult) -> List[str]:
        """Extract primitive names from a solution."""
        source = result.memory.program_source
        # Simple extraction - look for known primitive names
        primitives = []
        known = [
            "identity", "crop_to_content", "rotate90", "reflect_h", "reflect_v",
            "transpose", "scale", "tile_h", "tile_v", "recolor",
        ]
        for prim in known:
            if prim in source:
                primitives.append(prim)
        return primitives

    def _extract_primitives_from_multiple(
        self,
        results: List[RetrievalResult],
    ) -> List[str]:
        """Extract primitives from multiple solutions, ranked by frequency."""
        from collections import Counter

        all_prims: List[str] = []
        for r in results:
            all_prims.extend(self._extract_primitives(r))

        # Return most common
        counter = Counter(all_prims)
        return [p for p, _ in counter.most_common(5)]


def create_gating_mechanism(
    memory_threshold: float = 0.8,
    adapt_threshold: float = 0.5,
) -> GatingMechanism:
    """Create a gating mechanism with specified thresholds."""
    return GatingMechanism(
        memory_threshold=memory_threshold,
        adapt_threshold=adapt_threshold,
    )


# ============================================================================
# Macro Acceptance Gating with MDL Analysis
# ============================================================================

@dataclass
class MacroAcceptanceResult:
    """Result of macro acceptance evaluation."""
    accepted: bool
    mdl_gain: float  # Positive = macro reduces description length
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "accepted": self.accepted,
            "mdl_gain": self.mdl_gain,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class MacroGate:
    """
    Gating mechanism for accepting macros into the library.

    Evaluates whether a candidate macro provides MDL benefit
    and should be stored for future use.

    Currently a stub that logs MDL_gain but always returns False.
    """

    def __init__(
        self,
        mdl_threshold: float = 0.0,
        min_frequency: int = 2,
        log_decisions: bool = True,
    ):
        """
        Initialize macro gate.

        Args:
            mdl_threshold: Minimum MDL gain to accept (currently ignored)
            min_frequency: Minimum occurrences before considering
            log_decisions: Whether to log acceptance decisions
        """
        self.mdl_threshold = mdl_threshold
        self.min_frequency = min_frequency
        self.log_decisions = log_decisions
        self._decision_log: List[Dict[str, Any]] = []

    def evaluate_macro(
        self,
        macro_code: str,
        frequency: int,
        mdl_cost: int,
        avg_program_length: float = 10.0,
    ) -> MacroAcceptanceResult:
        """
        Evaluate whether to accept a macro.

        Computes MDL gain as: (frequency * saved_length - macro_cost)
        where saved_length is the reduction in program description when using macro.

        Currently always returns False but logs the MDL analysis.

        Args:
            macro_code: The macro's DSL code
            frequency: How often this pattern appeared
            mdl_cost: MDL cost of defining the macro
            avg_program_length: Average program length for context

        Returns:
            MacroAcceptanceResult with acceptance decision and MDL analysis
        """
        # Estimate MDL gain
        # Savings: each use of macro saves (inline_cost - 1) where 1 is the macro call cost
        inline_cost = self._estimate_inline_cost(macro_code)
        savings_per_use = max(0, inline_cost - 1)
        total_savings = frequency * savings_per_use

        # Cost: storing the macro definition
        definition_cost = mdl_cost

        # Net MDL gain
        mdl_gain = total_savings - definition_cost

        # Confidence based on frequency and gain magnitude
        confidence = min(1.0, frequency / 10.0) * (1.0 if mdl_gain > 0 else 0.5)

        # STUB: Always reject for now, but log the decision
        accepted = False  # TODO: Enable once MDL threshold is tuned
        reason = "Stub: macro acceptance disabled (logging MDL_gain for analysis)"

        if mdl_gain < self.mdl_threshold:
            reason = f"MDL gain ({mdl_gain:.2f}) below threshold ({self.mdl_threshold})"
        elif frequency < self.min_frequency:
            reason = f"Frequency ({frequency}) below minimum ({self.min_frequency})"

        result = MacroAcceptanceResult(
            accepted=accepted,
            mdl_gain=mdl_gain,
            confidence=confidence,
            reason=reason,
        )

        # Log the decision
        if self.log_decisions:
            self._log_decision(macro_code, result, frequency, mdl_cost)

        return result

    def _estimate_inline_cost(self, code: str) -> int:
        """Estimate MDL cost of inlining the macro code."""
        cost = 1

        # Count compositions
        cost += code.count(" >> ")

        # Count function calls
        cost += code.count("(")

        # Count arguments
        cost += code.count(",")

        return cost

    def _log_decision(
        self,
        macro_code: str,
        result: MacroAcceptanceResult,
        frequency: int,
        mdl_cost: int,
    ) -> None:
        """Log a macro acceptance decision."""
        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "macro_code": macro_code[:100],  # Truncate long code
            "frequency": frequency,
            "mdl_cost": mdl_cost,
            **result.to_dict(),
        }
        self._decision_log.append(entry)

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get all logged decisions."""
        return self._decision_log.copy()

    def get_mdl_statistics(self) -> Dict[str, Any]:
        """Get statistics about MDL gains across all evaluated macros."""
        if not self._decision_log:
            return {
                "total_evaluated": 0,
                "avg_mdl_gain": 0.0,
                "max_mdl_gain": 0.0,
                "min_mdl_gain": 0.0,
                "positive_gain_count": 0,
            }

        gains = [d["mdl_gain"] for d in self._decision_log]
        return {
            "total_evaluated": len(gains),
            "avg_mdl_gain": sum(gains) / len(gains),
            "max_mdl_gain": max(gains),
            "min_mdl_gain": min(gains),
            "positive_gain_count": sum(1 for g in gains if g > 0),
        }

    def clear_log(self) -> None:
        """Clear the decision log."""
        self._decision_log.clear()


def accept_macro(
    macro_code: str,
    frequency: int,
    mdl_cost: int,
    gate: Optional[MacroGate] = None,
) -> MacroAcceptanceResult:
    """
    Convenience function to evaluate macro acceptance.

    Stub implementation that logs MDL_gain but always returns False.

    Args:
        macro_code: The macro's DSL code
        frequency: How often this pattern appeared
        mdl_cost: MDL cost of defining the macro
        gate: Optional pre-configured gate

    Returns:
        MacroAcceptanceResult (currently always not accepted)
    """
    if gate is None:
        gate = MacroGate()

    return gate.evaluate_macro(macro_code, frequency, mdl_cost)
