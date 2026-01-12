"""
VC Decision Trace with Counterfactual Analysis.

Stores complete audit trail of decision reasoning including
counterfactual analysis for interpretability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
import json

from ..evidence.schema import EvidenceGraph
from .decision_analysis import (
    DecisionAnalysisResult,
    DecisionOutcome,
    DecisionCriticalClaim,
    DecisionRobustness,
    CounterfactualExplanation,
    DecisionAnalyzer,
)
from .counterfactuals import CounterfactualEvidenceGraph


@dataclass
class VCDecisionTraceEntry:
    """
    A single entry in the decision trace.

    Records an event or step in the decision process.
    """
    timestamp: datetime
    entry_type: str
    """Type: 'analysis', 'counterfactual', 'robustness', 'explanation', etc."""

    content: Dict[str, Any]
    """Entry content."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type,
            "content": self.content,
        }


@dataclass
class VCDecisionTrace:
    """
    Complete audit trace for a VC investment decision.

    Stores all reasoning steps, counterfactual analysis, and explanations.
    """
    company_id: str
    """Company being evaluated."""

    decision: DecisionOutcome
    """Final decision."""

    confidence: float
    """Confidence in the decision."""

    # Analysis results
    critical_claims: List[DecisionCriticalClaim] = field(default_factory=list)
    """Claims identified as decision-critical."""

    robustness: Optional[DecisionRobustness] = None
    """Robustness analysis."""

    counterfactual_explanations: List[CounterfactualExplanation] = field(default_factory=list)
    """Natural language explanations."""

    # Counterfactual details
    counterfactuals_tested: List[CounterfactualEvidenceGraph] = field(default_factory=list)
    """All counterfactuals that were tested."""

    decision_flips: List[Dict[str, Any]] = field(default_factory=list)
    """Details of each decision flip found."""

    # Trace entries
    entries: List[VCDecisionTraceEntry] = field(default_factory=list)
    """Chronological trace entries."""

    # Metadata
    created_at: Optional[datetime] = None
    analyst_id: Optional[str] = None
    model_version: str = "1.0"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def add_entry(
        self,
        entry_type: str,
        content: Dict[str, Any],
    ) -> None:
        """Add an entry to the trace."""
        self.entries.append(VCDecisionTraceEntry(
            timestamp=datetime.utcnow(),
            entry_type=entry_type,
            content=content,
        ))

    def add_counterfactual_result(
        self,
        counterfactual: CounterfactualEvidenceGraph,
        original_decision: DecisionOutcome,
        new_decision: DecisionOutcome,
    ) -> None:
        """Record a counterfactual test result."""
        self.counterfactuals_tested.append(counterfactual)

        if new_decision != original_decision:
            flip_info = {
                "perturbation_summary": counterfactual.perturbation_summary,
                "total_magnitude": counterfactual.total_perturbation_magnitude,
                "original_decision": original_decision.value,
                "new_decision": new_decision.value,
                "perturbations": [p.to_dict() for p in counterfactual.perturbations],
            }
            self.decision_flips.append(flip_info)

            self.add_entry("decision_flip", flip_info)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the trace."""
        return {
            "company_id": self.company_id,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "num_critical_claims": len(self.critical_claims),
            "num_counterfactuals_tested": len(self.counterfactuals_tested),
            "num_flips_found": len(self.decision_flips),
            "robustness_score": self.robustness.overall_score if self.robustness else None,
            "stability_margin": self.robustness.stability_margin if self.robustness else None,
        }

    def get_flip_summary(self) -> List[str]:
        """Get human-readable summary of decision flips."""
        summaries = []
        for flip in self.decision_flips:
            summaries.append(
                f"Decision flips from {flip['original_decision']} to {flip['new_decision']} "
                f"if: {flip['perturbation_summary']}"
            )
        return summaries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "company_id": self.company_id,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "analyst_id": self.analyst_id,
            "model_version": self.model_version,

            # Summary
            "summary": self.get_summary(),

            # Critical claims
            "critical_claims": [c.to_dict() for c in self.critical_claims],

            # Robustness
            "robustness": self.robustness.to_dict() if self.robustness else None,

            # Counterfactual explanations
            "counterfactual_explanations": [e.to_dict() for e in self.counterfactual_explanations],

            # Decision flips
            "decision_flips": self.decision_flips,
            "flip_summary": self.get_flip_summary(),

            # Trace entries
            "trace_entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_analysis_result(
        cls,
        company_id: str,
        result: DecisionAnalysisResult,
        counterfactuals: Optional[List[CounterfactualEvidenceGraph]] = None,
    ) -> "VCDecisionTrace":
        """
        Create trace from analysis result.

        Args:
            company_id: Company identifier
            result: Decision analysis result
            counterfactuals: Optional list of counterfactuals tested

        Returns:
            VCDecisionTrace
        """
        trace = cls(
            company_id=company_id,
            decision=result.decision,
            confidence=result.confidence,
            critical_claims=result.critical_claims,
            robustness=result.robustness,
            counterfactual_explanations=result.counterfactual_explanations,
        )

        # Add initial analysis entry
        trace.add_entry("analysis_start", {
            "decision": result.decision.value,
            "confidence": result.confidence,
            "metadata": result.analysis_metadata,
        })

        # Add critical claims entries
        for cc in result.critical_claims:
            trace.add_entry("critical_claim_identified", {
                "claim_index": cc.claim_index,
                "claim_type": cc.claim.claim_type.value,
                "claim_field": cc.claim.field,
                "criticality_score": cc.criticality_score,
                "flip_description": cc.flip_description,
            })

        # Add robustness entry
        if result.robustness:
            trace.add_entry("robustness_analysis", result.robustness.to_dict())

        # Add counterfactual entries
        for exp in result.counterfactual_explanations:
            trace.add_entry("counterfactual_explanation", {
                "explanation": exp.explanation,
                "key_changes": exp.key_changes,
                "original_decision": exp.original_decision.value,
                "flipped_decision": exp.flipped_decision.value,
            })

            # Record the flip
            trace.decision_flips.append({
                "perturbation_summary": exp.counterfactual.perturbation_summary,
                "total_magnitude": exp.counterfactual.total_perturbation_magnitude,
                "original_decision": exp.original_decision.value,
                "new_decision": exp.flipped_decision.value,
                "perturbations": [p.to_dict() for p in exp.counterfactual.perturbations],
            })

        # Add tested counterfactuals if provided
        if counterfactuals:
            trace.counterfactuals_tested = counterfactuals

        trace.add_entry("analysis_complete", trace.get_summary())

        return trace


class VCDecisionTracer:
    """
    Generates decision traces with counterfactual analysis.

    Wraps DecisionAnalyzer and creates complete audit trails.
    """

    def __init__(
        self,
        decision_fn: Callable[[EvidenceGraph], Tuple[DecisionOutcome, float]],
        seed: Optional[int] = None,
        num_counterfactuals: int = 20,
        store_all_counterfactuals: bool = False,
    ):
        """
        Initialize tracer.

        Args:
            decision_fn: Decision function
            seed: Random seed
            num_counterfactuals: Number of counterfactuals to generate
            store_all_counterfactuals: Whether to store all tested counterfactuals
        """
        self.decision_fn = decision_fn
        self.analyzer = DecisionAnalyzer(
            decision_fn,
            seed=seed,
            num_counterfactuals=num_counterfactuals,
        )
        self.store_all_counterfactuals = store_all_counterfactuals

    def trace(
        self,
        graph: EvidenceGraph,
        analyst_id: Optional[str] = None,
    ) -> VCDecisionTrace:
        """
        Generate complete decision trace with counterfactual analysis.

        Args:
            graph: Evidence graph to analyze
            analyst_id: Optional analyst identifier

        Returns:
            Complete VCDecisionTrace
        """
        # Perform analysis
        result = self.analyzer.analyze(graph)

        # Create trace
        trace = VCDecisionTrace.from_analysis_result(
            company_id=graph.company_id,
            result=result,
        )
        trace.analyst_id = analyst_id

        # Add additional counterfactual testing if requested
        if self.store_all_counterfactuals:
            from .counterfactuals import generate_counterfactuals

            original_decision = result.decision
            counterfactuals = generate_counterfactuals(graph, num_counterfactuals=20)

            for cf in counterfactuals:
                new_decision, _ = self.decision_fn(cf.modified_graph)
                trace.add_counterfactual_result(cf, original_decision, new_decision)

        return trace


def create_decision_trace(
    graph: EvidenceGraph,
    decision_fn: Callable[[EvidenceGraph], Tuple[DecisionOutcome, float]],
    analyst_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> VCDecisionTrace:
    """
    Convenience function to create a decision trace.

    Args:
        graph: Evidence graph to analyze
        decision_fn: Decision function
        analyst_id: Optional analyst identifier
        seed: Random seed

    Returns:
        Complete VCDecisionTrace
    """
    tracer = VCDecisionTracer(decision_fn, seed=seed)
    return tracer.trace(graph, analyst_id=analyst_id)
