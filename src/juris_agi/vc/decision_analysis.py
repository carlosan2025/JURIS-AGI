"""
VC Decision Analysis with Counterfactual Reasoning.

Analyzes which claims are decision-critical and computes robustness scores.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from copy import deepcopy

from ..evidence.schema import EvidenceGraph, Claim, Polarity
from ..evidence.ontology import ClaimType, get_all_claim_types
from .counterfactuals import (
    EvidenceCounterfactualGenerator,
    CounterfactualEvidenceGraph,
    ClaimPerturbation,
    PerturbationType,
)


class DecisionOutcome(Enum):
    """Possible investment decision outcomes."""
    INVEST = "invest"
    PASS = "pass"
    DEFER = "defer"  # Need more information


@dataclass
class DecisionCriticalClaim:
    """
    A claim identified as critical to the decision.

    A claim is critical if changing it would flip the decision.
    """
    claim_index: int
    claim: Claim
    criticality_score: float
    """How critical (0-1). 1.0 = single change flips decision."""

    minimal_flip_perturbation: Optional[ClaimPerturbation] = None
    """The smallest perturbation that flips the decision."""

    flip_description: Optional[str] = None
    """Natural language description of what change would flip the decision."""

    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    """How sensitive the decision is to different perturbation types."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_index": self.claim_index,
            "claim_type": self.claim.claim_type.value,
            "claim_field": self.claim.field,
            "claim_value": str(self.claim.value),
            "criticality_score": self.criticality_score,
            "minimal_flip": self.minimal_flip_perturbation.to_dict() if self.minimal_flip_perturbation else None,
            "flip_description": self.flip_description,
            "sensitivity": self.sensitivity_analysis,
        }


@dataclass
class DecisionRobustness:
    """
    Robustness analysis of a decision.

    Measures how stable the decision is under perturbations.
    """
    overall_score: float
    """Robustness score 0-1. 1.0 = very robust, 0.0 = fragile."""

    stability_margin: float
    """How much perturbation is needed to flip (average across all flips)."""

    num_critical_claims: int
    """Number of claims that are decision-critical."""

    most_fragile_claim_index: Optional[int] = None
    """Index of the claim most likely to flip the decision."""

    perturbations_tested: int = 0
    """Total number of perturbations tested."""

    flips_found: int = 0
    """Number of perturbations that flipped the decision."""

    robustness_by_claim_type: Dict[str, float] = field(default_factory=dict)
    """Robustness breakdown by claim type."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "stability_margin": self.stability_margin,
            "num_critical_claims": self.num_critical_claims,
            "most_fragile_claim_index": self.most_fragile_claim_index,
            "perturbations_tested": self.perturbations_tested,
            "flips_found": self.flips_found,
            "robustness_by_claim_type": self.robustness_by_claim_type,
        }


@dataclass
class CounterfactualExplanation:
    """
    Natural language explanation of a counterfactual.

    Describes what would need to change to flip the decision.
    """
    counterfactual: CounterfactualEvidenceGraph
    original_decision: DecisionOutcome
    flipped_decision: DecisionOutcome
    explanation: str
    """Natural language explanation."""

    key_changes: List[str]
    """List of key changes that caused the flip."""

    confidence: float = 1.0
    """Confidence in this explanation."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_decision": self.original_decision.value,
            "flipped_decision": self.flipped_decision.value,
            "explanation": self.explanation,
            "key_changes": self.key_changes,
            "confidence": self.confidence,
            "perturbation_summary": self.counterfactual.perturbation_summary,
            "total_magnitude": self.counterfactual.total_perturbation_magnitude,
        }


@dataclass
class DecisionAnalysisResult:
    """
    Complete decision analysis result.

    Contains criticality analysis, robustness scores, and counterfactual explanations.
    """
    decision: DecisionOutcome
    """The original decision."""

    confidence: float
    """Confidence in the decision (0-1)."""

    critical_claims: List[DecisionCriticalClaim]
    """Claims identified as decision-critical."""

    robustness: DecisionRobustness
    """Robustness analysis."""

    counterfactual_explanations: List[CounterfactualExplanation]
    """Natural language counterfactual explanations."""

    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the analysis."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "critical_claims": [c.to_dict() for c in self.critical_claims],
            "robustness": self.robustness.to_dict(),
            "counterfactual_explanations": [e.to_dict() for e in self.counterfactual_explanations],
            "metadata": self.analysis_metadata,
        }


class DecisionAnalyzer:
    """
    Analyzes VC investment decisions using counterfactual reasoning.

    Identifies critical claims, computes robustness, and generates explanations.
    """

    def __init__(
        self,
        decision_fn: Callable[[EvidenceGraph], Tuple[DecisionOutcome, float]],
        seed: Optional[int] = None,
        num_counterfactuals: int = 20,
        max_perturbations_per_claim: int = 5,
    ):
        """
        Initialize analyzer.

        Args:
            decision_fn: Function that takes EvidenceGraph and returns
                        (DecisionOutcome, confidence) tuple
            seed: Random seed for reproducibility
            num_counterfactuals: Number of counterfactuals to generate
            max_perturbations_per_claim: Max perturbations to try per claim
        """
        self.decision_fn = decision_fn
        self.generator = EvidenceCounterfactualGenerator(seed=seed)
        self.num_counterfactuals = num_counterfactuals
        self.max_perturbations_per_claim = max_perturbations_per_claim

    def analyze(self, graph: EvidenceGraph) -> DecisionAnalysisResult:
        """
        Perform complete decision analysis.

        Args:
            graph: Evidence graph to analyze

        Returns:
            DecisionAnalysisResult with criticality, robustness, and explanations
        """
        # Get original decision
        original_decision, original_confidence = self.decision_fn(graph)

        # Find critical claims
        critical_claims = self._find_critical_claims(graph, original_decision)

        # Compute robustness
        robustness = self._compute_robustness(graph, original_decision, critical_claims)

        # Generate counterfactual explanations
        explanations = self._generate_explanations(graph, original_decision, critical_claims)

        return DecisionAnalysisResult(
            decision=original_decision,
            confidence=original_confidence,
            critical_claims=critical_claims,
            robustness=robustness,
            counterfactual_explanations=explanations,
            analysis_metadata={
                "graph_company_id": graph.company_id,
                "num_claims": graph.claim_count,
                "coverage_ratio": graph.coverage_ratio,
            },
        )

    def _find_critical_claims(
        self,
        graph: EvidenceGraph,
        original_decision: DecisionOutcome,
    ) -> List[DecisionCriticalClaim]:
        """Identify which claims are decision-critical."""
        critical_claims = []

        for i, claim in enumerate(graph.claims):
            criticality, min_flip, sensitivity = self._analyze_claim_criticality(
                graph, i, original_decision
            )

            if criticality > 0.0:
                flip_desc = self._generate_flip_description(claim, min_flip)

                critical_claims.append(DecisionCriticalClaim(
                    claim_index=i,
                    claim=claim,
                    criticality_score=criticality,
                    minimal_flip_perturbation=min_flip,
                    flip_description=flip_desc,
                    sensitivity_analysis=sensitivity,
                ))

        # Sort by criticality (most critical first)
        critical_claims.sort(key=lambda c: c.criticality_score, reverse=True)
        return critical_claims

    def _analyze_claim_criticality(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
        original_decision: DecisionOutcome,
    ) -> Tuple[float, Optional[ClaimPerturbation], Dict[str, float]]:
        """
        Analyze how critical a specific claim is.

        Returns (criticality_score, minimal_flip_perturbation, sensitivity_dict).
        """
        min_flip_magnitude = float('inf')
        min_flip_perturbation = None
        sensitivity = {}

        # Try different perturbation types
        perturbation_types = [
            ("value_small", lambda g, i: self.generator._perturb_claim_value(g, i, 0.1)),
            ("value_medium", lambda g, i: self.generator._perturb_claim_value(g, i, 0.3)),
            ("value_large", lambda g, i: self.generator._perturb_claim_value(g, i, 0.5)),
            ("polarity", lambda g, i: self.generator._create_polarity_flip(g, i)),
            ("confidence_low", lambda g, i: self.generator._create_confidence_change(g, i, 0.2)),
            ("confidence_medium", lambda g, i: self.generator._create_confidence_change(g, i, 0.5)),
            ("removal", lambda g, i: self.generator._create_claim_removal(g, i)),
        ]

        for ptype_name, perturb_fn in perturbation_types:
            cf = perturb_fn(graph, claim_idx)
            if cf is None:
                continue

            new_decision, _ = self.decision_fn(cf.modified_graph)
            flipped = (new_decision != original_decision)

            # Record sensitivity
            sensitivity[ptype_name] = 1.0 if flipped else 0.0

            # Track minimum flip magnitude
            if flipped and cf.total_perturbation_magnitude < min_flip_magnitude:
                min_flip_magnitude = cf.total_perturbation_magnitude
                min_flip_perturbation = cf.perturbations[0] if cf.perturbations else None

        # Compute criticality score
        if min_flip_magnitude < float('inf'):
            # Criticality is inverse of perturbation magnitude needed
            # Small perturbation to flip = high criticality
            criticality = 1.0 - min(min_flip_magnitude, 1.0) * 0.5
        else:
            # No single-claim perturbation flipped the decision
            criticality = 0.0

        return criticality, min_flip_perturbation, sensitivity

    def _compute_robustness(
        self,
        graph: EvidenceGraph,
        original_decision: DecisionOutcome,
        critical_claims: List[DecisionCriticalClaim],
    ) -> DecisionRobustness:
        """Compute overall decision robustness."""
        # Generate random counterfactuals
        counterfactuals = self.generator.generate(graph, self.num_counterfactuals)

        perturbations_tested = len(counterfactuals)
        flips_found = 0
        flip_magnitudes = []

        # Track robustness by claim type
        type_flips: Dict[str, List[bool]] = {}

        for cf in counterfactuals:
            new_decision, _ = self.decision_fn(cf.modified_graph)
            flipped = (new_decision != original_decision)

            if flipped:
                flips_found += 1
                flip_magnitudes.append(cf.total_perturbation_magnitude)

            # Track by claim type
            for p in cf.perturbations:
                if p.original_claim:
                    ct = p.original_claim.claim_type.value
                    if ct not in type_flips:
                        type_flips[ct] = []
                    type_flips[ct].append(flipped)

        # Compute overall robustness
        if perturbations_tested > 0:
            flip_rate = flips_found / perturbations_tested
            overall_score = 1.0 - flip_rate
        else:
            overall_score = 1.0

        # Compute stability margin
        if flip_magnitudes:
            stability_margin = sum(flip_magnitudes) / len(flip_magnitudes)
        else:
            stability_margin = 1.0  # No flips found, very stable

        # Find most fragile claim
        most_fragile = None
        if critical_claims:
            most_fragile = critical_claims[0].claim_index  # Already sorted

        # Robustness by claim type
        robustness_by_type = {}
        for ct, flips in type_flips.items():
            if flips:
                robustness_by_type[ct] = 1.0 - (sum(flips) / len(flips))
            else:
                robustness_by_type[ct] = 1.0

        return DecisionRobustness(
            overall_score=overall_score,
            stability_margin=stability_margin,
            num_critical_claims=len(critical_claims),
            most_fragile_claim_index=most_fragile,
            perturbations_tested=perturbations_tested,
            flips_found=flips_found,
            robustness_by_claim_type=robustness_by_type,
        )

    def _generate_explanations(
        self,
        graph: EvidenceGraph,
        original_decision: DecisionOutcome,
        critical_claims: List[DecisionCriticalClaim],
    ) -> List[CounterfactualExplanation]:
        """Generate natural language counterfactual explanations."""
        explanations = []

        # Generate explanation for each critical claim
        for cc in critical_claims[:5]:  # Top 5 most critical
            if cc.minimal_flip_perturbation is None:
                continue

            # Find the counterfactual that caused the flip
            cf = self._reconstruct_counterfactual(graph, cc)
            if cf is None:
                continue

            new_decision, _ = self.decision_fn(cf.modified_graph)
            if new_decision == original_decision:
                continue  # Didn't actually flip

            explanation = self._generate_explanation_text(
                cc, original_decision, new_decision
            )
            key_changes = self._extract_key_changes(cf)

            explanations.append(CounterfactualExplanation(
                counterfactual=cf,
                original_decision=original_decision,
                flipped_decision=new_decision,
                explanation=explanation,
                key_changes=key_changes,
                confidence=cc.criticality_score,
            ))

        return explanations

    def _reconstruct_counterfactual(
        self,
        graph: EvidenceGraph,
        critical_claim: DecisionCriticalClaim,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Reconstruct the counterfactual from a critical claim."""
        p = critical_claim.minimal_flip_perturbation
        if p is None:
            return None

        if p.perturbation_type == PerturbationType.VALUE_CHANGE:
            return self.generator._perturb_claim_value(
                graph, critical_claim.claim_index, p.magnitude
            )
        elif p.perturbation_type == PerturbationType.POLARITY_FLIP:
            return self.generator._create_polarity_flip(
                graph, critical_claim.claim_index
            )
        elif p.perturbation_type == PerturbationType.CONFIDENCE_CHANGE:
            return self.generator._create_confidence_change(
                graph, critical_claim.claim_index, p.new_value
            )
        elif p.perturbation_type == PerturbationType.CLAIM_REMOVAL:
            return self.generator._create_claim_removal(
                graph, critical_claim.claim_index
            )

        return None

    def _generate_explanation_text(
        self,
        critical_claim: DecisionCriticalClaim,
        original: DecisionOutcome,
        flipped: DecisionOutcome,
    ) -> str:
        """Generate natural language explanation for a flip."""
        claim = critical_claim.claim
        p = critical_claim.minimal_flip_perturbation

        if p is None:
            return f"Decision sensitive to changes in {claim.claim_type.value}.{claim.field}"

        # Build explanation based on perturbation type
        if p.perturbation_type == PerturbationType.VALUE_CHANGE:
            return (
                f"Decision flips from {original.value} to {flipped.value} "
                f"if {claim.claim_type.value}.{claim.field} changes from "
                f"{p.original_value} to {p.new_value}"
            )

        elif p.perturbation_type == PerturbationType.POLARITY_FLIP:
            old_pol = claim.polarity.value
            new_pol = "risk" if old_pol == "supportive" else "supportive"
            return (
                f"Decision flips from {original.value} to {flipped.value} "
                f"if {claim.claim_type.value}.{claim.field} is reinterpreted "
                f"as {new_pol} instead of {old_pol}"
            )

        elif p.perturbation_type == PerturbationType.CONFIDENCE_CHANGE:
            return (
                f"Decision flips from {original.value} to {flipped.value} "
                f"if confidence in {claim.claim_type.value}.{claim.field} "
                f"drops from {p.original_value:.0%} to {p.new_value:.0%}"
            )

        elif p.perturbation_type == PerturbationType.CLAIM_REMOVAL:
            return (
                f"Decision flips from {original.value} to {flipped.value} "
                f"if {claim.claim_type.value}.{claim.field} is removed or unknown"
            )

        return f"Decision flips if {claim.claim_type.value}.{claim.field} changes"

    def _generate_flip_description(
        self,
        claim: Claim,
        perturbation: Optional[ClaimPerturbation],
    ) -> str:
        """Generate description of what flip would occur."""
        if perturbation is None:
            return f"Changes to {claim.field} may affect decision"

        if perturbation.perturbation_type == PerturbationType.VALUE_CHANGE:
            return f"{claim.field}: {perturbation.original_value} → {perturbation.new_value}"
        elif perturbation.perturbation_type == PerturbationType.POLARITY_FLIP:
            return f"{claim.field}: polarity flip"
        elif perturbation.perturbation_type == PerturbationType.CONFIDENCE_CHANGE:
            return f"{claim.field}: confidence {perturbation.original_value:.0%} → {perturbation.new_value:.0%}"
        elif perturbation.perturbation_type == PerturbationType.CLAIM_REMOVAL:
            return f"{claim.field}: removal"

        return f"Change to {claim.field}"

    def _extract_key_changes(
        self,
        counterfactual: CounterfactualEvidenceGraph,
    ) -> List[str]:
        """Extract key changes from a counterfactual."""
        changes = []
        for p in counterfactual.perturbations:
            if p.original_claim:
                change = f"{p.original_claim.claim_type.value}.{p.original_claim.field}"
                if p.original_value is not None and p.new_value is not None:
                    change += f": {p.original_value} → {p.new_value}"
                changes.append(change)
        return changes


def analyze_decision(
    graph: EvidenceGraph,
    decision_fn: Callable[[EvidenceGraph], Tuple[DecisionOutcome, float]],
    seed: Optional[int] = None,
) -> DecisionAnalysisResult:
    """
    Convenience function to analyze a VC decision.

    Args:
        graph: Evidence graph to analyze
        decision_fn: Function that returns (decision, confidence)
        seed: Random seed

    Returns:
        Complete decision analysis
    """
    analyzer = DecisionAnalyzer(decision_fn, seed=seed)
    return analyzer.analyze(graph)
