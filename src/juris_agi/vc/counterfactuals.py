"""
Counterfactual Generation for VC Investment Decisions.

Generates minimal perturbations to claim values while respecting ontology.
Used to identify decision-critical claims and compute robustness scores.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from copy import deepcopy
import math

from ..evidence.schema import EvidenceGraph, Claim, Polarity, Source
from ..evidence.ontology import ClaimType, get_all_claim_types, CLAIM_TYPE_METADATA


class PerturbationType(Enum):
    """Types of claim perturbations."""
    VALUE_CHANGE = "value_change"       # Change the value (increase/decrease)
    POLARITY_FLIP = "polarity_flip"     # Flip polarity (supportive ↔ risk)
    CONFIDENCE_CHANGE = "confidence_change"  # Adjust confidence
    CLAIM_REMOVAL = "claim_removal"     # Remove a claim
    CLAIM_ADDITION = "claim_addition"   # Add a missing claim


@dataclass
class ClaimPerturbation:
    """
    A single perturbation to a claim.

    Represents a minimal change to the evidence graph.
    """
    perturbation_type: PerturbationType
    claim_index: int
    """Index of the claim being perturbed (or -1 for additions)."""

    original_claim: Optional[Claim]
    """The original claim before perturbation."""

    modified_claim: Optional[Claim]
    """The claim after perturbation (None for removals)."""

    field_changed: Optional[str] = None
    """Which field was changed (for VALUE_CHANGE)."""

    original_value: Any = None
    """Original value before change."""

    new_value: Any = None
    """New value after change."""

    magnitude: float = 0.0
    """How large the perturbation is (0-1, normalized)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "perturbation_type": self.perturbation_type.value,
            "claim_index": self.claim_index,
            "field_changed": self.field_changed,
            "original_value": str(self.original_value) if self.original_value is not None else None,
            "new_value": str(self.new_value) if self.new_value is not None else None,
            "magnitude": self.magnitude,
            "claim_type": self.original_claim.claim_type.value if self.original_claim else None,
            "claim_field": self.original_claim.field if self.original_claim else None,
        }


@dataclass
class CounterfactualEvidenceGraph:
    """
    A counterfactual evidence graph with tracked perturbations.

    Represents a minimally modified version of the original evidence.
    """
    original_graph: EvidenceGraph
    """The original evidence graph."""

    modified_graph: EvidenceGraph
    """The perturbed evidence graph."""

    perturbations: List[ClaimPerturbation]
    """List of perturbations applied."""

    total_perturbation_magnitude: float = 0.0
    """Sum of all perturbation magnitudes."""

    @property
    def num_perturbations(self) -> int:
        return len(self.perturbations)

    @property
    def perturbation_summary(self) -> str:
        """Generate human-readable summary of perturbations."""
        if not self.perturbations:
            return "No changes"

        summaries = []
        for p in self.perturbations:
            if p.perturbation_type == PerturbationType.VALUE_CHANGE:
                summaries.append(
                    f"{p.original_claim.claim_type.value}.{p.field_changed}: "
                    f"{p.original_value} → {p.new_value}"
                )
            elif p.perturbation_type == PerturbationType.POLARITY_FLIP:
                old_pol = p.original_claim.polarity.value if p.original_claim else "unknown"
                new_pol = p.modified_claim.polarity.value if p.modified_claim else "unknown"
                summaries.append(
                    f"{p.original_claim.claim_type.value}.{p.original_claim.field}: "
                    f"polarity {old_pol} → {new_pol}"
                )
            elif p.perturbation_type == PerturbationType.CONFIDENCE_CHANGE:
                summaries.append(
                    f"{p.original_claim.claim_type.value}.{p.original_claim.field}: "
                    f"confidence {p.original_value:.2f} → {p.new_value:.2f}"
                )
            elif p.perturbation_type == PerturbationType.CLAIM_REMOVAL:
                summaries.append(
                    f"Remove {p.original_claim.claim_type.value}.{p.original_claim.field}"
                )
            elif p.perturbation_type == PerturbationType.CLAIM_ADDITION:
                summaries.append(
                    f"Add {p.modified_claim.claim_type.value}.{p.modified_claim.field}"
                )

        return "; ".join(summaries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_company_id": self.original_graph.company_id,
            "num_perturbations": self.num_perturbations,
            "total_magnitude": self.total_perturbation_magnitude,
            "perturbations": [p.to_dict() for p in self.perturbations],
            "summary": self.perturbation_summary,
        }


class EvidenceCounterfactualGenerator:
    """
    Generates counterfactual evidence graphs.

    Applies minimal perturbations while respecting ontology constraints.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_perturbations: int = 3,
        prefer_minimal: bool = True,
    ):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
            max_perturbations: Maximum perturbations per counterfactual
            prefer_minimal: Prefer smaller perturbations
        """
        import random
        self.rng = random.Random(seed)
        self.max_perturbations = max_perturbations
        self.prefer_minimal = prefer_minimal

    def generate(
        self,
        graph: EvidenceGraph,
        num_counterfactuals: int = 5,
        target_claims: Optional[List[int]] = None,
    ) -> List[CounterfactualEvidenceGraph]:
        """
        Generate counterfactual evidence graphs.

        Args:
            graph: Original evidence graph
            num_counterfactuals: Number of counterfactuals to generate
            target_claims: Optional list of claim indices to perturb
                          (if None, considers all claims)

        Returns:
            List of counterfactual evidence graphs
        """
        counterfactuals = []

        # Generate different types of counterfactuals
        strategies = [
            self._single_value_change,
            self._polarity_flip,
            self._confidence_reduction,
            self._claim_removal,
            self._multi_perturbation,
        ]

        for i in range(num_counterfactuals):
            strategy = strategies[i % len(strategies)]
            cf = strategy(graph, target_claims)
            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def generate_minimal_flip(
        self,
        graph: EvidenceGraph,
        decision_fn: Callable[[EvidenceGraph], bool],
        max_attempts: int = 100,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """
        Generate the minimal counterfactual that flips the decision.

        Uses binary search style approach to find smallest perturbation
        that changes the decision outcome.

        Args:
            graph: Original evidence graph
            decision_fn: Function that returns True/False for the decision
            max_attempts: Maximum perturbation attempts

        Returns:
            Minimal counterfactual that flips decision, or None if not found
        """
        original_decision = decision_fn(graph)

        # Try single-claim perturbations first (minimal)
        for i, claim in enumerate(graph.claims):
            # Try value changes
            for magnitude in [0.1, 0.25, 0.5, 1.0]:
                cf = self._perturb_claim_value(graph, i, magnitude)
                if cf and decision_fn(cf.modified_graph) != original_decision:
                    return cf

            # Try polarity flip
            cf = self._create_polarity_flip(graph, i)
            if cf and decision_fn(cf.modified_graph) != original_decision:
                return cf

            # Try confidence reduction
            for new_conf in [0.5, 0.3, 0.1]:
                cf = self._create_confidence_change(graph, i, new_conf)
                if cf and decision_fn(cf.modified_graph) != original_decision:
                    return cf

        # Try claim removal
        for i in range(len(graph.claims)):
            cf = self._create_claim_removal(graph, i)
            if cf and decision_fn(cf.modified_graph) != original_decision:
                return cf

        # Try multi-claim perturbations
        attempts = 0
        while attempts < max_attempts:
            num_perturb = self.rng.randint(2, min(self.max_perturbations, len(graph.claims)))
            indices = self.rng.sample(range(len(graph.claims)), num_perturb)

            cf = self._multi_perturbation(graph, indices)
            if cf and decision_fn(cf.modified_graph) != original_decision:
                return cf
            attempts += 1

        return None

    def _single_value_change(
        self,
        graph: EvidenceGraph,
        target_claims: Optional[List[int]] = None,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Generate counterfactual with single value change."""
        if not graph.claims:
            return None

        # Select a claim to perturb
        candidates = target_claims if target_claims else list(range(len(graph.claims)))
        if not candidates:
            return None

        claim_idx = self.rng.choice(candidates)
        magnitude = self.rng.choice([0.1, 0.25, 0.5]) if self.prefer_minimal else self.rng.random()

        return self._perturb_claim_value(graph, claim_idx, magnitude)

    def _perturb_claim_value(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
        magnitude: float,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Perturb a single claim's value."""
        if claim_idx >= len(graph.claims):
            return None

        original_claim = graph.claims[claim_idx]
        new_value, field_changed = self._generate_perturbed_value(
            original_claim.value,
            original_claim.claim_type,
            original_claim.field,
            magnitude,
        )

        if new_value == original_claim.value:
            return None

        # Create modified claim
        modified_claim = Claim(
            claim_type=original_claim.claim_type,
            field=original_claim.field,
            value=new_value,
            confidence=original_claim.confidence,
            polarity=original_claim.polarity,
            source=original_claim.source,
            unit=original_claim.unit,
            notes=original_claim.notes,
            claim_id=original_claim.claim_id,
        )

        # Create modified graph
        modified_graph = self._copy_graph_with_modified_claim(graph, claim_idx, modified_claim)

        perturbation = ClaimPerturbation(
            perturbation_type=PerturbationType.VALUE_CHANGE,
            claim_index=claim_idx,
            original_claim=original_claim,
            modified_claim=modified_claim,
            field_changed=field_changed or original_claim.field,
            original_value=original_claim.value,
            new_value=new_value,
            magnitude=magnitude,
        )

        return CounterfactualEvidenceGraph(
            original_graph=graph,
            modified_graph=modified_graph,
            perturbations=[perturbation],
            total_perturbation_magnitude=magnitude,
        )

    def _generate_perturbed_value(
        self,
        value: Any,
        claim_type: ClaimType,
        field: str,
        magnitude: float,
    ) -> Tuple[Any, Optional[str]]:
        """
        Generate a perturbed value respecting ontology.

        Returns (new_value, field_name) tuple.
        """
        if isinstance(value, (int, float)):
            # Numeric perturbation
            if value == 0:
                # Special case: 0 -> small positive or negative
                new_value = magnitude * 100 * (1 if self.rng.random() > 0.5 else -1)
            else:
                # Percentage change based on magnitude
                direction = 1 if self.rng.random() > 0.5 else -1
                change = value * magnitude * direction
                new_value = value + change

            # Keep same type
            if isinstance(value, int):
                new_value = int(round(new_value))

            return new_value, None

        elif isinstance(value, str):
            # String perturbation - use semantic changes based on claim type
            new_value = self._perturb_string_value(value, claim_type, field, magnitude)
            return new_value, None

        elif isinstance(value, bool):
            # Boolean flip
            return not value, None

        elif isinstance(value, list):
            # List perturbation - add/remove/modify elements
            if magnitude > 0.5 and value:
                # Remove elements
                new_value = value[:-max(1, int(len(value) * magnitude))]
            else:
                # Keep as is (could add elements but risky)
                new_value = value
            return new_value, None

        elif isinstance(value, dict):
            # Dict perturbation - modify a random key
            if value:
                key = self.rng.choice(list(value.keys()))
                new_dict = dict(value)
                new_dict[key], _ = self._generate_perturbed_value(
                    value[key], claim_type, key, magnitude
                )
                return new_dict, key

        return value, None

    def _perturb_string_value(
        self,
        value: str,
        claim_type: ClaimType,
        field: str,
        magnitude: float,
    ) -> str:
        """Perturb string value semantically based on claim type."""
        # Risk-related claim types
        if claim_type in [ClaimType.EXECUTION_RISK, ClaimType.REGULATORY_RISK]:
            levels = ["low", "moderate", "high", "critical"]
            if any(level in value.lower() for level in levels):
                current_idx = next(
                    (i for i, l in enumerate(levels) if l in value.lower()),
                    1
                )
                # Shift based on magnitude
                shift = int(magnitude * 2) * (1 if self.rng.random() > 0.5 else -1)
                new_idx = max(0, min(len(levels) - 1, current_idx + shift))
                return levels[new_idx]

        # Stage-related fields
        if field in ["stage", "product_stage", "readiness"]:
            stages = ["idea", "prototype", "mvp", "beta", "ga", "growth"]
            if any(s in value.lower() for s in stages):
                current_idx = next(
                    (i for i, s in enumerate(stages) if s in value.lower()),
                    2
                )
                shift = int(magnitude * 2) * (1 if self.rng.random() > 0.5 else -1)
                new_idx = max(0, min(len(stages) - 1, current_idx + shift))
                return stages[new_idx]

        # Default: append modifier
        modifiers = ["(weak)", "(uncertain)", "(disputed)", "(revised)"]
        if not any(m in value for m in modifiers):
            return f"{value} {self.rng.choice(modifiers)}"

        return value

    def _polarity_flip(
        self,
        graph: EvidenceGraph,
        target_claims: Optional[List[int]] = None,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Generate counterfactual with polarity flip."""
        candidates = target_claims if target_claims else list(range(len(graph.claims)))
        # Prefer claims that aren't neutral
        candidates = [
            i for i in candidates
            if i < len(graph.claims) and graph.claims[i].polarity != Polarity.NEUTRAL
        ]

        if not candidates:
            return None

        claim_idx = self.rng.choice(candidates)
        return self._create_polarity_flip(graph, claim_idx)

    def _create_polarity_flip(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Create a polarity flip counterfactual."""
        if claim_idx >= len(graph.claims):
            return None

        original_claim = graph.claims[claim_idx]

        # Flip polarity
        if original_claim.polarity == Polarity.SUPPORTIVE:
            new_polarity = Polarity.RISK
        elif original_claim.polarity == Polarity.RISK:
            new_polarity = Polarity.SUPPORTIVE
        else:
            # Neutral -> random
            new_polarity = self.rng.choice([Polarity.SUPPORTIVE, Polarity.RISK])

        modified_claim = Claim(
            claim_type=original_claim.claim_type,
            field=original_claim.field,
            value=original_claim.value,
            confidence=original_claim.confidence,
            polarity=new_polarity,
            source=original_claim.source,
            unit=original_claim.unit,
            notes=original_claim.notes,
            claim_id=original_claim.claim_id,
        )

        modified_graph = self._copy_graph_with_modified_claim(graph, claim_idx, modified_claim)

        perturbation = ClaimPerturbation(
            perturbation_type=PerturbationType.POLARITY_FLIP,
            claim_index=claim_idx,
            original_claim=original_claim,
            modified_claim=modified_claim,
            original_value=original_claim.polarity.value,
            new_value=new_polarity.value,
            magnitude=1.0,  # Polarity flip is significant
        )

        return CounterfactualEvidenceGraph(
            original_graph=graph,
            modified_graph=modified_graph,
            perturbations=[perturbation],
            total_perturbation_magnitude=1.0,
        )

    def _confidence_reduction(
        self,
        graph: EvidenceGraph,
        target_claims: Optional[List[int]] = None,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Generate counterfactual with reduced confidence."""
        candidates = target_claims if target_claims else list(range(len(graph.claims)))
        # Prefer high-confidence claims
        candidates = [
            i for i in candidates
            if i < len(graph.claims) and graph.claims[i].confidence > 0.5
        ]

        if not candidates:
            return None

        claim_idx = self.rng.choice(candidates)
        new_confidence = self.rng.uniform(0.1, 0.5)

        return self._create_confidence_change(graph, claim_idx, new_confidence)

    def _create_confidence_change(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
        new_confidence: float,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Create a confidence change counterfactual."""
        if claim_idx >= len(graph.claims):
            return None

        original_claim = graph.claims[claim_idx]

        modified_claim = Claim(
            claim_type=original_claim.claim_type,
            field=original_claim.field,
            value=original_claim.value,
            confidence=new_confidence,
            polarity=original_claim.polarity,
            source=original_claim.source,
            unit=original_claim.unit,
            notes=original_claim.notes,
            claim_id=original_claim.claim_id,
        )

        modified_graph = self._copy_graph_with_modified_claim(graph, claim_idx, modified_claim)

        magnitude = abs(original_claim.confidence - new_confidence)

        perturbation = ClaimPerturbation(
            perturbation_type=PerturbationType.CONFIDENCE_CHANGE,
            claim_index=claim_idx,
            original_claim=original_claim,
            modified_claim=modified_claim,
            original_value=original_claim.confidence,
            new_value=new_confidence,
            magnitude=magnitude,
        )

        return CounterfactualEvidenceGraph(
            original_graph=graph,
            modified_graph=modified_graph,
            perturbations=[perturbation],
            total_perturbation_magnitude=magnitude,
        )

    def _claim_removal(
        self,
        graph: EvidenceGraph,
        target_claims: Optional[List[int]] = None,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Generate counterfactual with claim removed."""
        if not graph.claims:
            return None

        candidates = target_claims if target_claims else list(range(len(graph.claims)))
        if not candidates:
            return None

        claim_idx = self.rng.choice(candidates)
        return self._create_claim_removal(graph, claim_idx)

    def _create_claim_removal(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Create a claim removal counterfactual."""
        if claim_idx >= len(graph.claims):
            return None

        original_claim = graph.claims[claim_idx]

        # Create modified graph without this claim
        modified_claims = [
            c for i, c in enumerate(graph.claims) if i != claim_idx
        ]
        modified_graph = EvidenceGraph(
            company_id=graph.company_id,
            claims=modified_claims,
            analyst_id=graph.analyst_id,
            version=graph.version,
        )

        perturbation = ClaimPerturbation(
            perturbation_type=PerturbationType.CLAIM_REMOVAL,
            claim_index=claim_idx,
            original_claim=original_claim,
            modified_claim=None,
            magnitude=1.0,  # Removal is significant
        )

        return CounterfactualEvidenceGraph(
            original_graph=graph,
            modified_graph=modified_graph,
            perturbations=[perturbation],
            total_perturbation_magnitude=1.0,
        )

    def _multi_perturbation(
        self,
        graph: EvidenceGraph,
        target_claims: Optional[List[int]] = None,
    ) -> Optional[CounterfactualEvidenceGraph]:
        """Generate counterfactual with multiple perturbations."""
        if not graph.claims:
            return None

        candidates = target_claims if target_claims else list(range(len(graph.claims)))
        num_perturb = min(
            self.rng.randint(2, self.max_perturbations + 1),
            len(candidates)
        )

        if num_perturb < 2:
            return self._single_value_change(graph, target_claims)

        selected = self.rng.sample(candidates, num_perturb)

        # Apply perturbations sequentially
        current_graph = graph
        all_perturbations = []
        total_magnitude = 0.0

        for i, claim_idx in enumerate(selected):
            # Choose perturbation type
            perturb_type = self.rng.choice([
                PerturbationType.VALUE_CHANGE,
                PerturbationType.CONFIDENCE_CHANGE,
            ])

            if perturb_type == PerturbationType.VALUE_CHANGE:
                magnitude = self.rng.uniform(0.1, 0.5)
                cf = self._perturb_claim_value(current_graph, claim_idx, magnitude)
            else:
                new_conf = self.rng.uniform(0.2, 0.8)
                cf = self._create_confidence_change(current_graph, claim_idx, new_conf)

            if cf:
                current_graph = cf.modified_graph
                all_perturbations.extend(cf.perturbations)
                total_magnitude += cf.total_perturbation_magnitude

        if not all_perturbations:
            return None

        return CounterfactualEvidenceGraph(
            original_graph=graph,
            modified_graph=current_graph,
            perturbations=all_perturbations,
            total_perturbation_magnitude=total_magnitude,
        )

    def _copy_graph_with_modified_claim(
        self,
        graph: EvidenceGraph,
        claim_idx: int,
        new_claim: Claim,
    ) -> EvidenceGraph:
        """Create a copy of graph with one claim modified."""
        modified_claims = [
            new_claim if i == claim_idx else c
            for i, c in enumerate(graph.claims)
        ]
        return EvidenceGraph(
            company_id=graph.company_id,
            claims=modified_claims,
            analyst_id=graph.analyst_id,
            version=graph.version,
        )


def generate_counterfactuals(
    graph: EvidenceGraph,
    num_counterfactuals: int = 5,
    seed: Optional[int] = None,
) -> List[CounterfactualEvidenceGraph]:
    """
    Convenience function to generate counterfactual evidence graphs.

    Args:
        graph: Original evidence graph
        num_counterfactuals: Number of counterfactuals to generate
        seed: Random seed for reproducibility

    Returns:
        List of counterfactual evidence graphs
    """
    generator = EvidenceCounterfactualGenerator(seed=seed)
    return generator.generate(graph, num_counterfactuals)
