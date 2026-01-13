"""
Conflict detection and clustering for claims.

Detects conflicts between claims and clusters related conflicts
for efficient resolution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import math

from juris_agi.evidence_client.types import Claim, Conflict, ConflictType


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""

    CRITICAL = "critical"  # Must be resolved before decision
    HIGH = "high"  # Strongly impacts decision
    MEDIUM = "medium"  # Notable but not blocking
    LOW = "low"  # Minor discrepancy


@dataclass
class DetectedConflict:
    """A detected conflict between claims with analysis metadata."""

    conflict_id: str
    conflict_type: ConflictType
    claims: list[Claim]
    severity: ConflictSeverity
    description: str
    resolution_hint: Optional[str] = None
    value_delta: Optional[float] = None  # For numeric conflicts
    temporal_gap_days: Optional[int] = None  # For temporal conflicts

    def to_conflict(self) -> Conflict:
        """Convert to Evidence API Conflict type."""
        severity_map = {
            ConflictSeverity.CRITICAL: 1.0,
            ConflictSeverity.HIGH: 0.75,
            ConflictSeverity.MEDIUM: 0.5,
            ConflictSeverity.LOW: 0.25,
        }
        return Conflict(
            conflict_id=self.conflict_id,
            conflict_type=self.conflict_type,
            claim_ids=[c.claim_id for c in self.claims],
            description=self.description,
            severity=severity_map[self.severity],
            resolution_hint=self.resolution_hint,
        )


@dataclass
class ConflictCluster:
    """A cluster of related conflicts."""

    cluster_id: str
    conflicts: list[DetectedConflict] = field(default_factory=list)
    primary_claim_type: Optional[str] = None
    primary_field: Optional[str] = None

    @property
    def max_severity(self) -> ConflictSeverity:
        """Get the highest severity in the cluster."""
        if not self.conflicts:
            return ConflictSeverity.LOW
        severity_order = [
            ConflictSeverity.CRITICAL,
            ConflictSeverity.HIGH,
            ConflictSeverity.MEDIUM,
            ConflictSeverity.LOW,
        ]
        for severity in severity_order:
            if any(c.severity == severity for c in self.conflicts):
                return severity
        return ConflictSeverity.LOW

    @property
    def all_claims(self) -> list[Claim]:
        """Get all unique claims in the cluster."""
        seen_ids = set()
        claims = []
        for conflict in self.conflicts:
            for claim in conflict.claims:
                if claim.claim_id not in seen_ids:
                    seen_ids.add(claim.claim_id)
                    claims.append(claim)
        return claims


def _are_values_conflicting(
    value1: Any,
    value2: Any,
    threshold: float = 0.15,
) -> tuple[bool, Optional[float]]:
    """
    Check if two values conflict.

    For numeric values, checks if they differ by more than threshold (15% by default).
    For strings, checks for semantic contradiction.
    For lists, checks for significant differences.

    Returns:
        Tuple of (is_conflicting, relative_delta for numerics)
    """
    # Handle None
    if value1 is None or value2 is None:
        return False, None

    # Numeric comparison
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        if value1 == 0 and value2 == 0:
            return False, 0.0
        if value1 == 0 or value2 == 0:
            return True, float("inf")

        # Calculate relative difference
        avg = (abs(value1) + abs(value2)) / 2
        delta = abs(value1 - value2) / avg
        return delta > threshold, delta

    # String comparison
    if isinstance(value1, str) and isinstance(value2, str):
        v1_lower = value1.lower().strip()
        v2_lower = value2.lower().strip()

        # Exact match
        if v1_lower == v2_lower:
            return False, None

        # Check for obvious contradictions
        contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("approved", "rejected"),
            ("profitable", "unprofitable"),
            ("growing", "declining"),
            ("increasing", "decreasing"),
        ]
        for pos, neg in contradiction_pairs:
            if (pos in v1_lower and neg in v2_lower) or (neg in v1_lower and pos in v2_lower):
                return True, None

        # Different non-trivial strings for same field = potential conflict
        if len(v1_lower) > 5 and len(v2_lower) > 5:
            # Check word overlap
            words1 = set(v1_lower.split())
            words2 = set(v2_lower.split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if overlap < 0.3:  # Less than 30% overlap
                return True, None

        return False, None

    # List comparison
    if isinstance(value1, list) and isinstance(value2, list):
        set1 = set(str(x).lower() for x in value1)
        set2 = set(str(x).lower() for x in value2)
        overlap = len(set1 & set2) / max(len(set1 | set2), 1)
        return overlap < 0.5, None  # Less than 50% overlap

    # Type mismatch
    if type(value1) != type(value2):
        return True, None

    return False, None


def _calculate_temporal_gap(claim1: Claim, claim2: Claim) -> Optional[int]:
    """Calculate temporal gap between claims in days."""
    if claim1.as_of_date is None or claim2.as_of_date is None:
        return None
    delta = abs((claim1.as_of_date - claim2.as_of_date).days)
    return delta


def _determine_conflict_type(
    claim1: Claim,
    claim2: Claim,
    value_conflict: bool,
    temporal_gap: Optional[int],
) -> ConflictType:
    """Determine the type of conflict between claims."""
    # Temporal conflict if gap > 180 days and values differ
    if temporal_gap is not None and temporal_gap > 180 and value_conflict:
        return ConflictType.TEMPORAL

    # Check polarity contradiction
    if claim1.polarity != claim2.polarity:
        if (claim1.polarity.value == "supportive" and claim2.polarity.value == "risk") or (
            claim1.polarity.value == "risk" and claim2.polarity.value == "supportive"
        ):
            return ConflictType.CONTRADICTION

    # Check source disagreement (different documents, same field)
    if claim1.citations and claim2.citations:
        docs1 = {c.document_id for c in claim1.citations}
        docs2 = {c.document_id for c in claim2.citations}
        if not docs1.intersection(docs2) and value_conflict:
            return ConflictType.SOURCE_DISAGREEMENT

    # Default to inconsistency
    return ConflictType.INCONSISTENCY


def _determine_severity(
    conflict_type: ConflictType,
    claim1: Claim,
    claim2: Claim,
    value_delta: Optional[float],
) -> ConflictSeverity:
    """Determine severity of a conflict."""
    # Critical claim types
    critical_types = {"round_terms", "capital_intensity", "regulatory_risk"}
    if claim1.claim_type in critical_types or claim2.claim_type in critical_types:
        if conflict_type == ConflictType.CONTRADICTION:
            return ConflictSeverity.CRITICAL

    # High severity for large numeric discrepancies
    if value_delta is not None:
        if value_delta > 0.5:  # >50% difference
            return ConflictSeverity.HIGH
        elif value_delta > 0.25:  # >25% difference
            return ConflictSeverity.MEDIUM

    # Contradiction type is at least medium
    if conflict_type == ConflictType.CONTRADICTION:
        return ConflictSeverity.HIGH

    # Source disagreement
    if conflict_type == ConflictType.SOURCE_DISAGREEMENT:
        return ConflictSeverity.MEDIUM

    # Temporal conflicts are usually lower severity
    if conflict_type == ConflictType.TEMPORAL:
        return ConflictSeverity.LOW

    return ConflictSeverity.MEDIUM


def detect_conflicts(
    claims: list[Claim],
    numeric_threshold: float = 0.15,
) -> list[DetectedConflict]:
    """
    Detect conflicts between claims.

    Compares claims with matching (claim_type, field) pairs
    for conflicting values.

    Args:
        claims: List of claims to analyze
        numeric_threshold: Threshold for numeric value conflicts (default 15%)

    Returns:
        List of detected conflicts
    """
    conflicts: list[DetectedConflict] = []
    conflict_counter = 0

    # Group claims by (claim_type, field)
    claim_groups: dict[tuple[str, str], list[Claim]] = {}
    for claim in claims:
        key = (claim.claim_type, claim.field)
        if key not in claim_groups:
            claim_groups[key] = []
        claim_groups[key].append(claim)

    # Check for conflicts within each group
    for (claim_type, field_name), group_claims in claim_groups.items():
        if len(group_claims) < 2:
            continue

        # Compare all pairs
        for i, claim1 in enumerate(group_claims):
            for claim2 in group_claims[i + 1 :]:
                is_conflict, value_delta = _are_values_conflicting(
                    claim1.value, claim2.value, numeric_threshold
                )

                if not is_conflict:
                    continue

                temporal_gap = _calculate_temporal_gap(claim1, claim2)
                conflict_type = _determine_conflict_type(
                    claim1, claim2, is_conflict, temporal_gap
                )
                severity = _determine_severity(
                    conflict_type, claim1, claim2, value_delta
                )

                # Generate description
                description = _generate_conflict_description(
                    claim1, claim2, conflict_type, value_delta, temporal_gap
                )

                # Generate resolution hint
                resolution_hint = _generate_resolution_hint(
                    conflict_type, claim1, claim2, temporal_gap
                )

                conflict_counter += 1
                conflicts.append(
                    DetectedConflict(
                        conflict_id=f"conflict_{conflict_counter:04d}",
                        conflict_type=conflict_type,
                        claims=[claim1, claim2],
                        severity=severity,
                        description=description,
                        resolution_hint=resolution_hint,
                        value_delta=value_delta,
                        temporal_gap_days=temporal_gap,
                    )
                )

    return conflicts


def _generate_conflict_description(
    claim1: Claim,
    claim2: Claim,
    conflict_type: ConflictType,
    value_delta: Optional[float],
    temporal_gap: Optional[int],
) -> str:
    """Generate human-readable conflict description."""
    base = f"{claim1.claim_type}.{claim1.field}: "

    if conflict_type == ConflictType.TEMPORAL:
        return (
            f"{base}Values differ across time ({temporal_gap} days apart). "
            f"'{claim1.value}' vs '{claim2.value}'"
        )

    if conflict_type == ConflictType.CONTRADICTION:
        return f"{base}Contradictory claims: '{claim1.value}' vs '{claim2.value}'"

    if conflict_type == ConflictType.SOURCE_DISAGREEMENT:
        return (
            f"{base}Different sources report different values: "
            f"'{claim1.value}' vs '{claim2.value}'"
        )

    if value_delta is not None:
        pct = value_delta * 100
        return f"{base}Values differ by {pct:.0f}%: {claim1.value} vs {claim2.value}"

    return f"{base}Inconsistent values: '{claim1.value}' vs '{claim2.value}'"


def _generate_resolution_hint(
    conflict_type: ConflictType,
    claim1: Claim,
    claim2: Claim,
    temporal_gap: Optional[int],
) -> str:
    """Generate resolution hint for a conflict."""
    if conflict_type == ConflictType.TEMPORAL:
        # Prefer more recent claim
        if claim1.as_of_date and claim2.as_of_date:
            newer = claim1 if claim1.as_of_date > claim2.as_of_date else claim2
            return f"Consider using more recent value (as of {newer.as_of_date.date()})"
        return "Verify which date is more relevant for the analysis"

    if conflict_type == ConflictType.SOURCE_DISAGREEMENT:
        # Prefer higher confidence
        higher_conf = claim1 if claim1.confidence >= claim2.confidence else claim2
        return f"Prefer claim with higher confidence ({higher_conf.confidence:.0%})"

    if conflict_type == ConflictType.CONTRADICTION:
        return "Requires manual review - claims directly contradict"

    return "Review source documents to determine accurate value"


def cluster_conflicts(conflicts: list[DetectedConflict]) -> list[ConflictCluster]:
    """
    Cluster related conflicts together.

    Conflicts are clustered if they:
    - Share claims
    - Relate to the same (claim_type, field)
    """
    if not conflicts:
        return []

    clusters: list[ConflictCluster] = []
    cluster_counter = 0
    assigned = set()

    # Group by (claim_type, field) first
    by_type_field: dict[tuple[str, str], list[DetectedConflict]] = {}
    for conflict in conflicts:
        if conflict.claims:
            key = (conflict.claims[0].claim_type, conflict.claims[0].field)
            if key not in by_type_field:
                by_type_field[key] = []
            by_type_field[key].append(conflict)

    # Create clusters
    for (claim_type, field_name), type_conflicts in by_type_field.items():
        # Further cluster by shared claims using union-find
        claim_to_conflicts: dict[str, list[DetectedConflict]] = {}
        for conflict in type_conflicts:
            for claim in conflict.claims:
                if claim.claim_id not in claim_to_conflicts:
                    claim_to_conflicts[claim.claim_id] = []
                claim_to_conflicts[claim.claim_id].append(conflict)

        # Build connected components
        visited_conflicts: set[str] = set()

        def dfs(conflict: DetectedConflict, component: list[DetectedConflict]):
            if conflict.conflict_id in visited_conflicts:
                return
            visited_conflicts.add(conflict.conflict_id)
            component.append(conflict)
            for claim in conflict.claims:
                for related in claim_to_conflicts.get(claim.claim_id, []):
                    dfs(related, component)

        for conflict in type_conflicts:
            if conflict.conflict_id not in visited_conflicts:
                component: list[DetectedConflict] = []
                dfs(conflict, component)
                if component:
                    cluster_counter += 1
                    clusters.append(
                        ConflictCluster(
                            cluster_id=f"cluster_{cluster_counter:04d}",
                            conflicts=component,
                            primary_claim_type=claim_type,
                            primary_field=field_name,
                        )
                    )

    return clusters


def select_representative_conflicts(
    clusters: list[ConflictCluster],
    max_per_cluster: int = 1,
) -> list[DetectedConflict]:
    """
    Select representative conflicts from clusters.

    Selects the highest-severity conflict from each cluster.

    Args:
        clusters: Conflict clusters
        max_per_cluster: Maximum conflicts to select per cluster

    Returns:
        Selected representative conflicts
    """
    selected: list[DetectedConflict] = []

    for cluster in clusters:
        # Sort by severity (highest first)
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
        }
        sorted_conflicts = sorted(
            cluster.conflicts, key=lambda c: severity_order[c.severity]
        )
        selected.extend(sorted_conflicts[:max_per_cluster])

    return selected
