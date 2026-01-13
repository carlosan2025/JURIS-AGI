"""
Claim selection logic for building bounded working sets.

Converts a large EvidenceContext into a high-signal EvidenceWorkingSet
suitable for reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from juris_agi.evidence_client.types import (
    Claim,
    Conflict,
    ContextConstraints,
    EvidenceContext,
)

from .conflicts import (
    ConflictCluster,
    DetectedConflict,
    cluster_conflicts,
    detect_conflicts,
    select_representative_conflicts,
)
from .quality import QualityScore, calculate_quality_score, rank_claims_by_quality


@dataclass
class DroppedStats:
    """Statistics about dropped claims during selection."""

    total_input: int = 0
    total_selected: int = 0
    dropped_by_cap: int = 0
    dropped_by_confidence: int = 0
    dropped_as_duplicate: int = 0
    dropped_by_bucket_cap: int = 0
    claims_by_type_input: dict[str, int] = field(default_factory=dict)
    claims_by_type_selected: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        return self.total_input - self.total_selected

    @property
    def selection_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_selected / self.total_input


@dataclass
class EvidenceWorkingSet:
    """
    A bounded, high-signal working set of claims for reasoning.

    This is the output of the context builder, ready to be consumed
    by the reasoning engine.
    """

    deal_id: str
    question: Optional[str]
    claims_selected: list[Claim]
    conflicts_selected: list[Conflict]
    dropped_stats: DroppedStats
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    constraints_applied: Optional[ContextConstraints] = None
    quality_scores: dict[str, float] = field(default_factory=dict)  # claim_id -> score

    @property
    def total_claims(self) -> int:
        return len(self.claims_selected)

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts_selected) > 0

    def get_claims_by_type(self, claim_type: str) -> list[Claim]:
        """Get all claims of a specific type."""
        return [c for c in self.claims_selected if c.claim_type == claim_type]

    def get_claim_quality(self, claim_id: str) -> Optional[float]:
        """Get quality score for a claim."""
        return self.quality_scores.get(claim_id)


def _compute_claim_signature(claim: Claim) -> str:
    """
    Compute a signature for deduplication.

    Claims are considered duplicates if they have the same:
    - claim_type
    - field
    - normalized value (within tolerance for numerics)
    """
    value_sig = str(claim.value)

    # For numeric values, round to reduce false negatives
    if isinstance(claim.value, float):
        value_sig = f"{claim.value:.2f}"
    elif isinstance(claim.value, int):
        value_sig = str(claim.value)
    elif isinstance(claim.value, list):
        value_sig = ",".join(sorted(str(v) for v in claim.value))

    return f"{claim.claim_type}:{claim.field}:{value_sig}"


def _are_claims_duplicate(
    claim1: Claim,
    claim2: Claim,
    numeric_tolerance: float = 0.05,
) -> bool:
    """
    Check if two claims are near-duplicates.

    Returns True if claims have same type/field and very similar values.
    """
    if claim1.claim_type != claim2.claim_type:
        return False
    if claim1.field != claim2.field:
        return False

    v1, v2 = claim1.value, claim2.value

    # Numeric comparison with tolerance
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if v1 == 0 and v2 == 0:
            return True
        if v1 == 0 or v2 == 0:
            return False
        avg = (abs(v1) + abs(v2)) / 2
        return abs(v1 - v2) / avg <= numeric_tolerance

    # String comparison (case-insensitive)
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.lower().strip() == v2.lower().strip()

    # List comparison
    if isinstance(v1, list) and isinstance(v2, list):
        set1 = set(str(x).lower() for x in v1)
        set2 = set(str(x).lower() for x in v2)
        return set1 == set2

    # Default: exact match
    return v1 == v2


def deduplicate_claims(
    claims: list[tuple[Claim, QualityScore]],
) -> tuple[list[tuple[Claim, QualityScore]], int]:
    """
    Remove near-duplicate claims, keeping the highest quality version.

    Args:
        claims: List of (claim, score) tuples, sorted by quality descending

    Returns:
        Tuple of (deduplicated claims, count of duplicates removed)
    """
    seen_signatures: dict[str, tuple[Claim, QualityScore]] = {}
    result: list[tuple[Claim, QualityScore]] = []
    duplicates_removed = 0

    for claim, score in claims:
        sig = _compute_claim_signature(claim)

        # Check for exact signature match
        if sig in seen_signatures:
            duplicates_removed += 1
            continue

        # Check for near-duplicates against all seen claims
        is_duplicate = False
        for seen_claim, _ in result:
            if _are_claims_duplicate(claim, seen_claim):
                is_duplicate = True
                duplicates_removed += 1
                break

        if not is_duplicate:
            seen_signatures[sig] = (claim, score)
            result.append((claim, score))

    return result, duplicates_removed


def select_claims(
    context: EvidenceContext,
    constraints: Optional[ContextConstraints] = None,
    reference_date: Optional[datetime] = None,
) -> EvidenceWorkingSet:
    """
    Select claims from an EvidenceContext to build a bounded working set.

    Selection process:
    1. Filter by minimum confidence
    2. Score all claims by quality
    3. Deduplicate near-identical claims
    4. Apply per-bucket caps (by claim_type)
    5. Apply global max_claims cap
    6. Detect and include conflicts if requested

    Args:
        context: Input evidence context
        constraints: Selection constraints (defaults to ContextConstraints defaults)
        reference_date: Reference date for recency scoring

    Returns:
        EvidenceWorkingSet with selected claims and stats
    """
    if constraints is None:
        constraints = ContextConstraints()

    stats = DroppedStats(total_input=len(context.claims))

    # Count input by type
    for claim in context.claims:
        stats.claims_by_type_input[claim.claim_type] = (
            stats.claims_by_type_input.get(claim.claim_type, 0) + 1
        )

    # Step 1: Filter by minimum confidence
    filtered_claims = [
        c for c in context.claims if c.confidence >= constraints.min_confidence
    ]
    stats.dropped_by_confidence = len(context.claims) - len(filtered_claims)

    # Step 1b: Filter by claim_types if specified
    if constraints.claim_types:
        allowed_types = set(constraints.claim_types)
        filtered_claims = [c for c in filtered_claims if c.claim_type in allowed_types]

    if constraints.exclude_claim_types:
        excluded_types = set(constraints.exclude_claim_types)
        filtered_claims = [c for c in filtered_claims if c.claim_type not in excluded_types]

    # Step 2: Score and rank by quality
    ranked = rank_claims_by_quality(filtered_claims, reference_date)

    # Step 3: Deduplicate
    deduped, dup_count = deduplicate_claims(ranked)
    stats.dropped_as_duplicate = dup_count

    # Step 4: Apply per-bucket caps
    by_type: dict[str, list[tuple[Claim, QualityScore]]] = {}
    for claim, score in deduped:
        if claim.claim_type not in by_type:
            by_type[claim.claim_type] = []
        by_type[claim.claim_type].append((claim, score))

    capped_claims: list[tuple[Claim, QualityScore]] = []
    for claim_type, type_claims in by_type.items():
        # Already sorted by quality from rank_claims_by_quality
        selected = type_claims[: constraints.per_bucket_cap]
        capped_claims.extend(selected)
        stats.dropped_by_bucket_cap += len(type_claims) - len(selected)

    # Re-sort by quality after combining buckets
    capped_claims.sort(key=lambda x: x[1].composite_score, reverse=True)

    # Step 5: Apply global cap
    if len(capped_claims) > constraints.max_claims:
        stats.dropped_by_cap = len(capped_claims) - constraints.max_claims
        capped_claims = capped_claims[: constraints.max_claims]

    # Extract final claims and scores
    final_claims = [c for c, _ in capped_claims]
    quality_scores = {c.claim_id: s.composite_score for c, s in capped_claims}

    # Count selected by type
    for claim in final_claims:
        stats.claims_by_type_selected[claim.claim_type] = (
            stats.claims_by_type_selected.get(claim.claim_type, 0) + 1
        )

    stats.total_selected = len(final_claims)

    # Step 6: Handle conflicts
    conflicts_selected: list[Conflict] = []

    if constraints.include_conflicts:
        # Detect conflicts among selected claims
        detected = detect_conflicts(final_claims)

        if detected:
            # Cluster and select representatives
            clusters = cluster_conflicts(detected)
            representatives = select_representative_conflicts(clusters, max_per_cluster=1)

            # Convert to API Conflict type
            conflicts_selected = [c.to_conflict() for c in representatives]

            # Ensure claims involved in conflicts are included
            conflict_claim_ids = set()
            for conflict in representatives:
                for claim in conflict.claims:
                    conflict_claim_ids.add(claim.claim_id)

            # Add any conflict claims that might have been dropped
            current_claim_ids = {c.claim_id for c in final_claims}
            missing_conflict_claims = conflict_claim_ids - current_claim_ids

            if missing_conflict_claims:
                # Find and add missing claims (they were filtered/capped but needed for conflict)
                for claim in context.claims:
                    if claim.claim_id in missing_conflict_claims:
                        final_claims.append(claim)
                        quality_scores[claim.claim_id] = calculate_quality_score(
                            claim, reference_date
                        ).composite_score

    return EvidenceWorkingSet(
        deal_id=context.deal_id,
        question=context.question,
        claims_selected=final_claims,
        conflicts_selected=conflicts_selected,
        dropped_stats=stats,
        constraints_applied=constraints,
        quality_scores=quality_scores,
    )


def build_working_set(
    source: EvidenceContext | dict[str, Any],
    constraints: Optional[ContextConstraints] = None,
    reference_date: Optional[datetime] = None,
) -> EvidenceWorkingSet:
    """
    Build an EvidenceWorkingSet from various input sources.

    Accepts:
    - EvidenceContext (from Evidence API or demo mode)
    - Raw dict (for backwards compatibility)

    Args:
        source: Input evidence data
        constraints: Selection constraints
        reference_date: Reference date for recency scoring

    Returns:
        EvidenceWorkingSet ready for reasoning
    """
    if isinstance(source, EvidenceContext):
        return select_claims(source, constraints, reference_date)

    # Handle raw dict input (backwards compatibility)
    if isinstance(source, dict):
        from juris_agi.evidence_client import EvidenceApiClient

        context = EvidenceApiClient.from_direct_claims(
            deal_id=source.get("company_id", "unknown"),
            claims=source.get("claims", []),
            question=None,
        )
        return select_claims(context, constraints, reference_date)

    raise TypeError(f"Unsupported source type: {type(source)}")
