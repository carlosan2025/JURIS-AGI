"""
Context Builder for JURIS-AGI.

Converts large EvidenceContext into bounded, high-signal working sets
suitable for neuro-symbolic reasoning.

Example usage:

    from juris_agi.context_builder import build_working_set, ContextConstraints
    from juris_agi.evidence_client import EvidenceApiClient

    # From Evidence API
    async with EvidenceApiClient() as client:
        context = await client.create_context(deal_id="healthbridge-2024")

    working_set = build_working_set(
        context,
        constraints=ContextConstraints(
            max_claims=50,
            per_bucket_cap=10,
            min_confidence=0.6,
            include_conflicts=True,
        )
    )

    print(f"Selected {working_set.total_claims} claims")
    print(f"Dropped {working_set.dropped_stats.total_dropped} claims")

    # From raw claims (demo mode)
    working_set = build_working_set({
        "company_id": "demo-company",
        "claims": [...]
    })
"""

from juris_agi.evidence_client.types import ContextConstraints

from .conflicts import (
    ConflictCluster,
    ConflictSeverity,
    DetectedConflict,
    cluster_conflicts,
    detect_conflicts,
    select_representative_conflicts,
)
from .normalization import (
    NormalizedValue,
    normalize_claim_value,
    normalize_currency,
    normalize_date,
    normalize_enum,
    normalize_numeric,
)
from .quality import (
    QualityScore,
    SourceQuality,
    calculate_quality_score,
    rank_claims_by_quality,
    score_completeness,
    score_recency,
    score_source_quality,
    score_specificity,
)
from .select_claims import (
    DroppedStats,
    EvidenceWorkingSet,
    build_working_set,
    deduplicate_claims,
    select_claims,
)

__all__ = [
    # Main entry points
    "build_working_set",
    "select_claims",
    # Types
    "EvidenceWorkingSet",
    "DroppedStats",
    "ContextConstraints",
    # Quality scoring
    "QualityScore",
    "SourceQuality",
    "calculate_quality_score",
    "rank_claims_by_quality",
    "score_completeness",
    "score_recency",
    "score_source_quality",
    "score_specificity",
    # Conflict detection
    "DetectedConflict",
    "ConflictCluster",
    "ConflictSeverity",
    "detect_conflicts",
    "cluster_conflicts",
    "select_representative_conflicts",
    # Normalization
    "NormalizedValue",
    "normalize_claim_value",
    "normalize_currency",
    "normalize_date",
    "normalize_enum",
    "normalize_numeric",
    # Deduplication
    "deduplicate_claims",
]
