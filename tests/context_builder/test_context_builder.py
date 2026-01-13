"""
Unit tests for Context Builder module.

Tests:
- Cap enforcement (per-bucket and global)
- Conflict retention
- Deduplication
- Quality scoring
- Normalization
"""

import pytest
from datetime import datetime, timedelta

from juris_agi.evidence_client import (
    Claim,
    ClaimPolarity,
    EvidenceContext,
    ContextConstraints,
    ContextSummary,
)
from juris_agi.context_builder import (
    build_working_set,
    select_claims,
    EvidenceWorkingSet,
    DroppedStats,
    # Quality
    QualityScore,
    calculate_quality_score,
    rank_claims_by_quality,
    score_recency,
    score_specificity,
    # Conflicts
    detect_conflicts,
    cluster_conflicts,
    select_representative_conflicts,
    ConflictSeverity,
    # Normalization
    normalize_numeric,
    normalize_currency,
    normalize_enum,
    normalize_date,
    # Deduplication
    deduplicate_claims,
)


# =============================================================================
# Fixtures
# =============================================================================


def make_claim(
    claim_id: str,
    claim_type: str,
    field: str,
    value,
    confidence: float = 0.8,
    polarity: str = "neutral",
) -> Claim:
    """Helper to create test claims."""
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        field=field,
        value=value,
        confidence=confidence,
        polarity=ClaimPolarity(polarity),
    )


def make_context(claims: list[Claim], deal_id: str = "test-deal") -> EvidenceContext:
    """Helper to create test context."""
    return EvidenceContext(
        context_id="test-context",
        deal_id=deal_id,
        claims=claims,
        conflicts=[],
        citations=[],
        summary=ContextSummary(
            total_claims=len(claims),
            claims_by_type={},
            claims_by_polarity={},
            avg_confidence=0.8,
        ),
    )


@pytest.fixture
def sample_claims() -> list[Claim]:
    """Sample claims for testing."""
    return [
        make_claim("c1", "traction", "arr", 1000000, 0.9, "supportive"),
        make_claim("c2", "traction", "mrr", 85000, 0.85, "supportive"),
        make_claim("c3", "traction", "growth_rate", 0.15, 0.7, "supportive"),
        make_claim("c4", "team_quality", "founder_background", "Ex-Google", 0.88, "supportive"),
        make_claim("c5", "team_quality", "team_size", 12, 0.9, "neutral"),
        make_claim("c6", "round_terms", "pre_money_valuation", 15000000, 0.95, "neutral"),
        make_claim("c7", "round_terms", "raise_amount", 3000000, 0.95, "neutral"),
        make_claim("c8", "execution_risk", "key_person", "Single founder", 0.75, "risk"),
        make_claim("c9", "market_scope", "tam", 5000000000, 0.6, "supportive"),
        make_claim("c10", "capital_intensity", "monthly_burn", 120000, 0.8, "neutral"),
    ]


# =============================================================================
# Cap Enforcement Tests
# =============================================================================


class TestCapEnforcement:
    """Tests for claim cap enforcement."""

    def test_global_max_claims_enforced(self, sample_claims):
        """Should enforce global max_claims limit."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(max_claims=5)

        result = select_claims(context, constraints)

        assert result.total_claims == 5
        assert result.dropped_stats.total_selected == 5

    def test_per_bucket_cap_enforced(self):
        """Should enforce per-bucket (claim_type) cap."""
        # Create 10 traction claims
        claims = [
            make_claim(f"t{i}", "traction", f"metric_{i}", i * 1000, 0.8)
            for i in range(10)
        ]
        context = make_context(claims)
        constraints = ContextConstraints(per_bucket_cap=3, max_claims=100)

        result = select_claims(context, constraints)

        assert result.total_claims == 3
        assert result.dropped_stats.dropped_by_bucket_cap == 7

    def test_both_caps_applied(self):
        """Should apply both per-bucket and global caps."""
        # Create 5 claims each for 4 types = 20 claims
        claims = []
        for claim_type in ["traction", "team_quality", "round_terms", "market_scope"]:
            for i in range(5):
                claims.append(
                    make_claim(f"{claim_type}_{i}", claim_type, f"field_{i}", i, 0.8)
                )

        context = make_context(claims)
        constraints = ContextConstraints(per_bucket_cap=3, max_claims=10)

        result = select_claims(context, constraints)

        # Per-bucket: 3 * 4 = 12, then global cap: 10
        assert result.total_claims == 10
        assert result.dropped_stats.dropped_by_bucket_cap == 8  # 5-3 = 2 per bucket * 4 = 8

    def test_higher_quality_claims_retained(self, sample_claims):
        """Higher quality claims should be retained when capping."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(max_claims=3)

        result = select_claims(context, constraints)

        # Top 3 by quality should be retained
        retained_ids = {c.claim_id for c in result.claims_selected}
        # High confidence claims should be in there
        assert any(c.confidence >= 0.9 for c in result.claims_selected)


class TestMinConfidenceFilter:
    """Tests for minimum confidence filtering."""

    def test_min_confidence_filter(self, sample_claims):
        """Should filter claims below minimum confidence."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(min_confidence=0.85)

        result = select_claims(context, constraints)

        # Only claims with confidence >= 0.85 should be included
        for claim in result.claims_selected:
            assert claim.confidence >= 0.85

    def test_dropped_by_confidence_tracked(self, sample_claims):
        """Should track claims dropped by confidence filter."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(min_confidence=0.85)

        result = select_claims(context, constraints)

        # Count claims below threshold
        low_conf_count = sum(1 for c in sample_claims if c.confidence < 0.85)
        assert result.dropped_stats.dropped_by_confidence == low_conf_count


# =============================================================================
# Conflict Retention Tests
# =============================================================================


class TestConflictRetention:
    """Tests for conflict detection and retention."""

    def test_conflicts_detected(self):
        """Should detect conflicts between claims."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 500000, 0.85),  # 50% different
        ]
        context = make_context(claims)
        constraints = ContextConstraints(include_conflicts=True)

        result = select_claims(context, constraints)

        assert len(result.conflicts_selected) > 0

    def test_conflict_claims_retained(self):
        """Claims involved in conflicts should be retained."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 500000, 0.85),
            make_claim("c3", "team_quality", "size", 10, 0.7),  # Lower quality
        ]
        context = make_context(claims)
        constraints = ContextConstraints(include_conflicts=True, max_claims=2)

        result = select_claims(context, constraints)

        # Both conflict claims should be retained
        retained_ids = {c.claim_id for c in result.claims_selected}
        assert "c1" in retained_ids
        assert "c2" in retained_ids

    def test_no_conflicts_when_disabled(self):
        """Should not include conflicts when disabled."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 500000, 0.85),
        ]
        context = make_context(claims)
        constraints = ContextConstraints(include_conflicts=False)

        result = select_claims(context, constraints)

        assert len(result.conflicts_selected) == 0

    def test_conflict_clustering(self):
        """Should cluster related conflicts."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 500000, 0.85),
            make_claim("c3", "traction", "arr", 300000, 0.8),  # Third conflicting value
        ]

        conflicts = detect_conflicts(claims)
        clusters = cluster_conflicts(conflicts)

        # All ARR conflicts should be in one cluster
        assert len(clusters) == 1
        assert len(clusters[0].conflicts) >= 1


# =============================================================================
# Deduplication Tests
# =============================================================================


class TestDeduplication:
    """Tests for claim deduplication."""

    def test_exact_duplicates_removed(self):
        """Should remove exact duplicate claims."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 1000000, 0.85),  # Same value, lower confidence
        ]
        context = make_context(claims)

        result = select_claims(context)

        # Only one ARR claim should remain
        arr_claims = [c for c in result.claims_selected if c.field == "arr"]
        assert len(arr_claims) == 1
        # Higher confidence should be kept
        assert arr_claims[0].confidence == 0.9

    def test_near_duplicates_removed(self):
        """Should remove near-duplicate claims (within tolerance)."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 1020000, 0.85),  # 2% different
        ]
        context = make_context(claims)

        result = select_claims(context)

        arr_claims = [c for c in result.claims_selected if c.field == "arr"]
        assert len(arr_claims) == 1
        assert result.dropped_stats.dropped_as_duplicate == 1

    def test_different_values_not_deduplicated(self):
        """Should not deduplicate claims with significantly different values."""
        claims = [
            make_claim("c1", "traction", "arr", 1000000, 0.9),
            make_claim("c2", "traction", "arr", 2000000, 0.85),  # 100% different
        ]
        context = make_context(claims)

        result = select_claims(context)

        arr_claims = [c for c in result.claims_selected if c.field == "arr"]
        assert len(arr_claims) == 2  # Both kept (they're actually a conflict)

    def test_string_duplicates_removed(self):
        """Should remove string duplicates (case-insensitive)."""
        claims = [
            make_claim("c1", "team_quality", "founder", "Ex-Google Engineer", 0.9),
            make_claim("c2", "team_quality", "founder", "ex-google engineer", 0.85),
        ]
        context = make_context(claims)

        result = select_claims(context)

        founder_claims = [c for c in result.claims_selected if c.field == "founder"]
        assert len(founder_claims) == 1


# =============================================================================
# Quality Scoring Tests
# =============================================================================


class TestQualityScoring:
    """Tests for quality scoring."""

    def test_confidence_impacts_score(self):
        """Higher confidence should result in higher score."""
        high_conf = make_claim("c1", "traction", "arr", 1000000, 0.95)
        low_conf = make_claim("c2", "traction", "arr", 1000000, 0.5)

        score_high = calculate_quality_score(high_conf)
        score_low = calculate_quality_score(low_conf)

        assert score_high.composite_score > score_low.composite_score

    def test_recency_scoring(self):
        """Recent claims should score higher."""
        now = datetime.utcnow()

        recent = score_recency(now - timedelta(days=30), now)
        old = score_recency(now - timedelta(days=400), now)
        very_old = score_recency(now - timedelta(days=800), now)

        assert recent > old > very_old

    def test_specificity_scoring(self):
        """Numeric values should score higher than vague text."""
        numeric = score_specificity(1000000)
        specific_text = score_specificity("$1.5M ARR as of Q4 2024")
        vague_text = score_specificity("good")

        assert numeric > specific_text > vague_text

    def test_ranking_order(self, sample_claims):
        """Claims should be ranked by composite score."""
        ranked = rank_claims_by_quality(sample_claims)

        # Verify descending order
        scores = [s.composite_score for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Tests for value normalization."""

    def test_numeric_with_suffix(self):
        """Should normalize numeric values with K/M/B suffixes."""
        assert normalize_numeric("1.5M").normalized == 1_500_000
        assert normalize_numeric("500K").normalized == 500_000
        assert normalize_numeric("2B").normalized == 2_000_000_000

    def test_currency_extraction(self):
        """Should extract currency from values."""
        result = normalize_currency("$1.5M")
        assert result.normalized == 1_500_000
        assert result.unit == "USD"

    def test_percentage_conversion(self):
        """Should convert percentages to ratios."""
        result = normalize_numeric("15%")
        assert result.normalized == 0.15
        assert result.unit == "ratio"

    def test_stage_normalization(self):
        """Should normalize funding stage values."""
        assert normalize_enum("Series A", "stage").normalized == "series_a"
        assert normalize_enum("series-b", "stage").normalized == "series_b"
        assert normalize_enum("Pre-Seed", "stage").normalized == "pre_seed"

    def test_date_normalization(self):
        """Should normalize date values."""
        result = normalize_date("2024-01-15")
        assert isinstance(result.normalized, datetime)
        assert result.normalized.year == 2024
        assert result.normalized.month == 1
        assert result.normalized.day == 15

    def test_quarter_to_date(self):
        """Should convert quarters to dates."""
        result = normalize_date("Q1 2024")
        assert isinstance(result.normalized, datetime)
        assert result.normalized.month == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildWorkingSet:
    """Integration tests for build_working_set."""

    def test_from_evidence_context(self, sample_claims):
        """Should build working set from EvidenceContext."""
        context = make_context(sample_claims)

        result = build_working_set(context)

        assert isinstance(result, EvidenceWorkingSet)
        assert result.deal_id == "test-deal"
        assert result.total_claims > 0

    def test_from_raw_dict(self):
        """Should build working set from raw dict (backwards compat)."""
        raw = {
            "company_id": "demo-company",
            "claims": [
                {
                    "claim_type": "traction",
                    "field": "arr",
                    "value": 1000000,
                    "confidence": 0.9,
                    "polarity": "supportive",
                },
            ],
        }

        result = build_working_set(raw)

        assert isinstance(result, EvidenceWorkingSet)
        assert result.deal_id == "demo-company"
        assert result.total_claims == 1

    def test_quality_scores_included(self, sample_claims):
        """Should include quality scores for selected claims."""
        context = make_context(sample_claims)

        result = build_working_set(context)

        # Every selected claim should have a quality score
        for claim in result.claims_selected:
            assert claim.claim_id in result.quality_scores
            assert 0 <= result.quality_scores[claim.claim_id] <= 1

    def test_dropped_stats_accurate(self, sample_claims):
        """Should accurately track dropped claims."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(max_claims=3)

        result = build_working_set(context, constraints)

        assert result.dropped_stats.total_input == len(sample_claims)
        assert result.dropped_stats.total_selected == 3
        assert result.dropped_stats.total_dropped == len(sample_claims) - 3


class TestClaimTypeFiltering:
    """Tests for claim type filtering."""

    def test_include_specific_types(self, sample_claims):
        """Should only include specified claim types."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(claim_types=["traction", "round_terms"])

        result = select_claims(context, constraints)

        for claim in result.claims_selected:
            assert claim.claim_type in ["traction", "round_terms"]

    def test_exclude_specific_types(self, sample_claims):
        """Should exclude specified claim types."""
        context = make_context(sample_claims)
        constraints = ContextConstraints(exclude_claim_types=["execution_risk"])

        result = select_claims(context, constraints)

        for claim in result.claims_selected:
            assert claim.claim_type != "execution_risk"
