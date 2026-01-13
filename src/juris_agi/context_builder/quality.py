"""
Claim quality scoring for context selection.

Quality factors:
- Confidence: raw confidence score from extraction
- Source type: pitch deck < IC memo < audited financials
- Recency: more recent claims score higher
- Completeness: claims with citations score higher
- Specificity: numeric values score higher than vague text
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from juris_agi.evidence_client.types import Claim, Citation


class SourceQuality(Enum):
    """Source quality tiers."""

    AUDITED = 1.0  # Audited financials, legal docs
    VERIFIED = 0.85  # IC memos with verification notes
    INTERNAL = 0.7  # Internal pitch decks, projections
    EXTERNAL = 0.6  # Third-party reports, market research
    UNVERIFIED = 0.4  # Unverified claims, estimates
    UNKNOWN = 0.5  # No source information


# Mapping of document types to quality tiers
SOURCE_TYPE_QUALITY: dict[str, SourceQuality] = {
    # High quality
    "audited_financials": SourceQuality.AUDITED,
    "legal_document": SourceQuality.AUDITED,
    "term_sheet": SourceQuality.AUDITED,
    "cap_table": SourceQuality.AUDITED,
    # Verified
    "ic_memo": SourceQuality.VERIFIED,
    "due_diligence_report": SourceQuality.VERIFIED,
    "reference_check": SourceQuality.VERIFIED,
    # Internal
    "pitch_deck": SourceQuality.INTERNAL,
    "financial_model": SourceQuality.INTERNAL,
    "tech_description": SourceQuality.INTERNAL,
    "data_room": SourceQuality.INTERNAL,
    # External
    "market_report": SourceQuality.EXTERNAL,
    "news_article": SourceQuality.EXTERNAL,
    "linkedin": SourceQuality.EXTERNAL,
    # Unverified
    "email": SourceQuality.UNVERIFIED,
    "verbal": SourceQuality.UNVERIFIED,
    "estimate": SourceQuality.UNVERIFIED,
}


@dataclass
class QualityScore:
    """Composite quality score for a claim."""

    confidence_score: float  # 0-1, from claim.confidence
    source_score: float  # 0-1, from source type
    recency_score: float  # 0-1, based on as_of_date
    completeness_score: float  # 0-1, based on citations/metadata
    specificity_score: float  # 0-1, based on value type

    @property
    def composite_score(self) -> float:
        """
        Weighted composite score.

        Weights:
        - Confidence: 35% (most important, directly from extraction)
        - Source: 25% (source reliability matters)
        - Recency: 15% (newer data preferred)
        - Completeness: 15% (traceable claims preferred)
        - Specificity: 10% (concrete values preferred)
        """
        return (
            self.confidence_score * 0.35
            + self.source_score * 0.25
            + self.recency_score * 0.15
            + self.completeness_score * 0.15
            + self.specificity_score * 0.10
        )

    def __lt__(self, other: "QualityScore") -> bool:
        return self.composite_score < other.composite_score

    def __gt__(self, other: "QualityScore") -> bool:
        return self.composite_score > other.composite_score


def score_source_quality(citations: list[Citation]) -> float:
    """
    Score source quality based on citation document types.

    Returns the highest quality score among all citations,
    or UNKNOWN if no citations.
    """
    if not citations:
        return SourceQuality.UNKNOWN.value

    best_quality = SourceQuality.UNKNOWN.value
    for citation in citations:
        doc_type = citation.document_type.lower().replace(" ", "_")
        quality = SOURCE_TYPE_QUALITY.get(doc_type, SourceQuality.UNKNOWN)
        best_quality = max(best_quality, quality.value)

    return best_quality


def score_recency(as_of_date: Optional[datetime], reference_date: Optional[datetime] = None) -> float:
    """
    Score recency of a claim.

    Claims decay in value over time:
    - < 3 months: 1.0
    - 3-6 months: 0.9
    - 6-12 months: 0.75
    - 1-2 years: 0.5
    - > 2 years: 0.25
    - No date: 0.5 (neutral)
    """
    if as_of_date is None:
        return 0.5  # Neutral score for undated claims

    if reference_date is None:
        reference_date = datetime.utcnow()

    age = reference_date - as_of_date
    days = age.days

    if days < 0:
        # Future date (projection) - still valuable but slightly discounted
        return 0.85
    elif days <= 90:
        return 1.0
    elif days <= 180:
        return 0.9
    elif days <= 365:
        return 0.75
    elif days <= 730:
        return 0.5
    else:
        return 0.25


def score_completeness(claim: Claim) -> float:
    """
    Score completeness of a claim based on available metadata.

    Factors:
    - Has citations: +0.3
    - Has as_of_date: +0.2
    - Has unit (for numeric values): +0.2
    - Has multiple citations: +0.1
    - Has locator in citation: +0.2
    """
    score = 0.0

    # Citations
    if claim.citations:
        score += 0.3
        if len(claim.citations) > 1:
            score += 0.1
        # Check for locators
        if any(c.locator for c in claim.citations):
            score += 0.2

    # Date
    if claim.as_of_date is not None:
        score += 0.2

    # Unit for numeric values
    if isinstance(claim.value, (int, float)) and claim.unit:
        score += 0.2

    return min(score, 1.0)


def score_specificity(value: Any) -> float:
    """
    Score specificity of a claim value.

    Specific, concrete values are preferred:
    - Numeric values: 1.0
    - Lists with items: 0.9
    - Specific strings (dates, percentages, names): 0.8
    - General descriptive strings: 0.5
    - Very short or vague strings: 0.3
    - None/empty: 0.0
    """
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return 1.0

    if isinstance(value, list):
        if len(value) > 0:
            return 0.9
        return 0.3

    if isinstance(value, str):
        if not value.strip():
            return 0.0

        # Check for specific patterns
        import re

        # Dates, percentages, currency
        if re.search(r"\d{4}[-/]\d{2}|[\d.]+%|\$[\d,]+", value):
            return 0.85

        # Names (capitalized words)
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", value):
            return 0.8

        # Length-based scoring for descriptive text
        length = len(value)
        if length > 100:
            return 0.7  # Detailed description
        elif length > 30:
            return 0.6  # Moderate detail
        elif length > 10:
            return 0.5  # Brief
        else:
            return 0.3  # Very short

    # Other types
    return 0.5


def calculate_quality_score(
    claim: Claim,
    reference_date: Optional[datetime] = None,
) -> QualityScore:
    """
    Calculate comprehensive quality score for a claim.

    Args:
        claim: The claim to score
        reference_date: Reference date for recency scoring (defaults to now)

    Returns:
        QualityScore with individual and composite scores
    """
    return QualityScore(
        confidence_score=claim.confidence,
        source_score=score_source_quality(claim.citations),
        recency_score=score_recency(claim.as_of_date, reference_date),
        completeness_score=score_completeness(claim),
        specificity_score=score_specificity(claim.value),
    )


def rank_claims_by_quality(
    claims: list[Claim],
    reference_date: Optional[datetime] = None,
) -> list[tuple[Claim, QualityScore]]:
    """
    Rank claims by quality score (highest first).

    Args:
        claims: Claims to rank
        reference_date: Reference date for recency scoring

    Returns:
        List of (claim, score) tuples sorted by composite score descending
    """
    scored = [(claim, calculate_quality_score(claim, reference_date)) for claim in claims]
    scored.sort(key=lambda x: x[1].composite_score, reverse=True)
    return scored
