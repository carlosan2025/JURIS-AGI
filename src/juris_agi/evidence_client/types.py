"""
Pydantic models for Evidence API contract.

These types define the stable contract between JURIS-AGI and the Evidence Repository.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ClaimPolarity(str, Enum):
    """Polarity of a claim relative to investment thesis."""

    SUPPORTIVE = "supportive"
    RISK = "risk"
    NEUTRAL = "neutral"


class ConfidenceLevel(str, Enum):
    """Confidence level for a claim."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """A citation linking a claim to its source document."""

    citation_id: str = Field(..., description="Unique identifier for the citation")
    claim_id: str = Field(..., description="ID of the claim this citation supports")
    document_id: str = Field(..., description="ID of the source document")
    document_type: str = Field(..., description="Type of document (pitch_deck, financial_model, etc.)")
    locator: str = Field(..., description="Location within document (page, section, line)")
    quote: str = Field(..., description="Exact quoted text from the source")
    extraction_date: datetime = Field(default_factory=datetime.utcnow)


class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""

    t: str = Field(..., description="Time point (e.g., 'Q1 2024', '2024-01', 'Jan 2024')")
    value: float = Field(..., description="Value at this time point")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence for this point")


class Claim(BaseModel):
    """A structured claim extracted from evidence documents."""

    claim_id: str = Field(..., description="Unique identifier for the claim")
    claim_type: str = Field(..., description="Category of claim (traction, team_quality, etc.)")
    field: str = Field(..., description="Specific field within claim type")
    value: Any = Field(..., description="The claim value (can be string, number, list, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    polarity: ClaimPolarity = Field(..., description="Whether claim supports or opposes investment")
    unit: Optional[str] = Field(None, description="Unit of measurement if applicable")
    as_of_date: Optional[datetime] = Field(None, description="Date the claim value is valid for")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    timeseries: Optional[list[TimeSeriesPoint]] = Field(
        None,
        description="Time series data for temporal claims (e.g., quarterly ARR)"
    )

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def has_timeseries(self) -> bool:
        """Check if this claim has time series data."""
        return self.timeseries is not None and len(self.timeseries) > 0


class ConflictType(str, Enum):
    """Type of conflict between claims."""

    CONTRADICTION = "contradiction"  # Claims directly contradict each other
    INCONSISTENCY = "inconsistency"  # Claims are logically inconsistent
    TEMPORAL = "temporal"  # Claims conflict due to different time periods
    SOURCE_DISAGREEMENT = "source_disagreement"  # Different sources report different values


class Conflict(BaseModel):
    """A detected conflict between two or more claims."""

    conflict_id: str = Field(..., description="Unique identifier for the conflict")
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    claim_ids: list[str] = Field(..., min_length=2, description="IDs of conflicting claims")
    description: str = Field(..., description="Human-readable description of the conflict")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score 0-1")
    resolution_hint: Optional[str] = Field(None, description="Suggested resolution approach")


class ContextConstraints(BaseModel):
    """Constraints for building an evidence context."""

    max_claims: int = Field(default=100, ge=1, le=1000, description="Maximum total claims to return")
    per_bucket_cap: int = Field(default=20, ge=1, le=100, description="Max claims per claim_type bucket")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_conflicts: bool = Field(default=True, description="Whether to include detected conflicts")
    include_citations: bool = Field(default=True, description="Whether to include full citations")
    claim_types: Optional[list[str]] = Field(None, description="Filter to specific claim types")
    exclude_claim_types: Optional[list[str]] = Field(None, description="Exclude specific claim types")


class ContextRequest(BaseModel):
    """Request to build an evidence context for a deal/question."""

    deal_id: str = Field(..., description="Identifier for the deal/company")
    question: Optional[str] = Field(None, description="Optional: specific question to focus evidence on")
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)


class ContextSummary(BaseModel):
    """Summary statistics for an evidence context."""

    total_claims: int = Field(..., description="Total number of claims in context")
    claims_by_type: dict[str, int] = Field(default_factory=dict, description="Count by claim_type")
    claims_by_polarity: dict[str, int] = Field(default_factory=dict, description="Count by polarity")
    avg_confidence: float = Field(..., description="Average confidence across claims")
    conflict_count: int = Field(default=0, description="Number of detected conflicts")
    document_count: int = Field(default=0, description="Number of unique source documents")


class EvidenceContext(BaseModel):
    """A bounded working set of claims for a deal and question."""

    context_id: str = Field(..., description="Unique identifier for this context")
    deal_id: str = Field(..., description="Deal/company this context is for")
    question: Optional[str] = Field(None, description="Question this context is focused on")
    claims: list[Claim] = Field(default_factory=list, description="Claims in this context")
    conflicts: list[Conflict] = Field(default_factory=list, description="Detected conflicts")
    citations: list[Citation] = Field(default_factory=list, description="All citations")
    summary: ContextSummary = Field(..., description="Summary statistics")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="When this context expires")


class ContextResponse(BaseModel):
    """Response from POST /context endpoint."""

    context_id: str
    claims: list[Claim]
    conflicts: list[Conflict]
    citations: list[Citation]
    summary: ContextSummary


class ClaimResponse(BaseModel):
    """Response from GET /claims/{claim_id} endpoint."""

    claim: Claim
    related_claims: list[str] = Field(default_factory=list, description="IDs of related claims")
    conflicts: list[Conflict] = Field(default_factory=list, description="Conflicts involving this claim")
