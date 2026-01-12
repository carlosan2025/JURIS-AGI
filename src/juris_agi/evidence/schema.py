"""
Evidence Graph Schema for VC Investment Decisions.

Defines data structures for representing structured evidence claims
with provenance, confidence, and polarity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .ontology import ClaimType, get_claim_type


class Polarity(Enum):
    """Polarity of a claim relative to investment decision."""
    SUPPORTIVE = "supportive"  # Evidence supporting investment
    RISK = "risk"              # Evidence against / risk factor
    NEUTRAL = "neutral"        # Informational, neither supports nor detracts


@dataclass(frozen=True)
class Source:
    """
    Provenance information for a claim.

    Tracks where evidence came from for auditability.
    """
    doc_id: str
    """Unique identifier for the source document."""

    locator: Optional[str] = None
    """Location within document (page, section, timestamp)."""

    quote: Optional[str] = None
    """Verbatim quote from source, if applicable."""

    doc_type: Optional[str] = None
    """Type of document (pitch_deck, data_room, interview, etc.)."""

    retrieved_at: Optional[datetime] = None
    """When the evidence was extracted."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "locator": self.locator,
            "quote": self.quote,
            "doc_type": self.doc_type,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        """Create from dictionary."""
        retrieved_at = None
        if data.get("retrieved_at"):
            retrieved_at = datetime.fromisoformat(data["retrieved_at"])
        return cls(
            doc_id=data["doc_id"],
            locator=data.get("locator"),
            quote=data.get("quote"),
            doc_type=data.get("doc_type"),
            retrieved_at=retrieved_at,
        )


# Type alias for claim values
ClaimValue = Union[str, int, float, bool, List[str], Dict[str, Any]]


@dataclass
class Claim:
    """
    A single evidence claim about a company.

    Claims are the atomic units of the evidence graph.
    Each claim has:
    - A type from the ontology
    - A field name (what aspect is being claimed)
    - A value (the claimed data)
    - Confidence score
    - Polarity (supportive/risk/neutral)
    - Source provenance
    """
    claim_type: ClaimType
    """Which category this claim belongs to."""

    field: str
    """The specific field being claimed (e.g., 'mrr', 'team_size')."""

    value: ClaimValue
    """The claimed value."""

    confidence: float = 1.0
    """Confidence in this claim (0-1). 1.0 = certain, 0.0 = no confidence."""

    polarity: Polarity = Polarity.NEUTRAL
    """Whether this claim supports or detracts from investment thesis."""

    source: Optional[Source] = None
    """Where this claim came from."""

    unit: Optional[str] = None
    """Unit of measurement if applicable (e.g., 'USD', 'months', 'percent')."""

    notes: Optional[str] = None
    """Optional analyst notes or context."""

    claim_id: Optional[str] = None
    """Optional unique identifier for this claim."""

    def __post_init__(self):
        """Validate claim after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def is_high_confidence(self) -> bool:
        """Whether this claim has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def is_risk(self) -> bool:
        """Whether this claim represents a risk factor."""
        return self.polarity == Polarity.RISK

    @property
    def epistemic_uncertainty(self) -> float:
        """
        Epistemic uncertainty of this claim.

        Inverse of confidence - higher uncertainty means less reliable.
        """
        return 1.0 - self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_type": self.claim_type.value,
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "polarity": self.polarity.value,
            "source": self.source.to_dict() if self.source else None,
            "unit": self.unit,
            "notes": self.notes,
            "claim_id": self.claim_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create from dictionary."""
        claim_type = get_claim_type(data["claim_type"])
        if claim_type is None:
            raise ValueError(f"Unknown claim type: {data['claim_type']}")

        polarity = Polarity(data.get("polarity", "neutral"))
        source = Source.from_dict(data["source"]) if data.get("source") else None

        return cls(
            claim_type=claim_type,
            field=data["field"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            polarity=polarity,
            source=source,
            unit=data.get("unit"),
            notes=data.get("notes"),
            claim_id=data.get("claim_id"),
        )


@dataclass
class EvidenceGraph:
    """
    Complete evidence graph for a company.

    Contains all claims organized by type, with metadata about
    coverage and overall confidence.
    """
    company_id: str
    """Unique identifier for the company being evaluated."""

    claims: List[Claim] = field(default_factory=list)
    """All evidence claims."""

    created_at: Optional[datetime] = None
    """When this evidence graph was created."""

    updated_at: Optional[datetime] = None
    """When this evidence graph was last updated."""

    analyst_id: Optional[str] = None
    """Who compiled this evidence."""

    version: str = "1.0"
    """Schema version for compatibility."""

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the graph."""
        self.claims.append(claim)
        self.updated_at = datetime.utcnow()

    def get_claims_by_type(self, claim_type: ClaimType) -> List[Claim]:
        """Get all claims of a specific type."""
        return [c for c in self.claims if c.claim_type == claim_type]

    def get_claims_by_field(self, field: str) -> List[Claim]:
        """Get all claims for a specific field."""
        return [c for c in self.claims if c.field == field]

    def get_risk_claims(self) -> List[Claim]:
        """Get all claims with risk polarity."""
        return [c for c in self.claims if c.polarity == Polarity.RISK]

    def get_supportive_claims(self) -> List[Claim]:
        """Get all claims with supportive polarity."""
        return [c for c in self.claims if c.polarity == Polarity.SUPPORTIVE]

    def get_high_confidence_claims(self, threshold: float = 0.8) -> List[Claim]:
        """Get claims above confidence threshold."""
        return [c for c in self.claims if c.confidence >= threshold]

    def get_low_confidence_claims(self, threshold: float = 0.5) -> List[Claim]:
        """Get claims below confidence threshold."""
        return [c for c in self.claims if c.confidence < threshold]

    @property
    def claim_count(self) -> int:
        """Total number of claims."""
        return len(self.claims)

    @property
    def covered_types(self) -> set:
        """Set of claim types that have at least one claim."""
        return {c.claim_type for c in self.claims}

    @property
    def missing_types(self) -> set:
        """Set of claim types with no claims."""
        from .ontology import get_all_claim_types
        all_types = set(get_all_claim_types())
        return all_types - self.covered_types

    @property
    def coverage_ratio(self) -> float:
        """
        Ratio of covered claim types to total types.

        1.0 = all types covered, 0.0 = none covered.
        """
        from .ontology import get_all_claim_types
        all_types = get_all_claim_types()
        if not all_types:
            return 1.0
        return len(self.covered_types) / len(all_types)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all claims."""
        if not self.claims:
            return 0.0
        return sum(c.confidence for c in self.claims) / len(self.claims)

    @property
    def overall_epistemic_uncertainty(self) -> float:
        """
        Overall epistemic uncertainty considering missing claims.

        Missing claim types contribute uncertainty. Low confidence claims
        also contribute uncertainty.
        """
        if not self.claims:
            return 1.0  # Maximum uncertainty if no evidence

        # Base uncertainty from confidence
        confidence_uncertainty = 1.0 - self.average_confidence

        # Penalty for missing coverage
        coverage_penalty = 1.0 - self.coverage_ratio

        # Combine: both factors contribute to uncertainty
        # Use geometric mean to balance the factors
        return (confidence_uncertainty * 0.6 + coverage_penalty * 0.4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "company_id": self.company_id,
            "claims": [c.to_dict() for c in self.claims],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "analyst_id": self.analyst_id,
            "version": self.version,
            # Computed fields for convenience
            "metadata": {
                "claim_count": self.claim_count,
                "covered_types": [ct.value for ct in self.covered_types],
                "missing_types": [ct.value for ct in self.missing_types],
                "coverage_ratio": self.coverage_ratio,
                "average_confidence": self.average_confidence,
                "epistemic_uncertainty": self.overall_epistemic_uncertainty,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceGraph":
        """Create from dictionary."""
        created_at = None
        updated_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])

        claims = [Claim.from_dict(c) for c in data.get("claims", [])]

        return cls(
            company_id=data["company_id"],
            claims=claims,
            created_at=created_at,
            updated_at=updated_at,
            analyst_id=data.get("analyst_id"),
            version=data.get("version", "1.0"),
        )


@dataclass
class ClaimSummary:
    """
    Summary statistics for claims of a particular type.

    Useful for quick assessment of evidence coverage.
    """
    claim_type: ClaimType
    count: int
    fields: List[str]
    avg_confidence: float
    risk_count: int
    supportive_count: int
    neutral_count: int

    @classmethod
    def from_claims(cls, claim_type: ClaimType, claims: List[Claim]) -> "ClaimSummary":
        """Build summary from list of claims."""
        if not claims:
            return cls(
                claim_type=claim_type,
                count=0,
                fields=[],
                avg_confidence=0.0,
                risk_count=0,
                supportive_count=0,
                neutral_count=0,
            )

        fields = list({c.field for c in claims})
        avg_conf = sum(c.confidence for c in claims) / len(claims)
        risk_count = sum(1 for c in claims if c.polarity == Polarity.RISK)
        supportive_count = sum(1 for c in claims if c.polarity == Polarity.SUPPORTIVE)
        neutral_count = sum(1 for c in claims if c.polarity == Polarity.NEUTRAL)

        return cls(
            claim_type=claim_type,
            count=len(claims),
            fields=fields,
            avg_confidence=avg_conf,
            risk_count=risk_count,
            supportive_count=supportive_count,
            neutral_count=neutral_count,
        )
