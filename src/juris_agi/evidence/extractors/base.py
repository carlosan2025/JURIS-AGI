"""
Base classes for LLM-based document extraction.

Provides the abstract interface and common functionality for all
document extractors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Type
import json
import uuid

from ..schema import Claim, ClaimValue, Source, Polarity
from ..ontology import ClaimType, get_claim_type


class ExtractionStatus(Enum):
    """Status of a proposed claim."""
    PENDING = "pending"        # Awaiting human review
    APPROVED = "approved"      # Human approved, ready to merge
    REJECTED = "rejected"      # Human rejected
    MODIFIED = "modified"      # Human modified and approved
    MERGED = "merged"          # Already merged into evidence graph


@dataclass
class ProposedClaim:
    """
    A claim proposed by LLM extraction, pending human review.

    ProposedClaims are NOT part of the evidence graph until explicitly
    approved by a human reviewer.
    """
    proposal_id: str
    """Unique identifier for this proposal."""

    claim_type: ClaimType
    """Category of the claim."""

    field: str
    """Field name being claimed."""

    value: ClaimValue
    """Proposed value."""

    confidence: float
    """LLM's confidence in extraction (0-1)."""

    polarity: Polarity
    """Proposed polarity (supportive/risk/neutral)."""

    source: Source
    """Source document and locator."""

    rationale: str
    """LLM's explanation for this extraction."""

    status: ExtractionStatus = ExtractionStatus.PENDING
    """Current review status."""

    reviewer_notes: Optional[str] = None
    """Notes from human reviewer."""

    modified_value: Optional[ClaimValue] = None
    """Human-modified value, if status is MODIFIED."""

    modified_confidence: Optional[float] = None
    """Human-modified confidence, if status is MODIFIED."""

    modified_polarity: Optional[Polarity] = None
    """Human-modified polarity, if status is MODIFIED."""

    created_at: Optional[datetime] = None
    """When extraction was performed."""

    reviewed_at: Optional[datetime] = None
    """When human review occurred."""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def approve(self, reviewer_notes: Optional[str] = None) -> None:
        """Mark as approved by human reviewer."""
        self.status = ExtractionStatus.APPROVED
        self.reviewer_notes = reviewer_notes
        self.reviewed_at = datetime.utcnow()

    def reject(self, reviewer_notes: Optional[str] = None) -> None:
        """Mark as rejected by human reviewer."""
        self.status = ExtractionStatus.REJECTED
        self.reviewer_notes = reviewer_notes
        self.reviewed_at = datetime.utcnow()

    def modify(
        self,
        value: Optional[ClaimValue] = None,
        confidence: Optional[float] = None,
        polarity: Optional[Polarity] = None,
        reviewer_notes: Optional[str] = None,
    ) -> None:
        """Modify and approve the claim."""
        self.status = ExtractionStatus.MODIFIED
        if value is not None:
            self.modified_value = value
        if confidence is not None:
            self.modified_confidence = confidence
        if polarity is not None:
            self.modified_polarity = polarity
        self.reviewer_notes = reviewer_notes
        self.reviewed_at = datetime.utcnow()

    def to_claim(self) -> Claim:
        """
        Convert approved proposal to a Claim.

        Uses modified values if status is MODIFIED.

        Raises:
            ValueError: If claim is not approved or modified.
        """
        if self.status not in (ExtractionStatus.APPROVED, ExtractionStatus.MODIFIED):
            raise ValueError(f"Cannot convert {self.status.value} proposal to claim")

        value = self.modified_value if self.modified_value is not None else self.value
        confidence = self.modified_confidence if self.modified_confidence is not None else self.confidence
        polarity = self.modified_polarity if self.modified_polarity is not None else self.polarity

        return Claim(
            claim_type=self.claim_type,
            field=self.field,
            value=value,
            confidence=confidence,
            polarity=polarity,
            source=self.source,
            notes=f"Extracted by LLM. Rationale: {self.rationale}",
            claim_id=self.proposal_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "proposal_id": self.proposal_id,
            "claim_type": self.claim_type.value,
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "polarity": self.polarity.value,
            "source": self.source.to_dict() if self.source else None,
            "rationale": self.rationale,
            "status": self.status.value,
            "reviewer_notes": self.reviewer_notes,
            "modified_value": self.modified_value,
            "modified_confidence": self.modified_confidence,
            "modified_polarity": self.modified_polarity.value if self.modified_polarity else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposedClaim":
        """Create from dictionary."""
        claim_type = get_claim_type(data["claim_type"])
        if claim_type is None:
            raise ValueError(f"Unknown claim type: {data['claim_type']}")

        polarity = Polarity(data.get("polarity", "neutral"))
        source = Source.from_dict(data["source"]) if data.get("source") else None

        modified_polarity = None
        if data.get("modified_polarity"):
            modified_polarity = Polarity(data["modified_polarity"])

        created_at = None
        reviewed_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        if data.get("reviewed_at"):
            reviewed_at = datetime.fromisoformat(data["reviewed_at"])

        return cls(
            proposal_id=data["proposal_id"],
            claim_type=claim_type,
            field=data["field"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            polarity=polarity,
            source=source,
            rationale=data.get("rationale", ""),
            status=ExtractionStatus(data.get("status", "pending")),
            reviewer_notes=data.get("reviewer_notes"),
            modified_value=data.get("modified_value"),
            modified_confidence=data.get("modified_confidence"),
            modified_polarity=modified_polarity,
            created_at=created_at,
            reviewed_at=reviewed_at,
        )


@dataclass
class ExtractionConfig:
    """Configuration for document extraction."""

    enabled: bool = True
    """Whether LLM extraction is enabled."""

    model: str = "gpt-4o"
    """LLM model to use for extraction."""

    temperature: float = 0.1
    """Temperature for LLM calls (lower = more deterministic)."""

    max_claims_per_type: int = 10
    """Maximum claims to extract per claim type."""

    min_confidence: float = 0.3
    """Minimum confidence threshold for proposals."""

    include_rationale: bool = True
    """Whether to include extraction rationale."""

    chunk_size: int = 4000
    """Maximum characters per chunk when splitting documents."""

    overlap: int = 200
    """Overlap between chunks."""

    timeout_seconds: int = 60
    """Timeout for LLM calls."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "temperature": self.temperature,
            "max_claims_per_type": self.max_claims_per_type,
            "min_confidence": self.min_confidence,
            "include_rationale": self.include_rationale,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ExtractionResult:
    """Result of document extraction."""

    doc_id: str
    """Document identifier."""

    doc_type: str
    """Type of document extracted."""

    proposed_claims: List[ProposedClaim] = field(default_factory=list)
    """Proposed claims from extraction."""

    extraction_time_seconds: float = 0.0
    """Time taken for extraction."""

    model_used: str = ""
    """LLM model used."""

    tokens_used: int = 0
    """Total tokens consumed."""

    errors: List[str] = field(default_factory=list)
    """Any errors encountered during extraction."""

    created_at: Optional[datetime] = None
    """When extraction was performed."""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def success(self) -> bool:
        """Whether extraction completed without errors."""
        return len(self.errors) == 0

    @property
    def pending_count(self) -> int:
        """Number of claims pending review."""
        return sum(1 for c in self.proposed_claims if c.status == ExtractionStatus.PENDING)

    @property
    def approved_count(self) -> int:
        """Number of approved claims."""
        return sum(1 for c in self.proposed_claims
                   if c.status in (ExtractionStatus.APPROVED, ExtractionStatus.MODIFIED))

    @property
    def rejected_count(self) -> int:
        """Number of rejected claims."""
        return sum(1 for c in self.proposed_claims if c.status == ExtractionStatus.REJECTED)

    def get_approved_claims(self) -> List[Claim]:
        """Get all approved claims converted to Claim objects."""
        return [
            pc.to_claim()
            for pc in self.proposed_claims
            if pc.status in (ExtractionStatus.APPROVED, ExtractionStatus.MODIFIED)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "proposed_claims": [c.to_dict() for c in self.proposed_claims],
            "extraction_time_seconds": self.extraction_time_seconds,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "errors": self.errors,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "summary": {
                "total": len(self.proposed_claims),
                "pending": self.pending_count,
                "approved": self.approved_count,
                "rejected": self.rejected_count,
            },
        }


# Type alias for LLM call function
LLMCallFn = Callable[[str, str], str]


class DocumentExtractor(ABC):
    """
    Abstract base class for document extractors.

    Each subclass handles a specific document type and provides
    appropriate prompts for extraction.
    """

    # Subclasses must define these
    DOC_TYPE: str = ""
    SUPPORTED_CLAIM_TYPES: List[ClaimType] = []

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        llm_fn: Optional[LLMCallFn] = None,
    ):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration
            llm_fn: Function to call LLM. Signature: (system_prompt, user_prompt) -> response
                   If None, extraction is disabled.
        """
        self.config = config or ExtractionConfig()
        self.llm_fn = llm_fn

    @property
    def enabled(self) -> bool:
        """Whether extraction is enabled and LLM is available."""
        return self.config.enabled and self.llm_fn is not None

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for LLM extraction."""
        pass

    @abstractmethod
    def get_extraction_prompt(self, document_text: str, doc_id: str) -> str:
        """
        Get extraction prompt for a specific document.

        Args:
            document_text: Text content of the document
            doc_id: Document identifier

        Returns:
            User prompt for LLM
        """
        pass

    def get_claim_types_description(self) -> str:
        """Get description of claim types this extractor handles."""
        from ..ontology import CLAIM_TYPE_METADATA

        descriptions = []
        for ct in self.SUPPORTED_CLAIM_TYPES:
            info = CLAIM_TYPE_METADATA.get(ct)
            if info:
                fields = ", ".join(info.typical_fields)
                descriptions.append(
                    f"- **{ct.value}**: {info.description}\n  Typical fields: {fields}"
                )
        return "\n".join(descriptions)

    def parse_llm_response(
        self,
        response: str,
        doc_id: str,
    ) -> List[ProposedClaim]:
        """
        Parse LLM response into proposed claims.

        Args:
            response: Raw LLM response
            doc_id: Document identifier for source

        Returns:
            List of proposed claims
        """
        proposals = []

        # Try to parse as JSON
        try:
            # Find JSON array in response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                return proposals

            json_str = response[start_idx:end_idx]
            claims_data = json.loads(json_str)

            for claim_data in claims_data:
                try:
                    proposal = self._parse_claim_data(claim_data, doc_id)
                    if proposal and proposal.confidence >= self.config.min_confidence:
                        proposals.append(proposal)
                except (KeyError, ValueError) as e:
                    continue

        except json.JSONDecodeError:
            pass

        return proposals

    def _parse_claim_data(
        self,
        data: Dict[str, Any],
        doc_id: str,
    ) -> Optional[ProposedClaim]:
        """Parse a single claim from LLM output."""
        claim_type = get_claim_type(data.get("claim_type", ""))
        if claim_type is None or claim_type not in self.SUPPORTED_CLAIM_TYPES:
            return None

        field = data.get("field", "").strip()
        if not field:
            return None

        value = data.get("value")
        if value is None:
            return None

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        polarity_str = data.get("polarity", "neutral").lower()
        try:
            polarity = Polarity(polarity_str)
        except ValueError:
            polarity = Polarity.NEUTRAL

        source = Source(
            doc_id=doc_id,
            locator=data.get("locator"),
            quote=data.get("quote"),
            doc_type=self.DOC_TYPE,
            retrieved_at=datetime.utcnow(),
        )

        return ProposedClaim(
            proposal_id=str(uuid.uuid4()),
            claim_type=claim_type,
            field=field,
            value=value,
            confidence=confidence,
            polarity=polarity,
            source=source,
            rationale=data.get("rationale", ""),
        )

    def extract(
        self,
        document_text: str,
        doc_id: str,
    ) -> ExtractionResult:
        """
        Extract claims from document.

        Args:
            document_text: Text content of the document
            doc_id: Document identifier

        Returns:
            ExtractionResult with proposed claims
        """
        result = ExtractionResult(
            doc_id=doc_id,
            doc_type=self.DOC_TYPE,
            model_used=self.config.model,
        )

        if not self.enabled:
            result.errors.append("Extraction is disabled or LLM not configured")
            return result

        import time
        start_time = time.time()

        try:
            system_prompt = self.get_system_prompt()
            user_prompt = self.get_extraction_prompt(document_text, doc_id)

            response = self.llm_fn(system_prompt, user_prompt)

            proposals = self.parse_llm_response(response, doc_id)

            # Limit proposals per claim type
            from collections import defaultdict
            by_type = defaultdict(list)
            for p in proposals:
                by_type[p.claim_type].append(p)

            limited_proposals = []
            for ct, claim_list in by_type.items():
                # Sort by confidence and take top N
                sorted_claims = sorted(claim_list, key=lambda x: x.confidence, reverse=True)
                limited_proposals.extend(sorted_claims[:self.config.max_claims_per_type])

            result.proposed_claims = limited_proposals

        except Exception as e:
            result.errors.append(f"Extraction failed: {str(e)}")

        result.extraction_time_seconds = time.time() - start_time
        return result

    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into chunks for processing.

        Args:
            text: Full document text

        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at paragraph boundary
            if end < len(text):
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.config.chunk_size // 2:
                    end = para_break + 2

            chunks.append(text[start:end])
            start = end - self.config.overlap

        return chunks
