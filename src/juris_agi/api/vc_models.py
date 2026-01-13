"""
Pydantic models for VC Decision API request/response schemas.

These models support the end-to-end VC reasoning workflow:
1. Accept deal_id + question (fetch from Evidence API) or direct claims
2. Return policies, traces, and uncertainty analysis
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class VCJobStatus(str, Enum):
    """Status of a VC solve job."""

    PENDING = "pending"
    FETCHING_CONTEXT = "fetching_context"
    BUILDING_WORKING_SET = "building_working_set"
    PROPOSING_THRESHOLDS = "proposing_thresholds"
    LEARNING_POLICIES = "learning_policies"
    ANALYZING_UNCERTAINTY = "analyzing_uncertainty"
    COMPLETED = "completed"
    FAILED = "failed"


class VCConstraints(BaseModel):
    """Constraints for VC reasoning."""

    max_claims: int = Field(default=100, ge=1, le=1000, description="Maximum claims to consider")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum claim confidence")
    require_sources: Optional[list[str]] = Field(
        None, description="Require claims from these source types"
    )
    focus_claim_types: Optional[list[str]] = Field(
        None, description="Focus on specific claim types (e.g., ['traction', 'team_quality'])"
    )
    max_policies: int = Field(default=5, ge=1, le=20, description="Maximum policy hypotheses to return")
    min_coverage: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum coverage threshold")


class DirectClaim(BaseModel):
    """A claim passed directly (for demo mode without Evidence API)."""

    claim_type: str = Field(..., description="Category (traction, team_quality, market, etc.)")
    field: str = Field(..., description="Specific field (arr, growth_rate, founder_experience, etc.)")
    value: Any = Field(..., description="The claim value")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    polarity: str = Field(default="neutral", description="supportive, risk, or neutral")
    source_type: Optional[str] = Field(None, description="Source document type")
    timeseries: Optional[list[dict[str, Any]]] = Field(
        None, description="Time series data [{t: 'Q1 2024', value: 100}, ...]"
    )


class VCSolveRequest(BaseModel):
    """
    Request to analyze a VC investment decision.

    Supports two modes:
    A) Remote mode: Provide deal_id + question (fetches from Evidence API)
    B) Demo mode: Provide claims directly
    """

    # Mode A: Remote - fetch from Evidence API
    deal_id: Optional[str] = Field(None, description="Deal identifier (for Evidence API lookup)")
    question: Optional[str] = Field(
        None, description="Investment question (e.g., 'Should we invest in Series A?')"
    )

    # Mode B: Demo - direct claims
    claims: Optional[list[DirectClaim]] = Field(
        None, description="Direct claims (alternative to deal_id for demo mode)"
    )

    # Common options
    constraints: VCConstraints = Field(default_factory=VCConstraints, description="Processing constraints")
    include_trace: bool = Field(default=True, description="Include detailed trace data")
    include_events: bool = Field(default=True, description="Include processing events")

    # Historical decisions for policy learning (optional)
    historical_decisions: Optional[list[dict[str, Any]]] = Field(
        None, description="Historical decisions for policy learning [{claims: [...], decision: 'invest', ...}]"
    )

    def validate_request(self) -> None:
        """Validate that the request has either deal_id or claims."""
        if not self.deal_id and not self.claims:
            raise ValueError("Either deal_id or claims must be provided")

    class Config:
        json_schema_extra = {
            "example": {
                "deal_id": "acme-corp-2024",
                "question": "Should we invest in Series A?",
                "constraints": {
                    "min_confidence": 0.6,
                    "max_policies": 3,
                },
            }
        }


class WorkingSetSummary(BaseModel):
    """Summary of the evidence working set."""

    total_claims: int = Field(..., description="Total claims in working set")
    claims_by_type: dict[str, int] = Field(default_factory=dict, description="Claims per type")
    claims_by_polarity: dict[str, int] = Field(default_factory=dict, description="Claims per polarity")
    avg_confidence: float = Field(..., description="Average confidence")
    timeseries_fields: list[str] = Field(default_factory=list, description="Fields with time series data")
    conflict_count: int = Field(default=0, description="Number of conflicts detected")


class PolicyRule(BaseModel):
    """A rule in a policy."""

    rule_id: str
    name: str
    predicate_dsl: str = Field(..., description="DSL representation of the rule predicate")
    decision: str = Field(..., description="invest, pass, or defer")
    priority: int
    fields_used: list[str] = Field(default_factory=list)


class PolicyOutput(BaseModel):
    """A policy hypothesis output."""

    policy_id: str
    rules: list[PolicyRule] = Field(default_factory=list)
    mdl_score: float = Field(..., description="MDL (Minimum Description Length) score")
    coverage: float = Field(..., description="Coverage of training examples")
    exception_count: int = Field(default=0, description="Number of exceptions")
    partition: Optional[str] = Field(None, description="Partition (e.g., 'sector:biotech')")


class UncertaintyOutput(BaseModel):
    """Uncertainty analysis output."""

    epistemic_score: float = Field(..., description="Model uncertainty (0-1)")
    aleatoric_score: float = Field(..., description="Data uncertainty (0-1)")
    total_uncertainty: float = Field(..., description="Combined uncertainty")
    uncertainty_level: str = Field(..., description="low, medium, high, or critical")
    top_reasons: list[str] = Field(default_factory=list, description="Top uncertainty reasons")
    information_requests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Suggested information to gather [{'field': ..., 'reason': ..., 'importance': ...}]",
    )
    should_defer: bool = Field(default=False, description="Whether to defer decision due to uncertainty")


class DecisionOutput(BaseModel):
    """Final decision output."""

    decision: str = Field(..., description="invest, pass, or defer")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Human-readable explanation")
    supporting_rules: list[str] = Field(default_factory=list, description="Rules that fired for this decision")
    blocking_rules: list[str] = Field(default_factory=list, description="Rules that blocked alternatives")


class VCSolveResponse(BaseModel):
    """Response after submitting a VC solve request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: VCJobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    context_id: Optional[str] = Field(None, description="Evidence context ID (if fetched)")
    trace_url: Optional[str] = Field(None, description="URL to detailed trace")
    events_url: Optional[str] = Field(None, description="URL to job events stream")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "vcjob_abc123",
                "status": "pending",
                "created_at": "2024-01-15T10:30:00Z",
                "context_id": "ctx_xyz789",
                "trace_url": "/vc/jobs/vcjob_abc123/trace",
                "events_url": "/vc/jobs/vcjob_abc123/events",
            }
        }


class VCJobEvent(BaseModel):
    """A job processing event."""

    event_id: str
    event_type: str = Field(
        ...,
        description="context_fetched, working_set_built, thresholds_proposed, policies_learned, uncertainty_analyzed",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = Field(None, description="Human-readable message")


class VCJobResult(BaseModel):
    """Complete result of a VC solve job."""

    job_id: str
    status: VCJobStatus
    deal_id: Optional[str] = None
    question: Optional[str] = None

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runtime_seconds: Optional[float] = None

    # Context
    context_id: Optional[str] = None
    working_set: Optional[WorkingSetSummary] = None

    # Results
    decision: Optional[DecisionOutput] = None
    policies: list[PolicyOutput] = Field(default_factory=list)
    uncertainty: Optional[UncertaintyOutput] = None

    # Events
    events: list[VCJobEvent] = Field(default_factory=list)

    # Error info
    error_message: Optional[str] = None

    # Artifacts
    trace_url: Optional[str] = None
    events_url: Optional[str] = None
    trace_data: Optional[dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "vcjob_abc123",
                "status": "completed",
                "deal_id": "acme-corp-2024",
                "question": "Should we invest in Series A?",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:05Z",
                "runtime_seconds": 5.2,
                "context_id": "ctx_xyz789",
                "working_set": {
                    "total_claims": 42,
                    "claims_by_type": {"traction": 15, "team_quality": 10, "market": 8, "financial": 9},
                    "avg_confidence": 0.78,
                },
                "decision": {
                    "decision": "invest",
                    "confidence": 0.85,
                    "explanation": "Strong ARR growth ($2.5M) with experienced team in growing market.",
                    "supporting_rules": ["high_arr", "strong_growth"],
                },
                "policies": [
                    {
                        "policy_id": "policy_1",
                        "rules": [{"rule_id": "r1", "name": "High ARR", "predicate_dsl": "ge(traction.arr, 1000000)"}],
                        "mdl_score": 12.5,
                        "coverage": 0.92,
                    }
                ],
                "uncertainty": {
                    "epistemic_score": 0.15,
                    "aleatoric_score": 0.22,
                    "total_uncertainty": 0.18,
                    "uncertainty_level": "low",
                    "should_defer": False,
                },
            }
        }
