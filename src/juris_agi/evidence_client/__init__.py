"""
Evidence API Client for JURIS-AGI.

This module provides a client for fetching bounded working sets of claims
from the Evidence Repository service.

Configuration via environment variables:
- EVIDENCE_API_BASE_URL: Base URL of the Evidence API
- EVIDENCE_API_TOKEN: Optional bearer token for authentication
- EVIDENCE_API_TIMEOUT: Request timeout in seconds (default: 30)

Example usage:

    # Remote mode (requires Evidence API)
    from juris_agi.evidence_client import EvidenceApiClient, ContextConstraints

    async with EvidenceApiClient() as client:
        context = await client.create_context(
            deal_id="healthbridge-2024",
            question="Should we invest in this Series A?",
            constraints=ContextConstraints(max_claims=50, min_confidence=0.7)
        )
        for claim in context.claims:
            print(f"{claim.claim_type}.{claim.field}: {claim.value}")

    # Demo mode (no Evidence API required)
    from juris_agi.evidence_client import EvidenceApiClient

    context = EvidenceApiClient.from_direct_claims(
        deal_id="demo-company",
        claims=[
            {"claim_type": "traction", "field": "arr", "value": 1000000, ...},
            ...
        ]
    )
"""

from .client import EvidenceApiClient, fetch_evidence_context
from .errors import (
    EvidenceAPIError,
    EvidenceAuthenticationError,
    EvidenceAuthorizationError,
    EvidenceConnectionError,
    EvidenceNotFoundError,
    EvidenceRateLimitError,
    EvidenceServerError,
    EvidenceTimeoutError,
    EvidenceUnavailableError,
    EvidenceValidationError,
    RetryConfig,
)
from .types import (
    Claim,
    ClaimPolarity,
    ClaimResponse,
    Citation,
    ConfidenceLevel,
    Conflict,
    ConflictType,
    ContextConstraints,
    ContextRequest,
    ContextResponse,
    ContextSummary,
    EvidenceContext,
)

__all__ = [
    # Client
    "EvidenceApiClient",
    "fetch_evidence_context",
    # Types
    "Claim",
    "ClaimPolarity",
    "ClaimResponse",
    "Citation",
    "ConfidenceLevel",
    "Conflict",
    "ConflictType",
    "ContextConstraints",
    "ContextRequest",
    "ContextResponse",
    "ContextSummary",
    "EvidenceContext",
    # Errors
    "EvidenceAPIError",
    "EvidenceAuthenticationError",
    "EvidenceAuthorizationError",
    "EvidenceConnectionError",
    "EvidenceNotFoundError",
    "EvidenceRateLimitError",
    "EvidenceServerError",
    "EvidenceTimeoutError",
    "EvidenceUnavailableError",
    "EvidenceValidationError",
    "RetryConfig",
]
