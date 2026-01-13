"""
Evidence API Client for JURIS-AGI.

This client fetches bounded working sets of claims from the Evidence Repository.
It supports both sync and async operations, with retry/backoff logic.
"""

import logging
import os
from typing import Any, Optional

import httpx

from .errors import (
    EvidenceAPIError,
    EvidenceConnectionError,
    EvidenceNotFoundError,
    EvidenceRateLimitError,
    EvidenceTimeoutError,
    EvidenceUnavailableError,
    RetryConfig,
    classify_http_error,
    with_retry_async,
)
from .types import (
    Claim,
    ClaimResponse,
    ContextConstraints,
    ContextRequest,
    ContextResponse,
    EvidenceContext,
)

logger = logging.getLogger(__name__)


class EvidenceApiClient:
    """
    Client for interacting with the Evidence Repository API.

    Configuration via environment variables:
    - EVIDENCE_API_BASE_URL: Base URL of the Evidence API (required for remote mode)
    - EVIDENCE_API_TOKEN: Optional bearer token for authentication
    - EVIDENCE_API_TIMEOUT: Request timeout in seconds (default: 30)

    The client supports a "demo mode" where claims can be passed directly
    without requiring a running Evidence API.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the Evidence API client.

        Args:
            base_url: Base URL for the Evidence API. Defaults to EVIDENCE_API_BASE_URL env var.
            token: Bearer token for auth. Defaults to EVIDENCE_API_TOKEN env var.
            timeout: Request timeout in seconds.
            retry_config: Configuration for retry behavior.
        """
        self.base_url = base_url or os.getenv("EVIDENCE_API_BASE_URL")
        self.token = token or os.getenv("EVIDENCE_API_TOKEN")
        self.timeout = float(os.getenv("EVIDENCE_API_TIMEOUT", str(timeout)))
        self.retry_config = retry_config or RetryConfig()

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured with a base URL."""
        return self.base_url is not None and len(self.base_url) > 0

    def _get_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "EvidenceApiClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response and raise appropriate errors."""
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise EvidenceRateLimitError(
                "Rate limit exceeded",
                status_code=429,
                retry_after=float(retry_after) if retry_after else None,
            )

        if not response.is_success:
            try:
                error_body = response.json()
                message = error_body.get("detail", error_body.get("message", response.text))
            except Exception:
                message = response.text or f"HTTP {response.status_code}"

            raise classify_http_error(response.status_code, message, {"url": str(response.url)})

        return response.json()

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        if not self.is_configured:
            raise EvidenceUnavailableError(
                "Evidence API is not configured. Set EVIDENCE_API_BASE_URL environment variable."
            )

        async def _do_request() -> dict[str, Any]:
            try:
                client = await self._get_client()
                response = await client.request(method, path, json=json, params=params)
                return self._handle_response(response)
            except httpx.ConnectError as e:
                raise EvidenceConnectionError(f"Failed to connect to Evidence API: {e}")
            except httpx.TimeoutException as e:
                raise EvidenceTimeoutError(f"Request timed out: {e}")
            except httpx.HTTPError as e:
                raise EvidenceAPIError(f"HTTP error: {e}")

        return await with_retry_async(_do_request, self.retry_config)

    # =========================================================================
    # Evidence API Contract
    # =========================================================================

    async def create_context(
        self,
        deal_id: str,
        question: Optional[str] = None,
        constraints: Optional[ContextConstraints] = None,
    ) -> ContextResponse:
        """
        Create an evidence context for a deal and optional question.

        POST /context
        Request: {deal_id, question, constraints}
        Response: {context_id, claims[], conflicts[], citations[], summary}

        Args:
            deal_id: Identifier for the deal/company
            question: Optional question to focus the evidence retrieval
            constraints: Constraints for building the context

        Returns:
            ContextResponse with claims, conflicts, and summary
        """
        request = ContextRequest(
            deal_id=deal_id,
            question=question,
            constraints=constraints or ContextConstraints(),
        )

        data = await self._request("POST", "/context", json=request.model_dump())
        return ContextResponse.model_validate(data)

    async def get_context(self, context_id: str) -> EvidenceContext:
        """
        Retrieve a previously created evidence context.

        GET /context/{context_id}

        Args:
            context_id: ID of the context to retrieve

        Returns:
            EvidenceContext with full details
        """
        data = await self._request("GET", f"/context/{context_id}")
        return EvidenceContext.model_validate(data)

    async def get_claim(self, claim_id: str) -> ClaimResponse:
        """
        Retrieve a specific claim by ID.

        GET /claims/{claim_id}

        Args:
            claim_id: ID of the claim to retrieve

        Returns:
            ClaimResponse with claim details and related info
        """
        data = await self._request("GET", f"/claims/{claim_id}")
        return ClaimResponse.model_validate(data)

    # =========================================================================
    # Demo Mode Support
    # =========================================================================

    @staticmethod
    def from_direct_claims(
        deal_id: str,
        claims: list[dict[str, Any]],
        question: Optional[str] = None,
    ) -> EvidenceContext:
        """
        Create an EvidenceContext directly from claim data (demo mode).

        This allows JURIS to work without a running Evidence API by passing
        claims directly (e.g., from demo data or tests).

        Args:
            deal_id: Identifier for the deal
            claims: List of claim dictionaries
            question: Optional question context

        Returns:
            EvidenceContext built from the provided claims
        """
        from datetime import datetime
        import uuid

        # Convert raw claims to Claim objects
        parsed_claims = []
        claims_by_type: dict[str, int] = {}
        claims_by_polarity: dict[str, int] = {}
        total_confidence = 0.0

        for claim_data in claims:
            # Generate claim_id if not present
            if "claim_id" not in claim_data:
                claim_data["claim_id"] = f"claim_{uuid.uuid4().hex[:8]}"

            claim = Claim.model_validate(claim_data)
            parsed_claims.append(claim)

            # Update counts
            claims_by_type[claim.claim_type] = claims_by_type.get(claim.claim_type, 0) + 1
            claims_by_polarity[claim.polarity.value] = claims_by_polarity.get(claim.polarity.value, 0) + 1
            total_confidence += claim.confidence

        # Build summary
        from .types import ContextSummary

        summary = ContextSummary(
            total_claims=len(parsed_claims),
            claims_by_type=claims_by_type,
            claims_by_polarity=claims_by_polarity,
            avg_confidence=total_confidence / len(parsed_claims) if parsed_claims else 0.0,
            conflict_count=0,
            document_count=0,
        )

        return EvidenceContext(
            context_id=f"demo_{uuid.uuid4().hex[:8]}",
            deal_id=deal_id,
            question=question,
            claims=parsed_claims,
            conflicts=[],
            citations=[],
            summary=summary,
            created_at=datetime.utcnow(),
        )


# Convenience function for one-off requests
async def fetch_evidence_context(
    deal_id: str,
    question: Optional[str] = None,
    constraints: Optional[ContextConstraints] = None,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
) -> ContextResponse:
    """
    Convenience function to fetch an evidence context.

    Args:
        deal_id: Identifier for the deal/company
        question: Optional question to focus the evidence retrieval
        constraints: Constraints for building the context
        base_url: Override base URL (defaults to env var)
        token: Override auth token (defaults to env var)

    Returns:
        ContextResponse with claims, conflicts, and summary
    """
    async with EvidenceApiClient(base_url=base_url, token=token) as client:
        return await client.create_context(deal_id, question, constraints)
