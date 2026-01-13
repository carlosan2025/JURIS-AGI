"""
Unit tests for Evidence API client.

These tests mock the Evidence API to verify client behavior.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from juris_agi.evidence_client import (
    EvidenceApiClient,
    EvidenceContext,
    ContextConstraints,
    ContextResponse,
    ClaimResponse,
    Claim,
    ClaimPolarity,
    ContextSummary,
    EvidenceAPIError,
    EvidenceConnectionError,
    EvidenceNotFoundError,
    EvidenceRateLimitError,
    EvidenceTimeoutError,
    EvidenceUnavailableError,
    RetryConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context_response():
    """Sample context response from Evidence API."""
    return {
        "context_id": "ctx_abc123",
        "claims": [
            {
                "claim_id": "claim_1",
                "claim_type": "traction",
                "field": "arr",
                "value": 1000000,
                "confidence": 0.85,
                "polarity": "supportive",
            },
            {
                "claim_id": "claim_2",
                "claim_type": "team_quality",
                "field": "founder_background",
                "value": "Ex-Google, 10 years ML",
                "confidence": 0.90,
                "polarity": "supportive",
            },
        ],
        "conflicts": [],
        "citations": [],
        "summary": {
            "total_claims": 2,
            "claims_by_type": {"traction": 1, "team_quality": 1},
            "claims_by_polarity": {"supportive": 2},
            "avg_confidence": 0.875,
            "conflict_count": 0,
            "document_count": 1,
        },
    }


@pytest.fixture
def mock_claim_response():
    """Sample claim response from Evidence API."""
    return {
        "claim": {
            "claim_id": "claim_1",
            "claim_type": "traction",
            "field": "arr",
            "value": 1000000,
            "confidence": 0.85,
            "polarity": "supportive",
        },
        "related_claims": ["claim_2", "claim_3"],
        "conflicts": [],
    }


@pytest.fixture
def client():
    """Create a client with a test base URL."""
    return EvidenceApiClient(base_url="https://evidence-api.example.com", token="test-token")


# =============================================================================
# Client Configuration Tests
# =============================================================================


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_client_not_configured_without_url(self):
        """Client should report not configured without base URL."""
        client = EvidenceApiClient(base_url=None)
        assert not client.is_configured

    def test_client_configured_with_url(self):
        """Client should report configured with base URL."""
        client = EvidenceApiClient(base_url="https://api.example.com")
        assert client.is_configured

    def test_client_reads_env_vars(self):
        """Client should read configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "EVIDENCE_API_BASE_URL": "https://env-api.example.com",
                "EVIDENCE_API_TOKEN": "env-token",
                "EVIDENCE_API_TIMEOUT": "60",
            },
        ):
            client = EvidenceApiClient()
            assert client.base_url == "https://env-api.example.com"
            assert client.token == "env-token"
            assert client.timeout == 60.0

    def test_explicit_params_override_env(self):
        """Explicit parameters should override environment variables."""
        with patch.dict(
            "os.environ",
            {"EVIDENCE_API_BASE_URL": "https://env-api.example.com"},
        ):
            client = EvidenceApiClient(base_url="https://explicit-api.example.com")
            assert client.base_url == "https://explicit-api.example.com"


# =============================================================================
# API Request Tests
# =============================================================================


class TestCreateContext:
    """Tests for create_context method."""

    @pytest.mark.asyncio
    async def test_create_context_success(self, client, mock_context_response):
        """Should successfully create a context."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = mock_context_response

        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.create_context(
                deal_id="test-deal",
                question="Should we invest?",
                constraints=ContextConstraints(max_claims=50),
            )

            assert isinstance(result, ContextResponse)
            assert result.context_id == "ctx_abc123"
            assert len(result.claims) == 2
            assert result.claims[0].claim_type == "traction"
            assert result.summary.total_claims == 2

    @pytest.mark.asyncio
    async def test_create_context_not_configured(self):
        """Should raise error when not configured."""
        client = EvidenceApiClient(base_url=None)

        with pytest.raises(EvidenceUnavailableError) as exc_info:
            await client.create_context(deal_id="test-deal")

        assert "not configured" in str(exc_info.value).lower()


class TestGetContext:
    """Tests for get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_success(self, client):
        """Should successfully get a context."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "context_id": "ctx_abc123",
            "deal_id": "test-deal",
            "claims": [],
            "conflicts": [],
            "citations": [],
            "summary": {
                "total_claims": 0,
                "claims_by_type": {},
                "claims_by_polarity": {},
                "avg_confidence": 0.0,
            },
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_context("ctx_abc123")

            assert isinstance(result, EvidenceContext)
            assert result.context_id == "ctx_abc123"
            assert result.deal_id == "test-deal"


class TestGetClaim:
    """Tests for get_claim method."""

    @pytest.mark.asyncio
    async def test_get_claim_success(self, client, mock_claim_response):
        """Should successfully get a claim."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = mock_claim_response

        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.get_claim("claim_1")

            assert isinstance(result, ClaimResponse)
            assert result.claim.claim_id == "claim_1"
            assert result.claim.value == 1000000
            assert "claim_2" in result.related_claims

    @pytest.mark.asyncio
    async def test_get_claim_not_found(self, client):
        """Should raise NotFoundError for missing claim."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.is_success = False
        mock_response.text = "Claim not found"
        mock_response.url = "https://api.example.com/claims/missing"
        mock_response.json.return_value = {"detail": "Claim not found"}

        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(EvidenceNotFoundError):
                await client.get_claim("missing")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_connection_error(self, client):
        """Should raise connection error on network failure."""
        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(EvidenceConnectionError):
                await client.create_context(deal_id="test")

    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Should raise timeout error on request timeout."""
        with patch.object(
            httpx.AsyncClient, "request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(EvidenceTimeoutError):
                await client.create_context(deal_id="test")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client):
        """Should raise rate limit error with retry-after."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.is_success = False
        mock_response.headers = {"Retry-After": "30"}
        mock_response.text = "Rate limited"

        # Disable retries for this test
        client.retry_config = RetryConfig(max_retries=0)

        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(EvidenceRateLimitError) as exc_info:
                await client.create_context(deal_id="test")

            assert exc_info.value.retry_after == 30.0


# =============================================================================
# Demo Mode Tests
# =============================================================================


class TestDemoMode:
    """Tests for demo mode (direct claims)."""

    def test_from_direct_claims(self):
        """Should create context from direct claims."""
        claims = [
            {
                "claim_type": "traction",
                "field": "arr",
                "value": 1000000,
                "confidence": 0.85,
                "polarity": "supportive",
            },
            {
                "claim_type": "team_quality",
                "field": "founder_background",
                "value": "Ex-Google",
                "confidence": 0.90,
                "polarity": "supportive",
            },
            {
                "claim_type": "execution_risk",
                "field": "key_person",
                "value": "Single technical founder",
                "confidence": 0.75,
                "polarity": "risk",
            },
        ]

        context = EvidenceApiClient.from_direct_claims(
            deal_id="demo-company",
            claims=claims,
            question="Should we invest?",
        )

        assert isinstance(context, EvidenceContext)
        assert context.deal_id == "demo-company"
        assert context.question == "Should we invest?"
        assert len(context.claims) == 3
        assert context.summary.total_claims == 3
        assert context.summary.claims_by_type["traction"] == 1
        assert context.summary.claims_by_polarity["supportive"] == 2
        assert context.summary.claims_by_polarity["risk"] == 1

    def test_from_direct_claims_generates_ids(self):
        """Should generate claim IDs if not provided."""
        claims = [
            {
                "claim_type": "traction",
                "field": "arr",
                "value": 1000000,
                "confidence": 0.85,
                "polarity": "supportive",
            },
        ]

        context = EvidenceApiClient.from_direct_claims(
            deal_id="demo-company",
            claims=claims,
        )

        assert context.claims[0].claim_id.startswith("claim_")

    def test_from_direct_claims_preserves_ids(self):
        """Should preserve claim IDs if provided."""
        claims = [
            {
                "claim_id": "my_custom_id",
                "claim_type": "traction",
                "field": "arr",
                "value": 1000000,
                "confidence": 0.85,
                "polarity": "supportive",
            },
        ]

        context = EvidenceApiClient.from_direct_claims(
            deal_id="demo-company",
            claims=claims,
        )

        assert context.claims[0].claim_id == "my_custom_id"


# =============================================================================
# Retry Configuration Tests
# =============================================================================


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_calculate_delay_exponential(self):
        """Should calculate exponential backoff."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_respects_max(self):
        """Should not exceed max delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)

        assert config.calculate_delay(10) == 5.0

    def test_calculate_delay_uses_retry_after(self):
        """Should use retry-after header when provided."""
        config = RetryConfig(jitter=False)

        assert config.calculate_delay(0, retry_after=30.0) == 30.0

    def test_should_retry_respects_max(self):
        """Should not retry beyond max attempts."""
        config = RetryConfig(max_retries=3)

        assert config.should_retry(EvidenceConnectionError("test"), 2) is True
        assert config.should_retry(EvidenceConnectionError("test"), 3) is False

    def test_should_retry_for_retryable_exceptions(self):
        """Should retry for retryable exceptions."""
        config = RetryConfig()

        assert config.should_retry(EvidenceConnectionError("test"), 0) is True
        assert config.should_retry(EvidenceTimeoutError("test"), 0) is True
        assert config.should_retry(EvidenceRateLimitError("test"), 0) is True

    def test_should_not_retry_for_non_retryable(self):
        """Should not retry for non-retryable errors."""
        config = RetryConfig()

        assert config.should_retry(EvidenceNotFoundError("test", 404), 0) is False
        assert config.should_retry(ValueError("test"), 0) is False


# =============================================================================
# Type Tests
# =============================================================================


class TestTypes:
    """Tests for pydantic types."""

    def test_claim_confidence_level(self):
        """Should correctly categorize confidence levels."""
        high_conf = Claim(
            claim_id="1",
            claim_type="test",
            field="test",
            value="test",
            confidence=0.85,
            polarity=ClaimPolarity.SUPPORTIVE,
        )
        assert high_conf.confidence_level.value == "high"

        med_conf = Claim(
            claim_id="2",
            claim_type="test",
            field="test",
            value="test",
            confidence=0.65,
            polarity=ClaimPolarity.NEUTRAL,
        )
        assert med_conf.confidence_level.value == "medium"

        low_conf = Claim(
            claim_id="3",
            claim_type="test",
            field="test",
            value="test",
            confidence=0.3,
            polarity=ClaimPolarity.RISK,
        )
        assert low_conf.confidence_level.value == "low"

    def test_context_constraints_defaults(self):
        """Should have sensible defaults for constraints."""
        constraints = ContextConstraints()

        assert constraints.max_claims == 100
        assert constraints.per_bucket_cap == 20
        assert constraints.min_confidence == 0.0
        assert constraints.include_conflicts is True
        assert constraints.include_citations is True

    def test_context_constraints_validation(self):
        """Should validate constraint values."""
        with pytest.raises(ValueError):
            ContextConstraints(max_claims=0)  # Must be >= 1

        with pytest.raises(ValueError):
            ContextConstraints(min_confidence=1.5)  # Must be <= 1.0
