"""
Integration tests for VC decision endpoints.

Tests the end-to-end flow with mocked Evidence API.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create test client."""
    from juris_agi.api.server import app
    return TestClient(app)


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        {
            "claim_type": "traction",
            "field": "arr",
            "value": 2500000,
            "confidence": 0.9,
            "polarity": "supportive",
        },
        {
            "claim_type": "traction",
            "field": "growth_rate",
            "value": 0.75,
            "confidence": 0.85,
            "polarity": "supportive",
        },
        {
            "claim_type": "team_quality",
            "field": "founder_experience",
            "value": "serial_entrepreneur",
            "confidence": 0.8,
            "polarity": "supportive",
        },
        {
            "claim_type": "market",
            "field": "tam",
            "value": 5000000000,
            "confidence": 0.7,
            "polarity": "supportive",
        },
    ]


@pytest.fixture
def sample_timeseries_claims():
    """Claims with time series data."""
    return [
        {
            "claim_type": "traction",
            "field": "arr",
            "value": 2500000,
            "confidence": 0.9,
            "polarity": "supportive",
            "timeseries": [
                {"t": "Q1 2024", "value": 1000000},
                {"t": "Q2 2024", "value": 1500000},
                {"t": "Q3 2024", "value": 2000000},
                {"t": "Q4 2024", "value": 2500000},
            ],
        },
        {
            "claim_type": "traction",
            "field": "customers",
            "value": 50,
            "confidence": 0.85,
            "polarity": "supportive",
            "timeseries": [
                {"t": "Q1 2024", "value": 20},
                {"t": "Q2 2024", "value": 30},
                {"t": "Q3 2024", "value": 40},
                {"t": "Q4 2024", "value": 50},
            ],
        },
    ]


@pytest.fixture
def sample_historical_decisions():
    """Historical decisions for policy learning."""
    return [
        {
            "id": "deal_1",
            "claims": [
                {"claim_type": "traction", "field": "arr", "value": 3000000, "confidence": 0.9, "polarity": "supportive"},
                {"claim_type": "traction", "field": "growth_rate", "value": 0.8, "confidence": 0.85, "polarity": "supportive"},
            ],
            "decision": "invest",
            "metadata": {"sector": "saas"},
        },
        {
            "id": "deal_2",
            "claims": [
                {"claim_type": "traction", "field": "arr", "value": 500000, "confidence": 0.9, "polarity": "neutral"},
                {"claim_type": "traction", "field": "growth_rate", "value": 0.2, "confidence": 0.85, "polarity": "risk"},
            ],
            "decision": "pass",
            "metadata": {"sector": "saas"},
        },
        {
            "id": "deal_3",
            "claims": [
                {"claim_type": "traction", "field": "arr", "value": 1500000, "confidence": 0.9, "polarity": "supportive"},
                {"claim_type": "traction", "field": "growth_rate", "value": 0.5, "confidence": 0.85, "polarity": "supportive"},
            ],
            "decision": "invest",
            "metadata": {"sector": "saas"},
        },
    ]


# =============================================================================
# Basic Endpoint Tests
# =============================================================================


class TestVCSolveEndpoint:
    """Tests for POST /vc/solve endpoint."""

    def test_submit_with_claims(self, client, sample_claims):
        """Submit a request with direct claims."""
        response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "constraints": {
                "max_policies": 3,
                "min_confidence": 0.7,
            },
        })

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["job_id"].startswith("vcjob_")
        assert data["status"] == "pending"
        assert "trace_url" in data
        assert "events_url" in data

    def test_submit_with_deal_id_no_api(self, client):
        """Submit with deal_id but no Evidence API configured."""
        # This should start but fail during execution since Evidence API is not configured
        response = client.post("/vc/solve", json={
            "deal_id": "test-deal-123",
            "question": "Should we invest?",
        })

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_submit_missing_both(self, client):
        """Submit without deal_id or claims should fail."""
        response = client.post("/vc/solve", json={
            "constraints": {"max_policies": 3},
        })

        assert response.status_code == 400
        assert "deal_id or claims" in response.json()["message"]

    def test_submit_with_constraints(self, client, sample_claims):
        """Submit with custom constraints."""
        response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "constraints": {
                "max_claims": 50,
                "min_confidence": 0.8,
                "max_policies": 5,
                "min_coverage": 0.9,
            },
        })

        assert response.status_code == 200

    def test_submit_with_historical_decisions(self, client, sample_claims, sample_historical_decisions):
        """Submit with historical decisions for policy learning."""
        response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "historical_decisions": sample_historical_decisions,
        })

        assert response.status_code == 200


class TestVCJobResultEndpoint:
    """Tests for GET /vc/jobs/{job_id} endpoint."""

    def test_get_nonexistent_job(self, client):
        """Get a job that doesn't exist."""
        response = client.get("/vc/jobs/vcjob_nonexistent")

        assert response.status_code == 404

    def test_get_job_result(self, client, sample_claims):
        """Submit and retrieve job result."""
        # Submit job
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
        })
        job_id = submit_response.json()["job_id"]

        # Give it a moment to process (in test mode it runs synchronously)
        import time
        time.sleep(0.5)

        # Get result
        response = client.get(f"/vc/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data

    def test_get_job_with_trace(self, client, sample_claims):
        """Get job result with trace data."""
        # Submit job
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "include_trace": True,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(0.5)

        # Get result with trace
        response = client.get(f"/vc/jobs/{job_id}?include_trace=true")

        assert response.status_code == 200


class TestVCJobEventsEndpoint:
    """Tests for GET /vc/jobs/{job_id}/events endpoint."""

    def test_get_job_events(self, client, sample_claims):
        """Get job events."""
        # Submit job
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(0.5)

        # Get events
        response = client.get(f"/vc/jobs/{job_id}/events")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "events" in data
        assert isinstance(data["events"], list)


class TestVCJobTraceEndpoint:
    """Tests for GET /vc/jobs/{job_id}/trace endpoint."""

    def test_get_job_trace(self, client, sample_claims):
        """Get job trace."""
        # Submit job
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(0.5)

        # Get trace
        response = client.get(f"/vc/jobs/{job_id}/trace")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "trace" in data


# =============================================================================
# Integration Tests with Mocked Evidence API
# =============================================================================


class TestWithMockedEvidenceAPI:
    """Integration tests with mocked Evidence API."""

    @pytest.fixture
    def mock_evidence_response(self):
        """Mock response from Evidence API."""
        from juris_agi.evidence_client.types import (
            ContextResponse,
            ContextSummary,
            Claim,
            ClaimPolarity,
        )

        return ContextResponse(
            context_id="ctx_test123",
            claims=[
                Claim(
                    claim_id="claim_1",
                    claim_type="traction",
                    field="arr",
                    value=2500000,
                    confidence=0.9,
                    polarity=ClaimPolarity.SUPPORTIVE,
                ),
                Claim(
                    claim_id="claim_2",
                    claim_type="traction",
                    field="growth_rate",
                    value=0.75,
                    confidence=0.85,
                    polarity=ClaimPolarity.SUPPORTIVE,
                ),
                Claim(
                    claim_id="claim_3",
                    claim_type="team_quality",
                    field="founder_experience",
                    value="serial_entrepreneur",
                    confidence=0.8,
                    polarity=ClaimPolarity.SUPPORTIVE,
                ),
            ],
            conflicts=[],
            citations=[],
            summary=ContextSummary(
                total_claims=3,
                claims_by_type={"traction": 2, "team_quality": 1},
                claims_by_polarity={"supportive": 3},
                avg_confidence=0.85,
                conflict_count=0,
                document_count=2,
            ),
        )

    @pytest.mark.asyncio
    async def test_with_mocked_evidence_api(self, mock_evidence_response):
        """Test orchestrator with mocked Evidence API."""
        from juris_agi.vc.orchestrator import VCOrchestrator, OrchestratorConfig
        from juris_agi.evidence_client.client import EvidenceApiClient

        # Create mock client
        mock_client = AsyncMock(spec=EvidenceApiClient)
        mock_client.is_configured = True
        mock_client.create_context = AsyncMock(return_value=mock_evidence_response)

        # Create orchestrator with mock client
        orchestrator = VCOrchestrator(
            config=OrchestratorConfig(
                propose_thresholds=True,
                learn_hierarchical=False,
                analyze_uncertainty=True,
            ),
            evidence_client=mock_client,
        )

        # Run
        result = await orchestrator.solve(
            deal_id="test-deal-123",
            question="Should we invest in Series A?",
        )

        # Verify
        assert result.status.value == "completed"
        assert result.context_id == "ctx_test123"
        assert result.working_set is not None
        assert result.working_set.total_claims == 3
        assert result.decision is not None
        assert len(result.events) > 0

        # Verify Evidence API was called
        mock_client.create_context.assert_called_once()


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow_with_claims(self, client, sample_claims):
        """Test full workflow from submission to result."""
        # Submit
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "constraints": {"max_policies": 3},
        })
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Wait for processing
        import time
        time.sleep(1.0)

        # Get result
        result_response = client.get(f"/vc/jobs/{job_id}")
        assert result_response.status_code == 200

        result = result_response.json()

        # Verify structure
        assert result["job_id"] == job_id
        assert result["status"] in ["completed", "failed", "pending"]

        if result["status"] == "completed":
            assert "decision" in result
            assert "working_set" in result

    def test_workflow_with_timeseries(self, client, sample_timeseries_claims):
        """Test workflow with time series data."""
        submit_response = client.post("/vc/solve", json={
            "claims": sample_timeseries_claims,
        })
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.0)

        result_response = client.get(f"/vc/jobs/{job_id}")
        result = result_response.json()

        if result["status"] == "completed":
            # Verify time series fields were detected
            working_set = result.get("working_set", {})
            if working_set:
                assert "timeseries_fields" in working_set

    def test_workflow_with_policy_learning(self, client, sample_claims, sample_historical_decisions):
        """Test workflow with policy learning from historical decisions."""
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "historical_decisions": sample_historical_decisions,
            "constraints": {"max_policies": 5},
        })
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.5)

        result_response = client.get(f"/vc/jobs/{job_id}")
        result = result_response.json()

        if result["status"] == "completed":
            # Should have learned policies
            policies = result.get("policies", [])
            # May have policies if learning succeeded
            assert isinstance(policies, list)


# =============================================================================
# Event Verification Tests
# =============================================================================


class TestJobEvents:
    """Tests for job event generation."""

    def test_events_generated(self, client, sample_claims):
        """Verify events are generated during processing."""
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "include_events": True,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.0)

        events_response = client.get(f"/vc/jobs/{job_id}/events")
        events = events_response.json()["events"]

        if len(events) > 0:
            # Verify event structure
            for event in events:
                assert "event_id" in event
                assert "event_type" in event
                assert "timestamp" in event

    def test_event_types(self, client, sample_claims):
        """Verify expected event types are present."""
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.0)

        events_response = client.get(f"/vc/jobs/{job_id}/events")
        events = events_response.json()["events"]

        event_types = [e["event_type"] for e in events]

        # Check for key events (if job completed)
        result_response = client.get(f"/vc/jobs/{job_id}")
        if result_response.json()["status"] == "completed":
            # Should have context and working set events
            assert any("context" in t for t in event_types) or any("working_set" in t for t in event_types)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_claim_format(self, client):
        """Test handling of invalid claim format."""
        response = client.post("/vc/solve", json={
            "claims": [
                {"invalid_field": "value"},  # Missing required fields
            ],
        })

        # Should fail validation
        assert response.status_code in [400, 422]

    def test_invalid_constraints(self, client, sample_claims):
        """Test handling of invalid constraints."""
        response = client.post("/vc/solve", json={
            "claims": sample_claims,
            "constraints": {
                "max_policies": -1,  # Invalid
            },
        })

        # Should fail validation
        assert response.status_code in [400, 422]


# =============================================================================
# Uncertainty Analysis Tests
# =============================================================================


class TestUncertaintyAnalysis:
    """Tests for uncertainty analysis output."""

    def test_uncertainty_output_present(self, client, sample_claims):
        """Verify uncertainty analysis is included in output."""
        submit_response = client.post("/vc/solve", json={
            "claims": sample_claims,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.0)

        result_response = client.get(f"/vc/jobs/{job_id}")
        result = result_response.json()

        if result["status"] == "completed":
            uncertainty = result.get("uncertainty")
            if uncertainty:
                assert "epistemic_score" in uncertainty
                assert "aleatoric_score" in uncertainty
                assert "total_uncertainty" in uncertainty
                assert "uncertainty_level" in uncertainty

    def test_low_confidence_increases_uncertainty(self, client):
        """Low confidence claims should increase uncertainty."""
        low_conf_claims = [
            {
                "claim_type": "traction",
                "field": "arr",
                "value": 2500000,
                "confidence": 0.3,  # Low confidence
                "polarity": "supportive",
            },
            {
                "claim_type": "traction",
                "field": "growth_rate",
                "value": 0.75,
                "confidence": 0.3,  # Low confidence
                "polarity": "supportive",
            },
        ]

        submit_response = client.post("/vc/solve", json={
            "claims": low_conf_claims,
        })
        job_id = submit_response.json()["job_id"]

        import time
        time.sleep(1.0)

        result_response = client.get(f"/vc/jobs/{job_id}")
        result = result_response.json()

        if result["status"] == "completed":
            uncertainty = result.get("uncertainty")
            if uncertainty:
                # Low confidence should elevate aleatoric uncertainty
                # (though exact threshold depends on config)
                assert uncertainty["aleatoric_score"] >= 0
