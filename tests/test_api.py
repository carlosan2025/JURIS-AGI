"""
Tests for the JURIS-AGI API endpoints.

These tests use pytest-asyncio and httpx for async testing.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# Try to import FastAPI test client
try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import API modules if available
if FASTAPI_AVAILABLE:
    from juris_agi.api.server import app, create_app
    from juris_agi.api.models import (
        SolveRequest,
        SolveResponse,
        JobStatus,
        JobResult,
        HealthResponse,
        TaskPayload,
        TrainPair,
        TestPair,
        GridData,
        BudgetConfig,
    )
    from juris_agi.api.config import APIConfig


# Skip all tests if FastAPI not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_task():
    """Create a sample task payload."""
    return {
        "train": [
            {
                "input": {"data": [[1, 2], [3, 4]]},
                "output": {"data": [[3, 1], [4, 2]]}
            }
        ],
        "test": [
            {"input": {"data": [[5, 6], [7, 8]]}}
        ]
    }


@pytest.fixture
def sample_solve_request(sample_task):
    """Create a sample solve request."""
    return {
        "task_id": "test_task",
        "task": sample_task,
        "budget": {
            "max_time_seconds": 10.0,
            "max_iterations": 100,
            "beam_width": 10,
            "max_depth": 3
        },
        "use_neural": False,
        "return_trace": True
    }


# =============================================================================
# Test Health Endpoint
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "gpu_available" in data
        assert "torch_available" in data
        assert "redis_connected" in data

    def test_health_version_matches(self, client):
        """Version should match API version."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == "0.1.0"

    def test_health_status_is_valid(self, client):
        """Status should be healthy or degraded."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]


# =============================================================================
# Test Solve Endpoint
# =============================================================================

class TestSolveEndpoint:
    """Tests for POST /solve endpoint."""

    def test_solve_returns_202_or_200(self, client, sample_solve_request):
        """Solve endpoint should accept valid request."""
        response = client.post("/solve", json=sample_solve_request)
        # May be 200 or 202 depending on async handling
        assert response.status_code in [200, 202]

    def test_solve_returns_job_id(self, client, sample_solve_request):
        """Solve should return a job ID."""
        response = client.post("/solve", json=sample_solve_request)
        data = response.json()

        assert "job_id" in data
        assert data["job_id"].startswith("job_")

    def test_solve_returns_pending_status(self, client, sample_solve_request):
        """Initial status should be pending."""
        response = client.post("/solve", json=sample_solve_request)
        data = response.json()

        assert data["status"] == "pending"

    def test_solve_returns_created_at(self, client, sample_solve_request):
        """Should include created_at timestamp."""
        response = client.post("/solve", json=sample_solve_request)
        data = response.json()

        assert "created_at" in data
        # Should be parseable as ISO datetime
        datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

    def test_solve_with_priority(self, client, sample_solve_request):
        """Should accept priority parameter."""
        response = client.post("/solve?priority=high", json=sample_solve_request)
        assert response.status_code in [200, 202]

    def test_solve_invalid_priority_rejected(self, client, sample_solve_request):
        """Invalid priority should be rejected."""
        response = client.post("/solve?priority=invalid", json=sample_solve_request)
        assert response.status_code == 422

    def test_solve_missing_task_rejected(self, client):
        """Missing task should be rejected."""
        response = client.post("/solve", json={})
        assert response.status_code == 422

    def test_solve_empty_train_accepted(self, client):
        """Empty training set should be accepted."""
        request = {
            "task": {
                "train": [],
                "test": []
            }
        }
        response = client.post("/solve", json=request)
        # This is technically valid, solver will handle it
        assert response.status_code in [200, 202]

    def test_solve_budget_limits(self, client, sample_task):
        """Budget limits should be enforced."""
        # Max time too high
        request = {
            "task": sample_task,
            "budget": {"max_time_seconds": 1000}  # > 600 max
        }
        response = client.post("/solve", json=request)
        assert response.status_code == 422


# =============================================================================
# Test Job Status Endpoint
# =============================================================================

class TestJobStatusEndpoint:
    """Tests for GET /jobs/{job_id} endpoint."""

    def test_get_nonexistent_job_404(self, client):
        """Nonexistent job should return 404."""
        response = client.get("/jobs/job_nonexistent")
        assert response.status_code == 404

    def test_get_job_after_submit(self, client, sample_solve_request):
        """Should be able to get job after submit."""
        # Submit job
        submit_response = client.post("/solve", json=sample_solve_request)
        job_id = submit_response.json()["job_id"]

        # Get job status
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id

    def test_get_job_includes_task_id(self, client, sample_solve_request):
        """Job status should include task_id."""
        submit_response = client.post("/solve", json=sample_solve_request)
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")
        data = response.json()

        assert data["task_id"] == "test_task"

    def test_get_job_include_trace(self, client, sample_solve_request):
        """Should support include_trace parameter."""
        submit_response = client.post("/solve", json=sample_solve_request)
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}?include_trace=true")
        assert response.status_code == 200


# =============================================================================
# Test Request/Response Models
# =============================================================================

class TestModels:
    """Tests for Pydantic models."""

    def test_grid_data_validation(self):
        """GridData should validate data."""
        grid = GridData(data=[[0, 1], [2, 3]])
        assert grid.data == [[0, 1], [2, 3]]

    def test_task_payload_structure(self):
        """TaskPayload should have train and test."""
        task = TaskPayload(
            train=[TrainPair(
                input=GridData(data=[[1]]),
                output=GridData(data=[[2]])
            )],
            test=[]
        )
        assert len(task.train) == 1

    def test_budget_config_defaults(self):
        """BudgetConfig should have sensible defaults."""
        budget = BudgetConfig()
        assert budget.max_time_seconds == 60.0
        assert budget.max_iterations == 1000
        assert budget.beam_width == 50
        assert budget.max_depth == 4

    def test_solve_request_defaults(self):
        """SolveRequest should have sensible defaults."""
        request = SolveRequest(
            task=TaskPayload(train=[], test=[])
        )
        assert request.use_neural is True
        assert request.return_trace is True

    def test_job_status_enum(self):
        """JobStatus should have expected values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    def test_health_response_model(self):
        """HealthResponse should validate correctly."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            gpu_available=False,
            torch_available=True,
            redis_connected=False,
            worker_count=0,
            pending_jobs=0
        )
        assert health.status == "healthy"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_rejected(self, client):
        """Invalid JSON should be rejected."""
        response = client.post(
            "/solve",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Wrong content type should be handled."""
        response = client.post(
            "/solve",
            content="task=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Wrong HTTP method should return 405."""
        response = client.put("/solve", json={})
        assert response.status_code == 405


# =============================================================================
# Test Config
# =============================================================================

class TestConfig:
    """Tests for configuration."""

    def test_api_config_defaults(self):
        """APIConfig should have sensible defaults."""
        from juris_agi.api.config import APIConfig
        config = APIConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4

    def test_api_config_from_env(self):
        """APIConfig should read from environment."""
        import os
        from juris_agi.api.config import APIConfig

        # Set env vars
        os.environ["JURIS_PORT"] = "9000"

        config = APIConfig.from_env()
        assert config.port == 9000

        # Cleanup
        del os.environ["JURIS_PORT"]

    def test_worker_config_defaults(self):
        """WorkerConfig should have sensible defaults."""
        from juris_agi.api.config import WorkerConfig
        config = WorkerConfig()

        assert config.max_concurrent_jobs == 1
        assert "juris_default" in config.queue_names


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full API flow."""

    def test_full_solve_flow(self, client, sample_solve_request):
        """Test complete solve flow: submit -> check status."""
        # Submit
        submit_response = client.post("/solve", json=sample_solve_request)
        assert submit_response.status_code in [200, 202]

        job_id = submit_response.json()["job_id"]

        # Check status
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        # Status could be pending, running, or completed depending on timing
        assert status_data["status"] in ["pending", "running", "completed", "failed"]

    def test_multiple_jobs(self, client, sample_solve_request):
        """Should handle multiple concurrent jobs."""
        job_ids = []

        # Submit multiple jobs
        for i in range(3):
            request = sample_solve_request.copy()
            request["task_id"] = f"test_task_{i}"

            response = client.post("/solve", json=request)
            assert response.status_code in [200, 202]
            job_ids.append(response.json()["job_id"])

        # All should be unique
        assert len(set(job_ids)) == 3

        # All should be retrievable
        for job_id in job_ids:
            response = client.get(f"/jobs/{job_id}")
            assert response.status_code == 200
