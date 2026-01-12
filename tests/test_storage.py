"""Tests for storage client and model registry."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from juris_agi.core.storage import (
    StorageConfig,
    StorageClient,
    LocalStorageBackend,
    S3StorageBackend,
    ArtifactMetadata,
)
from juris_agi.core.model_registry import (
    ModelRegistry,
    ModelEntry,
    ModelVersion,
)


# =============================================================================
# Storage Config Tests
# =============================================================================


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StorageConfig()
        assert config.backend == "local"
        assert config.local_path == "/tmp/juris_storage"
        assert config.s3_bucket is None

    def test_s3_config(self):
        """Test S3 configuration."""
        config = StorageConfig(
            backend="s3",
            s3_bucket="test-bucket",
            s3_endpoint="http://localhost:9000",
            s3_region="us-east-1",
            s3_access_key="access",
            s3_secret_key="secret",
        )
        assert config.backend == "s3"
        assert config.s3_bucket == "test-bucket"
        assert config.s3_endpoint == "http://localhost:9000"


# =============================================================================
# Local Storage Backend Tests
# =============================================================================


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(backend="local", local_path=tmpdir)
            yield LocalStorageBackend(config)

    def test_put_and_get(self, temp_storage):
        """Test storing and retrieving data."""
        data = b'{"test": "data"}'
        key = "test/file.json"

        metadata = temp_storage.put(key, data, content_type="application/json")

        assert metadata.key == key
        assert metadata.size == len(data)
        assert metadata.content_type == "application/json"

        retrieved = temp_storage.get(key)
        assert retrieved == data

    def test_exists(self, temp_storage):
        """Test checking file existence."""
        key = "test/exists.json"
        assert not temp_storage.exists(key)

        temp_storage.put(key, b'{}')
        assert temp_storage.exists(key)

    def test_delete(self, temp_storage):
        """Test deleting files."""
        key = "test/delete.json"
        temp_storage.put(key, b'{}')
        assert temp_storage.exists(key)

        result = temp_storage.delete(key)
        assert result is True
        assert not temp_storage.exists(key)

    def test_list_artifacts(self, temp_storage):
        """Test listing artifacts with prefix."""
        temp_storage.put("traces/2024/01/file1.json", b'{}')
        temp_storage.put("traces/2024/01/file2.json", b'{}')
        temp_storage.put("results/2024/01/file1.json", b'{}')

        trace_artifacts = temp_storage.list("traces/")
        assert len(trace_artifacts) == 2
        assert all("traces/" in a.key for a in trace_artifacts)

    def test_get_url_local(self, temp_storage):
        """Test URL generation for local storage."""
        key = "test/url.json"
        temp_storage.put(key, b'{}')

        url = temp_storage.get_url(key)
        assert url.startswith("file://")
        assert key in url

    def test_metadata_storage(self, temp_storage):
        """Test metadata is stored and retrieved."""
        key = "test/meta.json"
        custom_meta = {"job_id": "123", "task_id": "456"}

        temp_storage.put(key, b'{}', metadata=custom_meta)

        # Read metadata file directly
        meta_path = Path(temp_storage.base_path) / f"{key}.meta"
        assert meta_path.exists()

        with open(meta_path) as f:
            stored_meta = json.load(f)
        assert stored_meta["metadata"]["job_id"] == "123"


# =============================================================================
# S3 Storage Backend Tests (Mocked)
# =============================================================================


try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
class TestS3StorageBackend:
    """Tests for S3StorageBackend (mocked)."""

    @pytest.fixture
    def mock_s3(self):
        """Create a mocked S3 backend."""
        with patch("juris_agi.core.storage.boto3") as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            config = StorageConfig(
                backend="s3",
                s3_bucket="test-bucket",
                s3_endpoint="http://localhost:9000",
                s3_region="us-east-1",
                s3_access_key="test",
                s3_secret_key="test",
            )

            backend = S3StorageBackend(config)
            backend.client = mock_client
            yield backend, mock_client

    def test_put_calls_s3(self, mock_s3):
        """Test that put calls S3 put_object."""
        backend, mock_client = mock_s3
        mock_client.put_object.return_value = {}

        metadata = backend.put("test/key.json", b'{}')

        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["Bucket"] == "test-bucket"
        assert call_args.kwargs["Key"] == "test/key.json"

    def test_get_calls_s3(self, mock_s3):
        """Test that get calls S3 get_object."""
        backend, mock_client = mock_s3
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"test": "data"}'
        mock_client.get_object.return_value = {"Body": mock_body}

        data = backend.get("test/key.json")

        assert data == b'{"test": "data"}'
        mock_client.get_object.assert_called_once()

    def test_get_url_generates_presigned(self, mock_s3):
        """Test that get_url generates presigned URL."""
        backend, mock_client = mock_s3
        mock_client.generate_presigned_url.return_value = "https://example.com/signed"

        url = backend.get_url("test/key.json", expiry=3600)

        assert url == "https://example.com/signed"
        mock_client.generate_presigned_url.assert_called_once()


# =============================================================================
# Storage Client Tests
# =============================================================================


class TestStorageClient:
    """Tests for StorageClient high-level API."""

    @pytest.fixture
    def storage_client(self):
        """Create a storage client with temporary local storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(backend="local", local_path=tmpdir)
            yield StorageClient(config)

    def test_save_and_get_trace(self, storage_client):
        """Test saving and retrieving traces."""
        trace_data = {
            "entries": [{"step": 1, "action": "test"}],
            "final_result": {"success": True},
        }

        key = storage_client.save_trace(
            job_id="job123",
            task_id="task456",
            trace_data=trace_data,
        )

        assert "traces/" in key
        assert "job123" in key

        # Retrieve the trace
        retrieved = storage_client.get_trace("job123", "task456")
        assert retrieved["entries"] == trace_data["entries"]

    def test_save_and_get_result(self, storage_client):
        """Test saving and retrieving results."""
        result_data = {
            "success": True,
            "predictions": [{"data": [[1, 2], [3, 4]]}],
            "program": "rotate90(1)",
        }

        key = storage_client.save_result(
            job_id="job123",
            task_id="task456",
            result_data=result_data,
        )

        assert "results/" in key
        assert "job123" in key

        # Retrieve the result
        retrieved = storage_client.get_result("job123", "task456")
        assert retrieved["success"] is True

    def test_trace_url(self, storage_client):
        """Test getting trace URL."""
        key = storage_client.save_trace(
            job_id="job123",
            task_id="task456",
            trace_data={"test": True},
        )

        url = storage_client.get_trace_url(key, expiry=3600)

        assert url is not None
        assert "job123" in url

    def test_save_model(self, storage_client):
        """Test saving model artifacts."""
        model_data = b"fake model weights"

        key = storage_client.save_model(
            model_type="sketcher",
            version="v1.0.0",
            model_data=model_data,
        )

        assert "models/" in key
        assert "sketcher" in key

        # Retrieve the model
        retrieved = storage_client.get_model("sketcher", "v1.0.0")
        assert retrieved == model_data


# =============================================================================
# Model Version Tests
# =============================================================================


class TestModelVersion:
    """Tests for ModelVersion."""

    def test_from_dict(self):
        """Test creating ModelVersion from dictionary."""
        data = {
            "version": "v1.0.0",
            "path": "models/sketcher/v1.0.0/model.pt",
            "created_at": "2024-01-15T00:00:00",
            "description": "Test model",
            "metrics": {"accuracy": 0.95},
            "config": {"hidden_dim": 256},
            "is_default": True,
            "tags": ["production"],
        }

        version = ModelVersion.from_dict(data)

        assert version.version == "v1.0.0"
        assert version.path == "models/sketcher/v1.0.0/model.pt"
        assert version.metrics["accuracy"] == 0.95
        assert version.is_default is True
        assert "production" in version.tags

    def test_to_dict(self):
        """Test converting ModelVersion to dictionary."""
        version = ModelVersion(
            version="v1.0.0",
            path="models/test.pt",
            created_at="2024-01-15T00:00:00",
            metrics={"accuracy": 0.9},
        )

        data = version.to_dict()

        assert data["version"] == "v1.0.0"
        assert data["path"] == "models/test.pt"
        assert data["metrics"]["accuracy"] == 0.9


# =============================================================================
# Model Entry Tests
# =============================================================================


class TestModelEntry:
    """Tests for ModelEntry."""

    def test_add_version(self):
        """Test adding versions to a model entry."""
        entry = ModelEntry(name="test-model", model_type="sketcher")

        version = ModelVersion(
            version="v1.0.0",
            path="models/test.pt",
            created_at="2024-01-15T00:00:00",
            is_default=True,
        )
        entry.add_version(version)

        assert "v1.0.0" in entry.versions
        assert entry.default_version == "v1.0.0"

    def test_get_default_version(self):
        """Test getting the default version."""
        entry = ModelEntry(name="test-model", model_type="sketcher")

        v1 = ModelVersion(
            version="v1.0.0",
            path="models/v1.pt",
            created_at="2024-01-01T00:00:00",
        )
        v2 = ModelVersion(
            version="v2.0.0",
            path="models/v2.pt",
            created_at="2024-01-15T00:00:00",
            is_default=True,
        )
        entry.add_version(v1)
        entry.add_version(v2)

        default = entry.get_default_version()
        assert default.version == "v2.0.0"

    def test_list_versions(self):
        """Test listing all versions."""
        entry = ModelEntry(name="test-model", model_type="sketcher")
        entry.add_version(ModelVersion(version="v1", path="p1", created_at="2024-01-01"))
        entry.add_version(ModelVersion(version="v2", path="p2", created_at="2024-01-02"))

        versions = entry.list_versions()
        assert len(versions) == 2
        assert "v1" in versions
        assert "v2" in versions


# =============================================================================
# Model Registry Tests
# =============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            yield ModelRegistry(registry_path=registry_path)

    def test_register_model(self, temp_registry):
        """Test registering a new model."""
        version = temp_registry.register_model(
            model_id="test-sketcher",
            model_type="sketcher",
            version="v1.0.0",
            path="models/sketcher/v1.0.0/model.pt",
            description="Test sketcher model",
            metrics={"accuracy": 0.92},
            is_default=True,
        )

        assert version.version == "v1.0.0"
        assert temp_registry.get_model("test-sketcher") is not None

    def test_get_model_version(self, temp_registry):
        """Test getting a specific model version."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="critic",
            version="v1.0.0",
            path="models/v1.pt",
        )
        temp_registry.register_model(
            model_id="test-model",
            model_type="critic",
            version="v2.0.0",
            path="models/v2.pt",
            is_default=True,
        )

        v1 = temp_registry.get_model_version("test-model", "v1.0.0")
        assert v1.version == "v1.0.0"

        default = temp_registry.get_model_version("test-model")
        assert default.version == "v2.0.0"

    def test_list_models(self, temp_registry):
        """Test listing models by type."""
        temp_registry.register_model(
            model_id="sketcher-1",
            model_type="sketcher",
            version="v1",
            path="p1",
        )
        temp_registry.register_model(
            model_id="critic-1",
            model_type="critic",
            version="v1",
            path="p1",
        )

        all_models = temp_registry.list_models()
        assert len(all_models) == 2

        sketchers = temp_registry.list_models(model_type="sketcher")
        assert len(sketchers) == 1
        assert "sketcher-1" in sketchers

    def test_set_default_version(self, temp_registry):
        """Test setting the default version."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v1.0.0",
            path="p1",
            is_default=True,
        )
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v2.0.0",
            path="p2",
        )

        # v1 should be default
        default = temp_registry.get_model_version("test-model")
        assert default.version == "v1.0.0"

        # Change default
        temp_registry.set_default_version("test-model", "v2.0.0")

        default = temp_registry.get_model_version("test-model")
        assert default.version == "v2.0.0"

    def test_delete_version(self, temp_registry):
        """Test deleting a model version."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v1.0.0",
            path="p1",
        )

        result = temp_registry.delete_version("test-model", "v1.0.0")
        assert result is True

        versions = temp_registry.list_versions("test-model")
        assert len(versions) == 0

    def test_delete_model(self, temp_registry):
        """Test deleting an entire model."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v1",
            path="p1",
        )
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v2",
            path="p2",
        )

        result = temp_registry.delete_model("test-model")
        assert result is True

        models = temp_registry.list_models()
        assert "test-model" not in models

    def test_get_model_path(self, temp_registry):
        """Test getting model path."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v1.0.0",
            path="models/sketcher/v1.0.0/model.pt",
        )

        path = temp_registry.get_model_path("test-model")
        assert path == "models/sketcher/v1.0.0/model.pt"

    def test_find_by_tag(self, temp_registry):
        """Test finding models by tag."""
        temp_registry.register_model(
            model_id="model-1",
            model_type="sketcher",
            version="v1",
            path="p1",
            tags=["production", "stable"],
        )
        temp_registry.register_model(
            model_id="model-2",
            model_type="critic",
            version="v1",
            path="p2",
            tags=["experimental"],
        )

        prod_models = temp_registry.find_by_tag("production")
        assert len(prod_models) == 1
        assert prod_models[0] == ("model-1", "v1")

    def test_find_by_metrics(self, temp_registry):
        """Test finding models by metrics."""
        temp_registry.register_model(
            model_id="model-1",
            model_type="sketcher",
            version="v1",
            path="p1",
            metrics={"accuracy": 0.85},
        )
        temp_registry.register_model(
            model_id="model-2",
            model_type="sketcher",
            version="v1",
            path="p2",
            metrics={"accuracy": 0.92},
        )

        high_acc = temp_registry.find_by_metrics("accuracy", min_value=0.90)
        assert len(high_acc) == 1
        assert high_acc[0][0] == "model-2"

    def test_persistence(self):
        """Test that registry persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"

            # Create registry and add model
            registry1 = ModelRegistry(registry_path=registry_path)
            registry1.register_model(
                model_id="test-model",
                model_type="sketcher",
                version="v1",
                path="p1",
            )

            # Create new registry instance
            registry2 = ModelRegistry(registry_path=registry_path)

            # Should have the same model
            models = registry2.list_models()
            assert "test-model" in models

    def test_export_import(self, temp_registry):
        """Test exporting and importing registry data."""
        temp_registry.register_model(
            model_id="test-model",
            model_type="sketcher",
            version="v1",
            path="p1",
        )

        # Export
        exported = temp_registry.export_registry()
        assert "models" in exported
        assert "test-model" in exported["models"]

        # Import to new registry
        with tempfile.TemporaryDirectory() as tmpdir:
            new_registry = ModelRegistry(registry_path=Path(tmpdir) / "new.json")
            new_registry.import_registry(exported)

            models = new_registry.list_models()
            assert "test-model" in models


# =============================================================================
# Integration Tests
# =============================================================================


class TestStorageRegistryIntegration:
    """Integration tests for storage and registry together."""

    def test_registry_with_storage_client(self):
        """Test using registry with storage client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup storage
            storage_config = StorageConfig(
                backend="local",
                local_path=os.path.join(tmpdir, "storage"),
            )
            storage_client = StorageClient(storage_config)

            # Setup registry
            registry_path = Path(tmpdir) / "registry.json"
            registry = ModelRegistry(
                registry_path=registry_path,
                storage_client=storage_client,
            )

            # Save a model artifact
            model_data = b"fake model weights"
            key = storage_client.save_model(
                model_type="sketcher",
                version="v1.0.0",
                model_data=model_data,
            )

            # Register in registry
            registry.register_model(
                model_id="test-sketcher",
                model_type="sketcher",
                version="v1.0.0",
                path=key,
                metrics={"accuracy": 0.92},
            )

            # Verify we can get the model info
            model = registry.get_model("test-sketcher")
            assert model is not None
            assert model.model_type == "sketcher"

            # Get path and load from storage
            path = registry.get_model_path("test-sketcher")
            assert path == key
