"""
Storage client for artifact handling.

Supports:
- Local filesystem storage
- S3-compatible storage (AWS S3, MinIO, etc.)

Artifacts are organized as:
- traces/{date}/{task_id}/{job_id}.json
- results/{date}/{task_id}/{job_id}.json
- models/{model_type}/{version}/model.pt
"""

import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO, Union
from urllib.parse import urlparse
import hashlib

# Optional S3 support
try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None
    ClientError = Exception


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StorageConfig:
    """Configuration for storage backend."""
    # Backend type: "local" or "s3"
    backend: str = "local"

    # Local storage
    local_path: str = "/tmp/juris_storage"

    # S3 configuration
    s3_bucket: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_use_ssl: bool = True

    # URL settings
    presigned_url_expiry: int = 3600  # 1 hour
    public_base_url: Optional[str] = None  # For public URLs

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load configuration from environment variables."""
        backend = os.getenv("STORAGE_BACKEND", "local")

        return cls(
            backend=backend,
            local_path=os.getenv("STORAGE_LOCAL_PATH", "/tmp/juris_storage"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_endpoint=os.getenv("S3_ENDPOINT"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key=os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID"),
            s3_secret_key=os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
            s3_use_ssl=os.getenv("S3_USE_SSL", "true").lower() == "true",
            presigned_url_expiry=int(os.getenv("PRESIGNED_URL_EXPIRY", "3600")),
            public_base_url=os.getenv("PUBLIC_BASE_URL"),
        )


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""
    key: str
    size: int
    content_type: str
    created_at: datetime
    etag: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "etag": self.etag,
            "metadata": self.metadata,
        }


# =============================================================================
# Abstract Storage Interface
# =============================================================================

class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def put(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO],
        content_type: str = "application/json",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store an artifact."""
        pass

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Retrieve an artifact."""
        pass

    @abstractmethod
    def get_json(self, key: str) -> Dict[str, Any]:
        """Retrieve and parse a JSON artifact."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an artifact exists."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an artifact."""
        pass

    @abstractmethod
    def list(self, prefix: str) -> List[ArtifactMetadata]:
        """List artifacts with a given prefix."""
        pass

    @abstractmethod
    def get_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Get a URL for accessing the artifact."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[ArtifactMetadata]:
        """Get metadata for an artifact."""
        pass


# =============================================================================
# Local Filesystem Backend
# =============================================================================

class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.base_path = Path(config.local_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get the full path for a key."""
        # Sanitize key to prevent path traversal
        safe_key = key.lstrip("/").replace("..", "")
        return self.base_path / safe_key

    def _compute_etag(self, data: bytes) -> str:
        """Compute ETag (MD5 hash) for data."""
        return hashlib.md5(data).hexdigest()

    def put(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO],
        content_type: str = "application/json",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store an artifact locally."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif hasattr(data, "read"):
            data_bytes = data.read()
        else:
            data_bytes = data

        # Write data
        with open(path, "wb") as f:
            f.write(data_bytes)

        # Write metadata
        meta_path = path.with_suffix(path.suffix + ".meta")
        meta = {
            "content_type": content_type,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return ArtifactMetadata(
            key=key,
            size=len(data_bytes),
            content_type=content_type,
            created_at=datetime.utcnow(),
            etag=self._compute_etag(data_bytes),
            metadata=metadata or {},
        )

    def get(self, key: str) -> bytes:
        """Retrieve an artifact."""
        path = self._get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")

        with open(path, "rb") as f:
            return f.read()

    def get_json(self, key: str) -> Dict[str, Any]:
        """Retrieve and parse a JSON artifact."""
        data = self.get(key)
        return json.loads(data.decode("utf-8"))

    def exists(self, key: str) -> bool:
        """Check if an artifact exists."""
        return self._get_path(key).exists()

    def delete(self, key: str) -> bool:
        """Delete an artifact."""
        path = self._get_path(key)
        meta_path = path.with_suffix(path.suffix + ".meta")

        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()

        return deleted

    def list(self, prefix: str) -> List[ArtifactMetadata]:
        """List artifacts with a given prefix."""
        prefix_path = self._get_path(prefix)
        results = []

        if prefix_path.is_dir():
            search_path = prefix_path
            pattern = "**/*"
        else:
            search_path = prefix_path.parent
            pattern = f"{prefix_path.name}*"

        if not search_path.exists():
            return []

        for path in search_path.glob(pattern):
            if path.is_file() and not path.suffix.endswith(".meta"):
                key = str(path.relative_to(self.base_path))
                meta = self.get_metadata(key)
                if meta:
                    results.append(meta)

        return results

    def get_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Get a URL for accessing the artifact."""
        path = self._get_path(key)

        if self.config.public_base_url:
            return f"{self.config.public_base_url.rstrip('/')}/{key}"

        return f"file://{path.absolute()}"

    def get_metadata(self, key: str) -> Optional[ArtifactMetadata]:
        """Get metadata for an artifact."""
        path = self._get_path(key)
        if not path.exists():
            return None

        meta_path = path.with_suffix(path.suffix + ".meta")

        content_type = "application/octet-stream"
        created_at = datetime.fromtimestamp(path.stat().st_mtime)
        metadata = {}

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                content_type = meta.get("content_type", content_type)
                created_at = datetime.fromisoformat(meta.get("created_at", created_at.isoformat()))
                metadata = meta.get("metadata", {})

        return ArtifactMetadata(
            key=key,
            size=path.stat().st_size,
            content_type=content_type,
            created_at=created_at,
            etag=self._compute_etag(path.read_bytes()),
            metadata=metadata,
        )


# =============================================================================
# S3 Storage Backend
# =============================================================================

class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend."""

    def __init__(self, config: StorageConfig):
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

        self.config = config

        # Create S3 client
        self.client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint,
            region_name=config.s3_region,
            aws_access_key_id=config.s3_access_key,
            aws_secret_access_key=config.s3_secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            use_ssl=config.s3_use_ssl,
        )

        self.bucket = config.s3_bucket

    def put(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO],
        content_type: str = "application/json",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store an artifact in S3."""
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif hasattr(data, "read"):
            data_bytes = data.read()
        else:
            data_bytes = data

        # Upload to S3
        extra_args = {
            "ContentType": content_type,
        }
        if metadata:
            extra_args["Metadata"] = metadata

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data_bytes,
            **extra_args,
        )

        return ArtifactMetadata(
            key=key,
            size=len(data_bytes),
            content_type=content_type,
            created_at=datetime.utcnow(),
            etag=hashlib.md5(data_bytes).hexdigest(),
            metadata=metadata or {},
        )

    def get(self, key: str) -> bytes:
        """Retrieve an artifact from S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Artifact not found: {key}")
            raise

    def get_json(self, key: str) -> Dict[str, Any]:
        """Retrieve and parse a JSON artifact."""
        data = self.get(key)
        return json.loads(data.decode("utf-8"))

    def exists(self, key: str) -> bool:
        """Check if an artifact exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> bool:
        """Delete an artifact from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list(self, prefix: str) -> List[ArtifactMetadata]:
        """List artifacts with a given prefix."""
        results = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append(ArtifactMetadata(
                    key=obj["Key"],
                    size=obj["Size"],
                    content_type="application/octet-stream",  # Not available from list
                    created_at=obj["LastModified"],
                    etag=obj["ETag"].strip('"'),
                ))

        return results

    def get_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Get a presigned URL for accessing the artifact."""
        expiry = expiry or self.config.presigned_url_expiry

        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expiry,
        )

        return url

    def get_metadata(self, key: str) -> Optional[ArtifactMetadata]:
        """Get metadata for an artifact."""
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return ArtifactMetadata(
                key=key,
                size=response["ContentLength"],
                content_type=response.get("ContentType", "application/octet-stream"),
                created_at=response["LastModified"],
                etag=response["ETag"].strip('"'),
                metadata=response.get("Metadata", {}),
            )
        except ClientError:
            return None


# =============================================================================
# Storage Client (High-Level API)
# =============================================================================

class StorageClient:
    """
    High-level storage client for JURIS-AGI artifacts.

    Provides methods for storing and retrieving:
    - Execution traces
    - Job results
    - Model files
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig.from_env()

        # Create backend
        if self.config.backend == "s3" and self.config.s3_bucket:
            self.backend = S3StorageBackend(self.config)
        else:
            self.backend = LocalStorageBackend(self.config)

    def _get_date_prefix(self) -> str:
        """Get date prefix for organizing artifacts."""
        return datetime.utcnow().strftime("%Y/%m/%d")

    # =========================================================================
    # Trace Storage
    # =========================================================================

    def save_trace(
        self,
        job_id: str,
        task_id: str,
        trace_data: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save an execution trace.

        Returns the storage key.
        """
        date_prefix = self._get_date_prefix()
        key = f"traces/{date_prefix}/{task_id}/{job_id}.json"

        self.backend.put(
            key=key,
            data=json.dumps(trace_data, indent=2, default=str),
            content_type="application/json",
            metadata=metadata,
        )

        return key

    def get_trace(self, job_id: str, task_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve an execution trace.

        If date is not provided, searches recent dates.
        """
        if date:
            key = f"traces/{date}/{task_id}/{job_id}.json"
            return self.backend.get_json(key)

        # Search recent dates
        for days_ago in range(7):
            search_date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y/%m/%d")
            key = f"traces/{search_date}/{task_id}/{job_id}.json"
            if self.backend.exists(key):
                return self.backend.get_json(key)

        raise FileNotFoundError(f"Trace not found for job {job_id}")

    def get_trace_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Get URL for accessing a trace."""
        return self.backend.get_url(key, expiry)

    # =========================================================================
    # Result Storage
    # =========================================================================

    def save_result(
        self,
        job_id: str,
        task_id: str,
        result_data: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save job results.

        Returns the storage key.
        """
        date_prefix = self._get_date_prefix()
        key = f"results/{date_prefix}/{task_id}/{job_id}.json"

        self.backend.put(
            key=key,
            data=json.dumps(result_data, indent=2, default=str),
            content_type="application/json",
            metadata=metadata,
        )

        return key

    def get_result(self, job_id: str, task_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve job results.

        If date is not provided, searches recent dates.
        """
        if date:
            key = f"results/{date}/{task_id}/{job_id}.json"
            return self.backend.get_json(key)

        # Search recent dates
        for days_ago in range(7):
            search_date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y/%m/%d")
            key = f"results/{search_date}/{task_id}/{job_id}.json"
            if self.backend.exists(key):
                return self.backend.get_json(key)

        raise FileNotFoundError(f"Result not found for job {job_id}")

    def get_result_url(self, key: str, expiry: Optional[int] = None) -> str:
        """Get URL for accessing a result."""
        return self.backend.get_url(key, expiry)

    # =========================================================================
    # Model Storage
    # =========================================================================

    def save_model(
        self,
        model_type: str,
        version: str,
        model_data: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a model file.

        Returns the storage key.
        """
        key = f"models/{model_type}/{version}/model.pt"

        self.backend.put(
            key=key,
            data=model_data,
            content_type="application/octet-stream",
            metadata=metadata,
        )

        return key

    def get_model(self, model_type: str, version: str) -> bytes:
        """Retrieve a model file."""
        key = f"models/{model_type}/{version}/model.pt"
        return self.backend.get(key)

    def get_model_url(self, model_type: str, version: str, expiry: Optional[int] = None) -> str:
        """Get URL for downloading a model."""
        key = f"models/{model_type}/{version}/model.pt"
        return self.backend.get_url(key, expiry)

    def list_model_versions(self, model_type: str) -> List[str]:
        """List available versions for a model type."""
        prefix = f"models/{model_type}/"
        artifacts = self.backend.list(prefix)

        versions = set()
        for artifact in artifacts:
            # Extract version from key: models/{type}/{version}/model.pt
            parts = artifact.key.split("/")
            if len(parts) >= 3:
                versions.add(parts[2])

        return sorted(versions)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_recent_traces(self, task_id: Optional[str] = None, days: int = 7) -> List[ArtifactMetadata]:
        """List recent traces."""
        results = []

        for days_ago in range(days):
            date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y/%m/%d")
            prefix = f"traces/{date}/"
            if task_id:
                prefix += f"{task_id}/"

            results.extend(self.backend.list(prefix))

        return results

    def list_recent_results(self, task_id: Optional[str] = None, days: int = 7) -> List[ArtifactMetadata]:
        """List recent results."""
        results = []

        for days_ago in range(days):
            date = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y/%m/%d")
            prefix = f"results/{date}/"
            if task_id:
                prefix += f"{task_id}/"

            results.extend(self.backend.list(prefix))

        return results

    def cleanup_old_artifacts(self, days: int = 30) -> int:
        """
        Delete artifacts older than specified days.

        Returns number of deleted artifacts.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = 0

        for prefix in ["traces/", "results/"]:
            for artifact in self.backend.list(prefix):
                if artifact.created_at < cutoff:
                    if self.backend.delete(artifact.key):
                        deleted += 1

        return deleted


# =============================================================================
# Factory Function
# =============================================================================

def get_storage_client(config: Optional[StorageConfig] = None) -> StorageClient:
    """Get a storage client instance."""
    return StorageClient(config)
