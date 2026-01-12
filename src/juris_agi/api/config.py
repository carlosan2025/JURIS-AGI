"""Configuration for API and worker."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    """Configuration for the FastAPI server."""
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10

    # Job settings
    job_timeout_seconds: int = 600  # 10 minutes max
    job_ttl_seconds: int = 3600  # Keep results for 1 hour
    max_pending_jobs: int = 1000

    # Storage
    storage_backend: str = "local"  # "local" or "s3"
    storage_local_path: str = "/tmp/juris_storage"
    s3_bucket: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    presigned_url_expiry: int = 3600  # 1 hour

    # Model registry
    model_registry_path: str = "models/registry.json"

    # Model paths (deprecated, use registry)
    sketcher_model_path: Optional[str] = None
    critic_model_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("JURIS_HOST", "0.0.0.0"),
            port=int(os.getenv("JURIS_PORT", "8000")),
            workers=int(os.getenv("JURIS_WORKERS", "4")),
            debug=os.getenv("JURIS_DEBUG", "false").lower() == "true",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
            job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "600")),
            job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS", "3600")),
            max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "1000")),
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            storage_local_path=os.getenv("STORAGE_LOCAL_PATH", "/tmp/juris_storage"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_endpoint=os.getenv("S3_ENDPOINT"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key=os.getenv("S3_ACCESS_KEY"),
            s3_secret_key=os.getenv("S3_SECRET_KEY"),
            presigned_url_expiry=int(os.getenv("PRESIGNED_URL_EXPIRY", "3600")),
            model_registry_path=os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"),
            sketcher_model_path=os.getenv("SKETCHER_MODEL_PATH"),
            critic_model_path=os.getenv("CRITIC_MODEL_PATH"),
        )


@dataclass
class WorkerConfig:
    """Configuration for the worker."""
    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Queues
    queue_names: list = field(default_factory=lambda: ["juris_high", "juris_default", "juris_low"])

    # Concurrency
    max_concurrent_jobs: int = 1  # GPU jobs should run one at a time

    # Timeouts
    job_timeout_seconds: int = 600
    health_check_interval: int = 30

    # Storage
    storage_backend: str = "local"  # "local" or "s3"
    storage_local_path: str = "/tmp/juris_storage"
    s3_bucket: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    presigned_url_expiry: int = 3600

    # Model registry
    model_registry_path: str = "models/registry.json"

    # Model paths (deprecated, use registry)
    sketcher_model_path: Optional[str] = None
    critic_model_path: Optional[str] = None

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load configuration from environment variables."""
        queue_names = os.getenv("QUEUE_NAMES", "juris_high,juris_default,juris_low").split(",")

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            queue_names=queue_names,
            max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "1")),
            job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "600")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            storage_local_path=os.getenv("STORAGE_LOCAL_PATH", "/tmp/juris_storage"),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_endpoint=os.getenv("S3_ENDPOINT"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key=os.getenv("S3_ACCESS_KEY"),
            s3_secret_key=os.getenv("S3_SECRET_KEY"),
            presigned_url_expiry=int(os.getenv("PRESIGNED_URL_EXPIRY", "3600")),
            model_registry_path=os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"),
            sketcher_model_path=os.getenv("SKETCHER_MODEL_PATH"),
            critic_model_path=os.getenv("CRITIC_MODEL_PATH"),
            device=os.getenv("DEVICE", "auto"),
        )
