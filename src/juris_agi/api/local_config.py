"""Local PoC configuration for running JURIS-AGI without cloud dependencies.

This module provides configuration for two local execution modes:
1. Pure Python mode (no Docker, no Redis) - synchronous execution
2. Docker-compose mode - local containers with Redis queue

All cloud features are disabled by default.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LocalPoCConfig:
    """Configuration for local PoC execution."""

    # Execution mode
    sync_mode: bool = True  # True = no Redis, direct execution
    redis_enabled: bool = False
    gpu_enabled: bool = False

    # Storage (always local in PoC)
    storage_backend: str = "local"
    runs_dir: str = "./runs"  # Local runs directory

    # Safety limits (strict for PoC)
    max_grid_size: int = 30  # Max grid dimension
    max_search_expansions: int = 500  # Max beam search expansions
    max_runtime_seconds: float = 60.0  # Max time per job
    max_program_depth: int = 4  # Max synthesized program depth
    max_pending_jobs: int = 10  # Max jobs in queue (async mode)

    # Server settings
    host: str = "127.0.0.1"  # localhost only for PoC
    port: int = 8000
    debug: bool = True

    # Redis (only used if redis_enabled=True)
    redis_url: str = "redis://localhost:6379/0"

    @property
    def traces_dir(self) -> Path:
        """Directory for trace files."""
        return Path(self.runs_dir) / "traces"

    @property
    def results_dir(self) -> Path:
        """Directory for result files."""
        return Path(self.runs_dir) / "results"

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "LocalPoCConfig":
        """Load configuration from environment variables."""
        # Check for explicit LOCAL_POC mode
        local_poc = os.getenv("LOCAL_POC", "true").lower() == "true"

        if local_poc:
            # Force local-only settings
            sync_mode = os.getenv("SYNC_MODE", "true").lower() == "true"
            redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"

            # In sync mode, Redis is always disabled
            if sync_mode:
                redis_enabled = False

            return cls(
                sync_mode=sync_mode,
                redis_enabled=redis_enabled,
                gpu_enabled=os.getenv("GPU_ENABLED", "false").lower() == "true",
                storage_backend="local",
                runs_dir=os.getenv("RUNS_DIR", "./runs"),
                max_grid_size=int(os.getenv("MAX_GRID_SIZE", "30")),
                max_search_expansions=int(os.getenv("MAX_SEARCH_EXPANSIONS", "500")),
                max_runtime_seconds=float(os.getenv("MAX_RUNTIME_SECONDS", "60.0")),
                max_program_depth=int(os.getenv("MAX_PROGRAM_DEPTH", "4")),
                max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "10")),
                host=os.getenv("JURIS_HOST", "127.0.0.1"),
                port=int(os.getenv("JURIS_PORT", "8000")),
                debug=os.getenv("JURIS_DEBUG", "true").lower() == "true",
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            )
        else:
            # Non-PoC mode - use standard config
            return cls(
                sync_mode=os.getenv("SYNC_MODE", "false").lower() == "true",
                redis_enabled=os.getenv("REDIS_ENABLED", "true").lower() == "true",
                gpu_enabled=os.getenv("GPU_ENABLED", "false").lower() == "true",
                storage_backend=os.getenv("STORAGE_BACKEND", "local"),
                runs_dir=os.getenv("RUNS_DIR", "./runs"),
                max_grid_size=int(os.getenv("MAX_GRID_SIZE", "30")),
                max_search_expansions=int(os.getenv("MAX_SEARCH_EXPANSIONS", "1000")),
                max_runtime_seconds=float(os.getenv("MAX_RUNTIME_SECONDS", "120.0")),
                max_program_depth=int(os.getenv("MAX_PROGRAM_DEPTH", "5")),
                max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "100")),
                host=os.getenv("JURIS_HOST", "0.0.0.0"),
                port=int(os.getenv("JURIS_PORT", "8000")),
                debug=os.getenv("JURIS_DEBUG", "false").lower() == "true",
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            )

    def validate_grid(self, grid: list) -> tuple[bool, str]:
        """Validate a grid against PoC limits.

        Returns:
            (is_valid, error_message)
        """
        if not grid:
            return False, "Grid cannot be empty"

        height = len(grid)
        if height > self.max_grid_size:
            return False, f"Grid height {height} exceeds max {self.max_grid_size}"

        for i, row in enumerate(grid):
            if not isinstance(row, list):
                return False, f"Row {i} is not a list"
            width = len(row)
            if width > self.max_grid_size:
                return False, f"Grid width {width} exceeds max {self.max_grid_size}"
            for j, val in enumerate(row):
                if not isinstance(val, int) or val < 0 or val > 9:
                    return False, f"Invalid value at [{i}][{j}]: must be int 0-9"

        return True, ""

    def to_dict(self) -> dict:
        """Convert to dictionary for health endpoint."""
        return {
            "mode": "sync" if self.sync_mode else "async",
            "redis_enabled": self.redis_enabled,
            "gpu_enabled": self.gpu_enabled,
            "storage_backend": self.storage_backend,
            "runs_dir": str(self.runs_dir),
            "limits": {
                "max_grid_size": self.max_grid_size,
                "max_search_expansions": self.max_search_expansions,
                "max_runtime_seconds": self.max_runtime_seconds,
                "max_program_depth": self.max_program_depth,
            },
        }


# Global config instance
_local_config: Optional[LocalPoCConfig] = None


def get_local_config() -> LocalPoCConfig:
    """Get the local PoC configuration."""
    global _local_config
    if _local_config is None:
        _local_config = LocalPoCConfig.from_env()
        _local_config.ensure_dirs()
    return _local_config


def is_local_poc_mode() -> bool:
    """Check if running in local PoC mode."""
    return os.getenv("LOCAL_POC", "true").lower() == "true"
