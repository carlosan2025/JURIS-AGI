"""Model registry for managing versioned model artifacts.

Provides a registry system that reads model metadata from registry.json
and integrates with the storage client for model artifact management.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""

    version: str
    path: str  # Storage key or local path
    created_at: str
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at,
            "description": self.description,
            "metrics": self.metrics,
            "config": self.config,
            "is_default": self.is_default,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            path=data["path"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            description=data.get("description", ""),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            is_default=data.get("is_default", False),
            tags=data.get("tags", []),
        )


@dataclass
class ModelEntry:
    """Represents a model type with multiple versions."""

    name: str
    model_type: str  # "sketcher", "critic", "composer", etc.
    description: str = ""
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    default_version: Optional[str] = None

    def get_default_version(self) -> Optional[ModelVersion]:
        """Get the default version of this model."""
        if self.default_version and self.default_version in self.versions:
            return self.versions[self.default_version]
        # Fall back to version marked as default
        for version in self.versions.values():
            if version.is_default:
                return version
        # Fall back to latest version
        if self.versions:
            return max(self.versions.values(), key=lambda v: v.created_at)
        return None

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        return self.versions.get(version)

    def list_versions(self) -> List[str]:
        """List all available versions."""
        return list(self.versions.keys())

    def add_version(self, version: ModelVersion) -> None:
        """Add a new version."""
        self.versions[version.version] = version
        if version.is_default:
            self.default_version = version.version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "default_version": self.default_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create from dictionary."""
        versions = {}
        for version_id, version_data in data.get("versions", {}).items():
            versions[version_id] = ModelVersion.from_dict(version_data)

        return cls(
            name=data["name"],
            model_type=data["model_type"],
            description=data.get("description", ""),
            versions=versions,
            default_version=data.get("default_version"),
        )


class ModelRegistry:
    """Registry for managing model versions and metadata.

    Reads and writes model metadata from/to a registry.json file,
    and integrates with StorageClient for artifact management.
    """

    def __init__(
        self,
        registry_path: Optional[Union[str, Path]] = None,
        storage_client: Optional[Any] = None,
    ):
        """Initialize the model registry.

        Args:
            registry_path: Path to registry.json file. Defaults to models/registry.json.
            storage_client: Optional StorageClient for artifact operations.
        """
        if registry_path is None:
            registry_path = Path("models/registry.json")
        self.registry_path = Path(registry_path)
        self.storage_client = storage_client
        self._models: Dict[str, ModelEntry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)

                for model_id, model_data in data.get("models", {}).items():
                    self._models[model_id] = ModelEntry.from_dict(model_data)

                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._models = {}
        else:
            logger.info(f"Registry file not found at {self.registry_path}, starting empty")
            self._models = {}

    def _save_registry(self) -> None:
        """Save the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry with {len(self._models)} models")

    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        path: str,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_id: Unique identifier for this model.
            model_type: Type of model (sketcher, critic, etc.).
            version: Version string (e.g., "1.0.0", "v1", "20240115").
            path: Storage key or local path to the model artifact.
            description: Human-readable description.
            metrics: Model performance metrics.
            config: Model configuration/hyperparameters.
            is_default: Whether this should be the default version.
            tags: Tags for filtering/categorization.

        Returns:
            The created ModelVersion.
        """
        if model_id not in self._models:
            self._models[model_id] = ModelEntry(
                name=model_id,
                model_type=model_type,
                description=description,
            )

        model_entry = self._models[model_id]

        version_obj = ModelVersion(
            version=version,
            path=path,
            created_at=datetime.utcnow().isoformat(),
            description=description,
            metrics=metrics or {},
            config=config or {},
            is_default=is_default,
            tags=tags or [],
        )

        model_entry.add_version(version_obj)

        # If this is the first version or marked as default, set as default
        if is_default or model_entry.default_version is None:
            model_entry.default_version = version

        self._save_registry()
        logger.info(f"Registered model {model_id} version {version}")

        return version_obj

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get a model entry by ID."""
        return self._models.get(model_id)

    def get_model_version(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """Get a specific model version.

        Args:
            model_id: Model identifier.
            version: Version string. If None, returns default version.

        Returns:
            ModelVersion if found, None otherwise.
        """
        model = self._models.get(model_id)
        if model is None:
            return None

        if version is None:
            return model.get_default_version()

        return model.get_version(version)

    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """List all registered model IDs.

        Args:
            model_type: Optional filter by model type.

        Returns:
            List of model IDs.
        """
        if model_type is None:
            return list(self._models.keys())

        return [
            model_id
            for model_id, entry in self._models.items()
            if entry.model_type == model_type
        ]

    def list_versions(self, model_id: str) -> List[str]:
        """List all versions of a model."""
        model = self._models.get(model_id)
        if model is None:
            return []
        return model.list_versions()

    def set_default_version(self, model_id: str, version: str) -> bool:
        """Set the default version for a model.

        Returns:
            True if successful, False if model/version not found.
        """
        model = self._models.get(model_id)
        if model is None:
            return False

        if version not in model.versions:
            return False

        # Clear is_default on all versions
        for v in model.versions.values():
            v.is_default = False

        # Set new default
        model.versions[version].is_default = True
        model.default_version = version

        self._save_registry()
        return True

    def delete_version(self, model_id: str, version: str) -> bool:
        """Delete a model version.

        Args:
            model_id: Model identifier.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        model = self._models.get(model_id)
        if model is None:
            return False

        if version not in model.versions:
            return False

        del model.versions[version]

        # Update default if we deleted it
        if model.default_version == version:
            if model.versions:
                model.default_version = next(iter(model.versions.keys()))
            else:
                model.default_version = None

        self._save_registry()
        return True

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions.

        Returns:
            True if deleted, False if not found.
        """
        if model_id not in self._models:
            return False

        del self._models[model_id]
        self._save_registry()
        return True

    def get_model_path(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Optional[str]:
        """Get the storage path for a model version.

        Args:
            model_id: Model identifier.
            version: Version string. If None, uses default version.

        Returns:
            Storage path/key if found, None otherwise.
        """
        model_version = self.get_model_version(model_id, version)
        if model_version is None:
            return None
        return model_version.path

    def load_model_artifact(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Optional[bytes]:
        """Load model artifact bytes using the storage client.

        Args:
            model_id: Model identifier.
            version: Version string. If None, uses default version.

        Returns:
            Model artifact bytes if found and storage client available.
        """
        if self.storage_client is None:
            logger.warning("No storage client configured")
            return None

        model_version = self.get_model_version(model_id, version)
        if model_version is None:
            return None

        try:
            return self.storage_client.get_model(
                model_type=self._models[model_id].model_type,
                version=model_version.version,
            )
        except Exception as e:
            logger.error(f"Failed to load model artifact: {e}")
            return None

    def get_model_url(
        self,
        model_id: str,
        version: Optional[str] = None,
        expiry_seconds: int = 3600,
    ) -> Optional[str]:
        """Get a URL for downloading the model artifact.

        Args:
            model_id: Model identifier.
            version: Version string. If None, uses default version.
            expiry_seconds: URL expiry time for presigned URLs.

        Returns:
            URL string if available, None otherwise.
        """
        if self.storage_client is None:
            logger.warning("No storage client configured")
            return None

        model_version = self.get_model_version(model_id, version)
        if model_version is None:
            return None

        try:
            return self.storage_client.get_model_url(
                model_version.path,
                expiry_seconds=expiry_seconds,
            )
        except Exception as e:
            logger.error(f"Failed to get model URL: {e}")
            return None

    def find_by_tag(self, tag: str) -> List[tuple]:
        """Find all model versions with a specific tag.

        Returns:
            List of (model_id, version) tuples.
        """
        results = []
        for model_id, model in self._models.items():
            for version_id, version in model.versions.items():
                if tag in version.tags:
                    results.append((model_id, version_id))
        return results

    def find_by_metrics(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> List[tuple]:
        """Find model versions by metric criteria.

        Args:
            metric_name: Name of the metric.
            min_value: Minimum value (inclusive).
            max_value: Maximum value (inclusive).

        Returns:
            List of (model_id, version, metric_value) tuples.
        """
        results = []
        for model_id, model in self._models.items():
            for version_id, version in model.versions.items():
                if metric_name in version.metrics:
                    value = version.metrics[metric_name]
                    if min_value is not None and value < min_value:
                        continue
                    if max_value is not None and value > max_value:
                        continue
                    results.append((model_id, version_id, value))
        return results

    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry as a dictionary."""
        return {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }

    def import_registry(self, data: Dict[str, Any], merge: bool = False) -> None:
        """Import registry data.

        Args:
            data: Registry data dictionary.
            merge: If True, merge with existing. If False, replace.
        """
        if not merge:
            self._models = {}

        for model_id, model_data in data.get("models", {}).items():
            self._models[model_id] = ModelEntry.from_dict(model_data)

        self._save_registry()
