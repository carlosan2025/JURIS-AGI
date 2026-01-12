"""Tests for local PoC configuration and server."""

import os
import pytest
from unittest.mock import patch

from juris_agi.api.local_config import LocalPoCConfig, get_local_config, is_local_poc_mode


class TestLocalPoCConfig:
    """Tests for LocalPoCConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LocalPoCConfig()
        assert config.sync_mode is True
        assert config.redis_enabled is False
        assert config.gpu_enabled is False
        assert config.storage_backend == "local"
        assert config.max_grid_size == 30
        assert config.max_runtime_seconds == 60.0

    def test_from_env_local_poc_mode(self):
        """Test loading config in LOCAL_POC mode."""
        with patch.dict(os.environ, {
            "LOCAL_POC": "true",
            "SYNC_MODE": "true",
            "MAX_GRID_SIZE": "25",
        }, clear=False):
            config = LocalPoCConfig.from_env()
            assert config.sync_mode is True
            assert config.redis_enabled is False
            assert config.max_grid_size == 25

    def test_from_env_async_mode(self):
        """Test async mode configuration."""
        with patch.dict(os.environ, {
            "LOCAL_POC": "true",
            "SYNC_MODE": "false",
            "REDIS_ENABLED": "true",
        }, clear=False):
            config = LocalPoCConfig.from_env()
            assert config.sync_mode is False
            assert config.redis_enabled is True

    def test_sync_mode_forces_redis_disabled(self):
        """Test that sync mode disables Redis."""
        with patch.dict(os.environ, {
            "LOCAL_POC": "true",
            "SYNC_MODE": "true",
            "REDIS_ENABLED": "true",  # Should be ignored
        }, clear=False):
            config = LocalPoCConfig.from_env()
            assert config.sync_mode is True
            assert config.redis_enabled is False

    def test_validate_grid_valid(self):
        """Test grid validation with valid input."""
        config = LocalPoCConfig()
        valid, error = config.validate_grid([[1, 2], [3, 4]])
        assert valid is True
        assert error == ""

    def test_validate_grid_empty(self):
        """Test grid validation rejects empty grid."""
        config = LocalPoCConfig()
        valid, error = config.validate_grid([])
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_grid_too_large(self):
        """Test grid validation rejects oversized grid."""
        config = LocalPoCConfig(max_grid_size=5)
        large_grid = [[0] * 10 for _ in range(10)]
        valid, error = config.validate_grid(large_grid)
        assert valid is False
        assert "exceeds" in error.lower()

    def test_validate_grid_invalid_values(self):
        """Test grid validation rejects invalid cell values."""
        config = LocalPoCConfig()

        # Value too high
        valid, error = config.validate_grid([[10]])
        assert valid is False
        assert "0-9" in error

        # Negative value
        valid, error = config.validate_grid([[-1]])
        assert valid is False
        assert "0-9" in error

        # Non-integer
        valid, error = config.validate_grid([["a"]])
        assert valid is False

    def test_to_dict(self):
        """Test config serialization."""
        config = LocalPoCConfig()
        d = config.to_dict()

        assert d["mode"] == "sync"
        assert d["redis_enabled"] is False
        assert d["gpu_enabled"] is False
        assert "limits" in d
        assert d["limits"]["max_grid_size"] == 30

    def test_traces_dir_property(self):
        """Test traces directory path."""
        config = LocalPoCConfig(runs_dir="./test_runs")
        assert str(config.traces_dir) == "test_runs/traces"

    def test_results_dir_property(self):
        """Test results directory path."""
        config = LocalPoCConfig(runs_dir="./test_runs")
        assert str(config.results_dir) == "test_runs/results"


class TestIsLocalPoCMode:
    """Tests for is_local_poc_mode function."""

    def test_default_is_true(self):
        """Test that LOCAL_POC defaults to true."""
        with patch.dict(os.environ, {}, clear=True):
            # Without LOCAL_POC set, should default to true
            assert is_local_poc_mode() is True

    def test_explicit_true(self):
        """Test explicit LOCAL_POC=true."""
        with patch.dict(os.environ, {"LOCAL_POC": "true"}, clear=False):
            assert is_local_poc_mode() is True

    def test_explicit_false(self):
        """Test explicit LOCAL_POC=false."""
        with patch.dict(os.environ, {"LOCAL_POC": "false"}, clear=False):
            assert is_local_poc_mode() is False
