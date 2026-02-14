"""
Unit tests for application configuration.
"""
import os
import pytest
from pathlib import Path
from app.core.config import Settings, get_settings


def test_settings_default_values():
    """Test that settings have proper default values."""
    settings = Settings()

    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.PROJECT_NAME == "Coin Detection API"
    assert settings.VERSION == "1.0.0"
    assert settings.CONFIDENCE_THRESHOLD == 0.25
    assert settings.SLANT_THRESHOLD_LOW == 0.8
    assert settings.SLANT_THRESHOLD_HIGH == 1.2


def test_settings_paths_exist():
    """Test that settings paths are valid Path objects."""
    settings = Settings()

    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.APP_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.UPLOAD_DIR, Path)
    assert isinstance(settings.ARTIFACTS_DIR, Path)
    assert isinstance(settings.DB_PATH, Path)
    assert isinstance(settings.MODEL_PATH, Path)


def test_settings_creates_directories():
    """Test that settings initialization creates required directories."""
    settings = Settings()

    # These directories should exist after Settings initialization
    assert settings.DATA_DIR.exists()
    assert settings.UPLOAD_DIR.exists()


def test_settings_database_url():
    """Test database URL generation."""
    settings = Settings()

    db_url = settings.database_url
    assert db_url.startswith("sqlite:///")
    assert str(settings.DB_PATH) in db_url


def test_get_settings_returns_cached_instance():
    """Test that get_settings returns cached singleton."""
    settings1 = get_settings()
    settings2 = get_settings()

    # Should be same instance (cached)
    assert settings1 is settings2


def test_settings_with_env_override(monkeypatch, tmp_path):
    """Test that environment variables override defaults."""
    test_data_dir = tmp_path / "custom_data"
    test_data_dir.mkdir()

    # Set environment variable
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))

    # Create new settings instance (not cached)
    settings = Settings()

    assert settings.DATA_DIR == test_data_dir
    assert settings.UPLOAD_DIR == test_data_dir / "uploads"
    assert settings.DB_PATH == test_data_dir / "database.db"


def test_settings_model_path_override(monkeypatch, tmp_path):
    """Test MODEL_PATH environment override."""
    test_model_path = tmp_path / "custom_model.pt"

    monkeypatch.setenv("MODEL_PATH", str(test_model_path))

    settings = Settings()
    assert settings.MODEL_PATH == test_model_path


def test_settings_model_path_default():
    """Test MODEL_PATH uses default if environment not set."""
    settings = Settings()

    assert settings.MODEL_PATH.name == "yolov8n-coin-finetuned.pt"
    assert "models" in str(settings.MODEL_PATH)
    assert settings.MODEL_PATH.parent.name == "models"
