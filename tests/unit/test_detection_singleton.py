"""
Unit tests for detection service singleton and error handling.
Tests the lazy-loading pattern, thread safety, and error scenarios.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from threading import Thread
import time

from app.services import detection
from app.services.detection import DetectionService, get_detector, get_model_status


@pytest.fixture(autouse=True)
def reset_detector_singleton():
    """Reset the global detector singleton before each test."""
    detection._detector_instance = None
    detection._model_load_error = None
    yield
    # Clean up after test
    detection._detector_instance = None
    detection._model_load_error = None


# ============================================================================
# Singleton Lazy Loading Tests
# ============================================================================


def test_get_detector_loads_model_once():
    """Test that get_detector creates singleton instance."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True

        with patch("app.services.detection.DetectionService") as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance

            # First call should create instance
            detector1 = get_detector()
            # Second call should return same instance
            detector2 = get_detector()

            # Should be the same instance
            assert detector1 is detector2
            # DetectionService constructor should only be called once
            assert mock_service.call_count == 1


def test_get_detector_returns_none_when_model_missing():
    """Test get_detector returns None when model file doesn't exist."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda self: "/fake/path/model.pt"

        detector = get_detector()

        assert detector is None
        assert detection._model_load_error is not None
        assert "Model not found" in detection._model_load_error


def test_get_detector_caches_error_state():
    """Test that get_detector caches error and doesn't retry."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda self: "/fake/path/model.pt"

        # First call fails and caches error
        detector1 = get_detector()
        assert detector1 is None

        # Second call should return None immediately without checking file again
        detector2 = get_detector()
        assert detector2 is None

        # exists() should only be called once (first attempt)
        assert mock_path.exists.call_count == 1


def test_get_detector_handles_model_initialization_exception():
    """Test get_detector handles exceptions during model loading."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: "/fake/path/model.pt"

        with patch("app.services.detection.DetectionService") as mock_service:
            # Simulate exception during model initialization
            mock_service.side_effect = RuntimeError("CUDA not available")

            detector = get_detector()

            assert detector is None
            assert detection._model_load_error is not None
            assert "Failed to load model" in detection._model_load_error
            assert "CUDA not available" in detection._model_load_error


def test_get_detector_thread_safety():
    """Test that get_detector is thread-safe under concurrent access."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True

        call_count = 0
        created_instances = []

        def mock_detection_service_init(model_path):
            nonlocal call_count
            call_count += 1
            # Simulate slow initialization
            time.sleep(0.01)
            instance = Mock()
            instance.model_path = model_path
            created_instances.append(instance)
            return instance

        with patch("app.services.detection.DetectionService", side_effect=mock_detection_service_init):
            results = []

            def get_detector_thread():
                results.append(get_detector())

            # Create 10 threads that all try to get detector simultaneously
            threads = [Thread(target=get_detector_thread) for _ in range(10)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All threads should get the same instance
            assert len(set(id(r) for r in results)) == 1

            # DetectionService should only be initialized once despite concurrent calls
            assert call_count == 1


# ============================================================================
# Model Status Tests
# ============================================================================


def test_get_model_status_when_loaded():
    """Test get_model_status returns 'loaded' when model is initialized."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: "/test/model.pt"

        with patch("app.services.detection.DetectionService"):
            # Load the detector
            get_detector()

            status = get_model_status()

            assert status["available"] is True
            assert status["status"] == "loaded"
            assert status["model_path"] == "/test/model.pt"
            assert "error" not in status


def test_get_model_status_when_error():
    """Test get_model_status returns 'error' when model failed to load."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda self: "/missing/model.pt"

        # Try to load detector (will fail)
        get_detector()

        status = get_model_status()

        assert status["available"] is False
        assert status["status"] == "error"
        assert status["model_path"] == "/missing/model.pt"
        assert "error" in status
        assert "Model not found" in status["error"]


def test_get_model_status_when_not_loaded():
    """Test get_model_status returns 'not_loaded' before any load attempt."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.__str__ = lambda self: "/test/model.pt"

        # Don't call get_detector() - check status immediately
        status = get_model_status()

        assert status["available"] is False
        assert status["status"] == "not_loaded"
        assert status["model_path"] == "/test/model.pt"
        assert "error" not in status


def test_get_model_status_with_exception_during_load():
    """Test get_model_status after exception during model initialization."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: "/test/model.pt"

        with patch("app.services.detection.DetectionService") as mock_service:
            mock_service.side_effect = ValueError("Invalid model format")

            # Try to load (will fail with exception)
            get_detector()

            status = get_model_status()

            assert status["available"] is False
            assert status["status"] == "error"
            assert "error" in status
            assert "Failed to load model" in status["error"]
            assert "Invalid model format" in status["error"]


# ============================================================================
# Edge Cases
# ============================================================================


def test_get_detector_double_check_locking_race_condition():
    """Test double-checked locking handles race condition correctly."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = True

        call_order = []

        class SlowMockService:
            def __init__(self, model_path):
                call_order.append("init_start")
                time.sleep(0.05)  # Simulate slow init
                call_order.append("init_end")
                self.model_path = model_path

        with patch("app.services.detection.DetectionService", SlowMockService):
            results = []

            def thread1_func():
                # This thread will win the lock and initialize
                results.append(("thread1", get_detector()))

            def thread2_func():
                # This thread will wait and use the initialized instance
                time.sleep(0.01)  # Ensure thread1 starts first
                results.append(("thread2", get_detector()))

            t1 = Thread(target=thread1_func)
            t2 = Thread(target=thread2_func)

            t1.start()
            t2.start()

            t1.join()
            t2.join()

            # Both threads should get the same instance
            assert results[0][1] is results[1][1]

            # Initialization should only happen once
            assert call_order.count("init_start") == 1
            assert call_order.count("init_end") == 1


def test_detector_with_empty_model_path():
    """Test behavior when MODEL_PATH is empty string."""
    with patch("app.services.detection.settings.MODEL_PATH", Path("")):
        with patch.object(Path, "exists", return_value=False):
            detector = get_detector()

            assert detector is None
            assert detection._model_load_error is not None


def test_multiple_error_scenarios_cached():
    """Test that once error is cached, no further attempts are made."""
    with patch("app.services.detection.settings.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda self: "/fake/model.pt"

        # First attempt
        get_detector()
        first_error = detection._model_load_error

        # Change the mock to return True (simulating file now exists)
        mock_path.exists.return_value = True

        # Second attempt should still return None without checking
        detector = get_detector()

        assert detector is None
        # Error should be unchanged (not checked again)
        assert detection._model_load_error == first_error
