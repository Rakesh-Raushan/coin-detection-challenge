"""
Shared pytest fixtures for the test suite.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing without loading actual weights."""
    model = Mock()
    return model


@pytest.fixture
def mock_single_detection():
    """Mock YOLO result with a single detection."""
    result = Mock()
    # Single box: [x1, y1, x2, y2] = [10, 20, 110, 120] -> 100x100 box
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.0, 20.0, 110.0, 120.0]
    ])
    return result


@pytest.fixture
def mock_multi_detection():
    """Mock YOLO result with multiple detections."""
    result = Mock()
    # Two boxes at different positions
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.0, 20.0, 110.0, 120.0],    # top-left coin (100x100)
        [150.0, 50.0, 250.0, 150.0],   # bottom-right coin (100x100)
    ])
    return result


@pytest.fixture
def mock_slanted_detection():
    """Mock YOLO result with a slanted coin (non-square bbox)."""
    result = Mock()
    # Slanted coin: width=120, height=80 (aspect ratio = 1.5)
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.0, 20.0, 130.0, 100.0]  # 120x80 box
    ])
    return result


@pytest.fixture
def mock_empty_detection():
    """Mock YOLO result with no detections."""
    result = Mock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([])
    return result


# =============================================================================
# Detection Service Fixtures
# =============================================================================

@pytest.fixture
def detection_service(mock_yolo_model):
    """Create DetectionService with mocked model."""
    with patch('app.services.detection.YOLO', return_value=mock_yolo_model):
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.MODEL_PATH = Path('/mock/model.pt')
            mock_settings.MODEL_PATH.exists = Mock(return_value=True)

            from app.services.detection import DetectionService
            service = DetectionService(Path('/mock/model.pt'))
            service.model = mock_yolo_model
            return service


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Generate a sample test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Sample bounding box in [x, y, width, height] format."""
    return [10, 20, 100, 100]


@pytest.fixture
def sample_slanted_bbox():
    """Sample slanted bounding box (aspect ratio > 1.2)."""
    return [10, 20, 120, 80]


# =============================================================================
# Database Fixtures (for integration tests)
# =============================================================================

@pytest.fixture
def test_image_id():
    """Generate a test image ID."""
    return "test1234"


@pytest.fixture
def test_coin_id(test_image_id):
    """Generate a test coin ID."""
    return f"{test_image_id}_coin_001"
