import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from app.services.detection import DetectionService
from app.db.models import Coin

@pytest.fixture
def mock_model():
    """Mock YOLO model for testing"""
    model = Mock()
    return model

@pytest.fixture
def detection_service(mock_model):
    """Create DetectionService with mocked model"""
    with patch('app.services.detection.YOLO', return_value=mock_model):
        service = DetectionService('dummy_path.pt')
        service.model = mock_model
    return service

@pytest.fixture
def single_detection_result():
    """Mock result with a single detection"""
    result = Mock()
    # Single box: [x1, y1, x2, y2] = [10, 20, 110, 120]
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 110, 120]])
    return result

@pytest.fixture
def multi_detection_result():
    """Mock result with multiple detections"""
    result = Mock()
    # Two boxes with different positions
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [10, 20, 110, 120],   # top-left coin
        [150, 50, 250, 150]   # bottom-right coin
    ])
    return result

def test_process_image_single_detection(detection_service, single_detection_result):
    """Test processing image with single coin detection"""
    detection_service.model.return_value = [single_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    assert len(coins) == 1
    assert isinstance(coins[0], Coin)
    assert coins[0].id == 'img_001_coin_001'
    assert coins[0].image_id == 'img_001'
    assert coins[0].bbox_w == 100  # x2 - x1 = 110 - 10
    assert coins[0].bbox_h == 100  # y2 - y1 = 120 - 20

def test_process_image_multiple_detections(detection_service, multi_detection_result):
    """Test processing image with multiple coin detections"""
    detection_service.model.return_value = [multi_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_002')
    
    assert len(coins) == 2
    assert coins[0].id == 'img_002_coin_001'
    assert coins[1].id == 'img_002_coin_002'

def test_coin_center_calculation(detection_service, single_detection_result):
    """Test that coin center is calculated correctly"""
    detection_service.model.return_value = [single_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    # Center should be (x1 + w/2, y1 + h/2) = (10 + 50, 20 + 50) = (60, 70)
    assert coins[0].center_x == 60.0
    assert coins[0].center_y == 70.0

def test_radius_calculation(detection_service, single_detection_result):
    """Test radius is max(width, height) / 2"""
    detection_service.model.return_value = [single_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    # For 100x100 square: radius should be 100/2 = 50
    assert coins[0].radius == 50.0

def test_slant_detection(detection_service):
    """Test that slanted coins are correctly identified"""
    result = Mock()
    # Slanted coin: width=120, height=80 (aspect ratio = 1.5 > 1.2)
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 130, 100]])
    detection_service.model.return_value = [result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    assert coins[0].is_slanted == True
    assert coins[0].bbox_w == 120
    assert coins[0].bbox_h == 80

def test_non_slanted_coin(detection_service):
    """Test that non-slanted coins are correctly identified"""
    result = Mock()
    # Nearly square: width=100, height=95 (aspect ratio = 1.05)
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 110, 115]])
    detection_service.model.return_value = [result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    assert coins[0].is_slanted == False

def test_spatial_sorting_order(detection_service, multi_detection_result):
    """Test that detections are sorted spatially (top-to-bottom, left-to-right)"""
    detection_service.model.return_value = [multi_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    # Both coins already in correct order (y1=20 before y1=50)
    assert coins[0].bbox_x == 10.0   # First coin is top-left
    assert coins[1].bbox_x == 150.0  # Second coin is bottom-right

def test_spatial_sorting_reversed_input(detection_service):
    """Test spatial sorting even when detections come in reverse order"""
    result = Mock()
    # Bottom-right coin first, then top-left
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [150, 50, 250, 150],   # bottom-right (will be sorted second)
        [10, 20, 110, 120]     # top-left (will be sorted first)
    ])
    detection_service.model.return_value = [result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    # After sorting, top-left should be first
    assert coins[0].bbox_x == 10.0
    assert coins[1].bbox_x == 150.0

def test_no_detections(detection_service):
    """Test handling of image with no coin detections"""
    result = Mock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([])
    detection_service.model.return_value = [result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    assert len(coins) == 0
    assert isinstance(coins, list)

def test_coin_id_format(detection_service, multi_detection_result):
    """Test that coin IDs follow the correct format: image_id_coin_NNN"""
    detection_service.model.return_value = [multi_detection_result]
    
    coins = detection_service.process_image('test_img_123', 'test_img_123')
    
    assert coins[0].id == 'test_img_123_coin_001'
    assert coins[1].id == 'test_img_123_coin_002'

def test_bbox_conversion_from_xyxy_to_xywh(detection_service):
    """Test correct conversion from xyxy format to xywh (COCO format)"""
    result = Mock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[50, 60, 200, 180]])
    detection_service.model.return_value = [result]
    
    coins = detection_service.process_image('test.jpg', 'img_001')
    
    assert coins[0].bbox_x == 50.0   # x1
    assert coins[0].bbox_y == 60.0   # y1
    assert coins[0].bbox_w == 150.0  # x2 - x1 = 200 - 50
    assert coins[0].bbox_h == 120.0  # y2 - y1 = 180 - 60

def test_all_coins_have_image_id(detection_service, multi_detection_result):
    """Test that all returned coins have the correct image_id"""
    detection_service.model.return_value = [multi_detection_result]
    
    coins = detection_service.process_image('test.jpg', 'img_xyz')
    
    for coin in coins:
        assert coin.image_id == 'img_xyz'

def test_returns_coin_instances(detection_service, multi_detection_result):
    """Test that process_image returns SQLModel Coin instances"""
    detection_service.model.return_value = [multi_detection_result]

    coins = detection_service.process_image('test.jpg', 'img_001')

    assert isinstance(coins, list)
    assert all(isinstance(coin, Coin) for coin in coins)
