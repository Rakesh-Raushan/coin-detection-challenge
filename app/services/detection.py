"""Detection service for coin detection using YOLO."""
from pathlib import Path
from typing import List, Optional
import threading
from ultralytics import YOLO
from app.core.config import settings
from app.services.geometry import CoinGeometry
from app.db.models import Coin


class DetectionService:
    def __init__(self, MODEL_PATH: Path):
        self.model = YOLO(str(MODEL_PATH))

    def process_image(self, image_path: str, image_id: str) -> List[Coin]:
        """
        Runs inference, applies geometric logic, sorts spatially,
        and returns a list of Coin objects ready for DB persistence.
        """

        # run inference - returns a list with one Result object per image
        results = self.model(image_path)
        
        # Extract the first (and only) result for single image inference
        result = results[0]

        # extract detections from the result
        detections = []
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        
        for box in boxes:  # This now iterates over all detected boxes
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            bbox = [float(x1), float(y1), float(w), float(h)]

            # applying elliptical geometry logic to determine radius and slant
            geo_data = CoinGeometry.analyze_detection(bbox)

            detections.append({
                "bbox": bbox,
                "geo": geo_data,
                # Sort Key: Y (top down) then X (left to right)
                "sort_key": (y1, x1)
            })

        # sort spatially (make the IDs deterministic based on position in the image)
        detections.sort(key=lambda x: x['sort_key'])

        # create DB objects
        coin_objects = []
        for idx, det in enumerate(detections):
            geo = det['geo']
            bbox = det['bbox']

            # generate deterministic id of form img_id_coin_001
            unique_id = f"{image_id}_coin_{idx+1:03d}"

            # create the sql model object for persistence
            coin = Coin(
                id=unique_id,
                image_id=image_id,
                center_x=geo['center_point'][0],
                center_y=geo['center_point'][1],
                radius=geo['radius'],
                is_slanted=geo['is_slanted'],
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_w=bbox[2],
                bbox_h=bbox[3]
            )
            coin_objects.append(coin)

        return coin_objects


# Lazy-loading singleton pattern with thread safety
_detector_instance: Optional[DetectionService] = None
_detector_lock = threading.Lock()
_model_load_error: Optional[str] = None


def get_detector() -> Optional[DetectionService]:
    """
    Lazy-load the detector singleton. Returns None if model is unavailable.
    Thread-safe initialization using double-checked locking pattern.
    """
    global _detector_instance, _model_load_error

    # Fast path: if already initialized, return immediately
    if _detector_instance is not None:
        return _detector_instance

    # If we've already tried and failed, don't retry
    if _model_load_error is not None:
        return None

    # Slow path: acquire lock and initialize
    with _detector_lock:
        # Double-check: another thread might have initialized while we waited
        if _detector_instance is not None:
            return _detector_instance

        if _model_load_error is not None:
            return None

        # Attempt initialization
        try:
            if not settings.MODEL_PATH.exists():
                _model_load_error = f"Model not found at {settings.MODEL_PATH}"
                return None

            _detector_instance = DetectionService(settings.MODEL_PATH)
            return _detector_instance
        except Exception as e:
            _model_load_error = f"Failed to load model: {str(e)}"
            return None


def get_model_status() -> dict:
    """
    Returns the current model availability status.
    """
    global _detector_instance, _model_load_error

    if _detector_instance is not None:
        return {
            "available": True,
            "model_path": str(settings.MODEL_PATH),
            "status": "loaded"
        }
    elif _model_load_error is not None:
        return {
            "available": False,
            "model_path": str(settings.MODEL_PATH),
            "status": "error",
            "error": _model_load_error
        }
    else:
        return {
            "available": False,
            "model_path": str(settings.MODEL_PATH),
            "status": "not_loaded"
        }