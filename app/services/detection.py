import uuid
import os
from pathlib import Path
from typing import List
from ultralytics import YOLO
from app.services.geometry import CoinGeometry
from app.models import Coin

class DetectionService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

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

_default_model_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'yolov8n.pt')
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(_default_model_path).resolve()))
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

detector = DetectionService(MODEL_PATH)  # keep it singleton