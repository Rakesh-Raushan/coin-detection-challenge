import json
from pathlib import Path
from typing import Dict, List
import numpy as np

def load_coco_annotations(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def build_image_annotation_map(coco_dict: Dict) -> Dict:
    image_map = {img["id"]: img for img in coco_dict["images"]}

    annotations_map = {}
    for ann in coco_dict["annotations"]:
        image_id = ann["image_id"]
        annotations_map.setdefault(image_id, []).append(ann)

    return image_map, annotations_map


def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    # COCO: [x, y, width, height]
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0
