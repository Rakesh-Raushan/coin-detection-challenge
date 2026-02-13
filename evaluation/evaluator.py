# app/evaluation/evaluator.py

import json
from pathlib import Path
from typing import List, Dict

import cv2
from ultralytics import YOLO

from evaluation.config import EvalConfig
from evaluation.utils import (
    build_image_annotation_map,
    load_coco_annotations,
    coco_bbox_to_xyxy,
)
from evaluation.metrics import compute_map, compute_count_accuracy
from evaluation.logger import get_logger



class Evaluator:
    def __init__(self, config: EvalConfig = EvalConfig()):
        self.config = config
        self.logger = get_logger()
        self.model = YOLO(str(self.config.MODEL_PATH))

        self.coco = load_coco_annotations(self.config.ANNOTATIONS_PATH)
        self.image_map, self.ann_map = build_image_annotation_map(self.coco)

    def run(self) -> Dict:
        self.logger.info("Starting evaluation...")

        results = []

        for image_id, image_info in self.image_map.items():
            image_path = self.config.DATA_DIR / image_info["file_name"]

            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))

            predictions = self.model.predict(
                image,
                conf=self.config.CONF_THRESHOLD,
                verbose=False
            )[0]

            preds = []
            for box, score in zip(predictions.boxes.xyxy, predictions.boxes.conf):
                preds.append({
                    "bbox": box.tolist(),
                    "score": float(score)
                })

            gts = []
            for ann in self.ann_map.get(image_id, []):
                gts.append({
                    "bbox": coco_bbox_to_xyxy(ann["bbox"])
                })

            results.append({
                "image_id": image_id,
                "preds": preds,
                "gts": gts
            })

        map_results = compute_map(results, self.config.IOU_THRESHOLDS)
        count_acc = compute_count_accuracy(results)

        summary = {
            **map_results,
            "count_accuracy": count_acc
        }

        self._save_report(summary)

        self.logger.info("Evaluation complete.")
        self.logger.info(summary)

        return summary

    def _save_report(self, metrics: Dict):
        output_path = self.config.OUTPUT_DIR / "evaluation_report.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
