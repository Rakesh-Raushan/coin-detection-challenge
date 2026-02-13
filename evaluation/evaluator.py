# app/evaluation/evaluator.py
"""
Model evaluation orchestrator.

Coordinates the full evaluation pipeline:
1. Load model and annotations
2. Run inference on all images
3. Compute all metrics
4. Generate structured artifacts
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
from ultralytics import YOLO

from evaluation.config import EvalConfig
from evaluation.utils import (
    build_image_annotation_map,
    load_coco_annotations,
    coco_bbox_to_xyxy,
)
from evaluation.metrics import compute_all_metrics, AllMetrics
from evaluation.metadata import extract_model_metadata, extract_dataset_metadata
from evaluation.artifacts import generate_artifacts, ArtifactBuilder
from evaluation.logger import get_logger


class Evaluator:
    """
    Main evaluation orchestrator.
    
    Runs model inference, computes all metrics, and generates structured artifacts:
    - evaluation_summary.json
    - evaluation_provenance.json
    - per_image_analysis.jsonl
    - failure_examples.json
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration. If None, creates new instance
                    with default settings from environment variables.
        """
        self.config = config or EvalConfig()
        self.logger = get_logger()
        self.model = YOLO(str(self.config.MODEL_PATH))

        # Load annotations
        self.coco = load_coco_annotations(self.config.ANNOTATIONS_PATH)
        self.image_map, self.ann_map = build_image_annotation_map(self.coco)

        # Extract metadata for provenance
        self.model_metadata = extract_model_metadata(
            self.config.MODEL_PATH,
            self.config.MODEL_FRAMEWORK,
            self.config.MODEL_INPUT_SIZE
        )
        self.dataset_metadata = extract_dataset_metadata(
            self.config.DATA_DIR,
            self.config.ANNOTATIONS_PATH,
            self.coco
        )

        self._log(f"Initialized evaluator")
        self._log(f"  Run ID: {self.config.run_id}")
        self._log(f"  Output: {self.config.OUTPUT_DIR}")

    def _log(self, message: str, level: str = "info"):
        """Log message if verbose mode is enabled."""
        if self.config.VERBOSE:
            getattr(self.logger, level)(message)

    def run(self) -> Dict[str, Path]:
        """
        Execute full evaluation pipeline.
        
        Returns:
            Dict mapping artifact names to their file paths.
        """
        self._log("=" * 60)
        self._log("Starting evaluation run")
        self._log(f"  Model: {self.model_metadata.name}")
        self._log(f"  Dataset: {self.dataset_metadata.name} ({self.dataset_metadata.num_images} images)")
        self._log("=" * 60)

        # Phase 1: Inference
        results = self._run_inference()

        # Phase 2: Compute all metrics
        metrics = self._compute_metrics(results)

        # Phase 3: Generate artifacts
        artifact_paths = self._generate_artifacts(metrics)

        # Phase 4: Log summary
        self._log_summary(metrics)

        return artifact_paths

    def _run_inference(self) -> List[Dict]:
        """
        Run model inference on all images.
        
        Returns:
            List of dicts with image_id, preds, and gts.
        """
        self._log("Phase 1: Running inference...")
        results = []
        processed = 0
        skipped = 0

        for image_id, image_info in self.image_map.items():
            image_path = self.config.DATA_DIR / image_info["file_name"]

            if not image_path.exists():
                self._log(f"  Skipping missing image: {image_info['file_name']}", level="warning")
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                self._log(f"  Failed to read image: {image_info['file_name']}", level="warning")
                skipped += 1
                continue

            # Run inference
            predictions = self.model.predict(
                image,
                conf=self.config.CONF_THRESHOLD,
                verbose=False
            )[0]

            # Extract predictions
            preds = []
            for box, score in zip(predictions.boxes.xyxy, predictions.boxes.conf):
                bbox = box.tolist()
                preds.append({
                    "bbox": bbox,
                    "score": float(score),
                    # Compute centroid and radius from bbox for geometric metrics
                    "centroid": (
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2
                    ),
                    "radius": min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
                })

            # Extract ground truth
            gts = []
            for ann in self.ann_map.get(image_id, []):
                gts.append({
                    "bbox": coco_bbox_to_xyxy(ann["bbox"])
                })

            results.append({
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "preds": preds,
                "gts": gts
            })

            processed += 1

        self._log(f"  Processed: {processed} images")
        if skipped > 0:
            self._log(f"  Skipped: {skipped} images", level="warning")

        return results

    def _compute_metrics(self, results: List[Dict]) -> AllMetrics:
        """
        Compute all metrics from inference results.
        
        Args:
            results: List of dicts with image_id, preds, and gts.
        
        Returns:
            AllMetrics containing all computed metrics.
        """
        self._log("Phase 2: Computing metrics...")
        
        metrics = compute_all_metrics(results, self.config.IOU_THRESHOLDS)
        
        self._log(f"  mAP@0.5:0.95: {metrics.map_results.map_50_95:.4f}")
        self._log(f"  mAP@0.5: {metrics.map_results.map_50:.4f}")
        self._log(f"  Count Accuracy: {metrics.counting_results.count_accuracy:.4f}")
        
        return metrics

    def _generate_artifacts(self, metrics: AllMetrics) -> Dict[str, Path]:
        """
        Generate and save all evaluation artifacts.
        
        Args:
            metrics: All computed metrics.
        
        Returns:
            Dict mapping artifact names to their file paths.
        """
        self._log("Phase 3: Generating artifacts...")
        
        artifact_paths = generate_artifacts(
            config=self.config,
            model_metadata=self.model_metadata,
            dataset_metadata=self.dataset_metadata,
            metrics=metrics
        )
        
        for name, path in artifact_paths.items():
            self._log(f"  Generated: {path.name}")
        
        return artifact_paths

    def _log_summary(self, metrics: AllMetrics):
        """Log final evaluation summary."""
        self._log("=" * 60)
        self._log("Evaluation Complete")
        self._log("-" * 60)
        self._log("METRICS:")
        self._log(f"  mAP@0.5:0.95: {metrics.map_results.map_50_95:.4f}")
        self._log(f"  mAP@0.5: {metrics.map_results.map_50:.4f}")
        self._log(f"  Count Accuracy: {metrics.counting_results.count_accuracy:.4f}")
        self._log(f"  Mean Absolute Count Error: {metrics.counting_results.mean_absolute_count_error:.2f}")
        self._log("-" * 60)
        self._log("FAILURE ANALYSIS:")
        self._log(f"  Severe: {metrics.failure_analysis.severe_failure_pct:.1f}%")
        self._log(f"  Moderate: {metrics.failure_analysis.moderate_failure_pct:.1f}%")
        self._log(f"  Good: {metrics.failure_analysis.good_pct:.1f}%")
        self._log(f"  Excellent: {metrics.failure_analysis.excellent_pct:.1f}%")
        self._log("-" * 60)
        self._log("SHAPE DISTRIBUTION:")
        self._log(f"  Near-circular: {metrics.shape_distribution.near_circular_pct:.1f}%")
        self._log(f"  Moderately slanted: {metrics.shape_distribution.moderately_slanted_pct:.1f}%")
        self._log(f"  Highly slanted: {metrics.shape_distribution.highly_slanted_pct:.1f}%")
        self._log("=" * 60)
        self._log(f"Artifacts saved to: {self.config.OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for running evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file (overrides EVAL_MODEL_PATH env var)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory (overrides EVAL_DATA_DIR env var)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create config with optional overrides
    config = EvalConfig()
    if args.model_path:
        config.MODEL_PATH = Path(args.model_path)
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    config.VERBOSE = args.verbose

    # Run evaluation
    evaluator = Evaluator(config)
    artifact_paths = evaluator.run()

    print(f"\nEvaluation complete. Artifacts saved to: {config.OUTPUT_DIR}")
    for name, path in artifact_paths.items():
        print(f"  - {name}: {path.name}")


if __name__ == "__main__":
    main()
