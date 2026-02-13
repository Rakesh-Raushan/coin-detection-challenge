# app/evaluation/artifacts.py
"""
Artifact generation and management module.

Handles the creation and serialization of all evaluation output artifacts:
- evaluation_summary.json: Executive summary for decision making
- evaluation_provenance.json: Audit trail and reproducibility metadata
- per_image_analysis.jsonl: Debug analysis (line-delimited JSON)
- failure_examples.json: Top-K representative failure/success examples
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from evaluation.config import EvalConfig
from evaluation.metadata import ModelMetadata, DatasetMetadata
from evaluation.metrics import AllMetrics


# ─────────────────────────────────────────────────────────────────────────────────
# Schema Definitions
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationSummary:
    """
    Executive summary for decision making.
    Schema: evaluation_summary.json
    
    Constraint: No per-image data or raw paths.
    """
    run_id: str
    model_id: str
    metrics: Dict[str, float]
    failure_analysis: Dict[str, float]
    shape_distribution: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "metrics": self.metrics,
            "failure_analysis": self.failure_analysis,
            "shape_distribution": self.shape_distribution
        }


@dataclass
class EvaluationProvenance:
    """
    Auditability and reproducibility metadata.
    Schema: evaluation_provenance.json
    """
    timestamp: str
    git_commit: Optional[str]
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    config: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "model": self.model,
            "dataset": self.dataset,
            "config": self.config
        }


@dataclass
class FailureExamples:
    """
    Top-K representative images per failure bucket.
    Schema: failure_examples.json
    """
    severe_failure: List[str]
    moderate_failure: List[str]
    good: List[str]
    excellent: List[str]

    def to_dict(self) -> Dict:
        return {
            "severe_failure": self.severe_failure,
            "moderate_failure": self.moderate_failure,
            "good": self.good,
            "excellent": self.excellent
        }


# ─────────────────────────────────────────────────────────────────────────────────
# Artifact Builder
# ─────────────────────────────────────────────────────────────────────────────────


class ArtifactBuilder:
    """
    Builds and serializes all evaluation artifacts.
    
    Generates exactly four files:
    1. evaluation_summary.json - Executive summary
    2. evaluation_provenance.json - Audit trail
    3. per_image_analysis.jsonl - Debug analysis
    4. failure_examples.json - Visualization curation
    """

    def __init__(
        self,
        config: EvalConfig,
        model_metadata: ModelMetadata,
        dataset_metadata: DatasetMetadata
    ):
        self.config = config
        self.model_metadata = model_metadata
        self.dataset_metadata = dataset_metadata
        self.output_dir = config.OUTPUT_DIR

    def build_summary(self, metrics: AllMetrics) -> EvaluationSummary:
        """Build executive summary from computed metrics."""
        return EvaluationSummary(
            run_id=self.config.run_id,
            model_id=self.config.model_id,
            metrics={
                "mAP@0.5:0.95": round(metrics.map_results.map_50_95, 4),
                "mAP@0.5": round(metrics.map_results.map_50, 4),
                "count_accuracy": round(metrics.counting_results.count_accuracy, 4),
                "mean_absolute_count_error": round(metrics.counting_results.mean_absolute_count_error, 4)
            },
            failure_analysis={
                "severe_failure_pct": round(metrics.failure_analysis.severe_failure_pct, 2),
                "moderate_failure_pct": round(metrics.failure_analysis.moderate_failure_pct, 2),
                "good_pct": round(metrics.failure_analysis.good_pct, 2),
                "excellent_pct": round(metrics.failure_analysis.excellent_pct, 2)
            },
            shape_distribution={
                "near_circular_pct": round(metrics.shape_distribution.near_circular_pct, 2),
                "moderately_slanted_pct": round(metrics.shape_distribution.moderately_slanted_pct, 2),
                "highly_slanted_pct": round(metrics.shape_distribution.highly_slanted_pct, 2)
            }
        )

    def build_provenance(self) -> EvaluationProvenance:
        """Build provenance report for audit trail."""
        return EvaluationProvenance(
            timestamp=self.config.timestamp.isoformat() + "Z",
            git_commit=self.model_metadata.git_commit,
            model={
                "name": self.model_metadata.name,
                "file_size_mb": self.model_metadata.file_size_mb,
                "input_size": self.model_metadata.input_size
            },
            dataset={
                "name": self.dataset_metadata.name,
                "num_images": self.dataset_metadata.num_images,
                "annotations_hash": self.dataset_metadata.annotations_hash
            },
            config={
                "iou_policy": self.config.IOU_POLICY,
                "confidence_threshold": self.config.CONF_THRESHOLD
            }
        )

    def build_failure_examples(
        self,
        failure_buckets: Dict[str, List],
        top_k: Optional[int] = None
    ) -> FailureExamples:
        """
        Select top-K representative images per failure bucket.
        
        Args:
            failure_buckets: Dict mapping bucket name to list of image_ids
            top_k: Number of examples per bucket (default from config)
        """
        k = top_k or self.config.FAILURE_EXAMPLE_TOP_K

        return FailureExamples(
            severe_failure=[str(x) for x in failure_buckets.get("severe", [])[:k]],
            moderate_failure=[str(x) for x in failure_buckets.get("moderate", [])[:k]],
            good=[str(x) for x in failure_buckets.get("good", [])[:k]],
            excellent=[str(x) for x in failure_buckets.get("excellent", [])[:k]]
        )

    def save_all(self, metrics: AllMetrics) -> Dict[str, Path]:
        """
        Generate and save all four artifact files.
        
        Returns:
            Dict mapping artifact name to saved file path.
        """
        saved_paths = {}

        # 1. Evaluation Summary
        summary = self.build_summary(metrics)
        summary_path = self.output_dir / "evaluation_summary.json"
        self._write_json(summary_path, summary.to_dict())
        saved_paths["evaluation_summary"] = summary_path

        # 2. Evaluation Provenance
        provenance = self.build_provenance()
        provenance_path = self.output_dir / "evaluation_provenance.json"
        self._write_json(provenance_path, provenance.to_dict())
        saved_paths["evaluation_provenance"] = provenance_path

        # 3. Per-Image Analysis (JSONL)
        per_image_path = self.output_dir / "per_image_analysis.jsonl"
        self._write_jsonl(per_image_path, metrics.per_image_analyses)
        saved_paths["per_image_analysis"] = per_image_path

        # 4. Failure Examples
        failure_examples = self.build_failure_examples(metrics.failure_buckets)
        failure_path = self.output_dir / "failure_examples.json"
        self._write_json(failure_path, failure_examples.to_dict())
        saved_paths["failure_examples"] = failure_path

        return saved_paths

    def _write_json(self, path: Path, data: Dict) -> None:
        """Write dictionary to JSON file with consistent formatting."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, path: Path, records: List[Dict]) -> None:
        """Write list of dictionaries to JSONL file (one JSON object per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────────


def generate_artifacts(
    config: EvalConfig,
    model_metadata: ModelMetadata,
    dataset_metadata: DatasetMetadata,
    metrics: AllMetrics
) -> Dict[str, Path]:
    """
    One-shot function to generate all evaluation artifacts.
    
    Args:
        config: Evaluation configuration
        model_metadata: Model metadata
        dataset_metadata: Dataset metadata
        metrics: All computed metrics
    
    Returns:
        Dict mapping artifact name to saved file path.
    """
    builder = ArtifactBuilder(config, model_metadata, dataset_metadata)
    return builder.save_all(metrics)
