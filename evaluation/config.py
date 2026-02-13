"""
Evaluation configuration module.

Manages all environment-driven settings for the evaluation pipeline.
Supports timestamped run directories for historical tracking.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List


# Project root directory (evaluation/ is one level below root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class EvalConfig:
    """
    Configuration for evaluation runs.
    
    All settings can be overridden via environment variables prefixed with EVAL_.
    Each run creates a timestamped subdirectory under OUTPUT_DIR for artifact isolation.
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # Dataset Configuration
    # ─────────────────────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(
        os.getenv(
            "EVAL_DATA_DIR",
            "/Users/rraushan/projects/G42/secondroundtechnicalassessmentseniorengineermach/coin-dataset/"
        )
    )
    ANNOTATIONS_PATH: Path = Path(
        os.getenv(
            "EVAL_ANNOTATIONS_PATH",
            "/Users/rraushan/projects/G42/secondroundtechnicalassessmentseniorengineermach/annotations/_annotations.coco.json"
        )
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Model Configuration
    # ─────────────────────────────────────────────────────────────────────────────
    MODEL_PATH: Path = Path(
        os.getenv("EVAL_MODEL_PATH", str(PROJECT_ROOT / "artifacts/models/yolov8n.pt"))
    )
    MODEL_FRAMEWORK: str = os.getenv("EVAL_MODEL_FRAMEWORK", "ultralytics")
    MODEL_INPUT_SIZE: List[int] = [640, 640]

    # ─────────────────────────────────────────────────────────────────────────────
    # Evaluation Settings
    # ─────────────────────────────────────────────────────────────────────────────
    IOU_THRESHOLDS: List[float] = [x / 100 for x in range(50, 100, 5)]  # 0.50:0.95
    IOU_POLICY: str = "COCO_0.5_0.95"
    CONF_THRESHOLD: float = float(os.getenv("EVAL_CONF_THRESHOLD", "0.25"))

    # ─────────────────────────────────────────────────────────────────────────────
    # Output & Logging
    # ─────────────────────────────────────────────────────────────────────────────
    OUTPUT_BASE_DIR: Path = Path(
        os.getenv("EVAL_OUTPUT_DIR", str(PROJECT_ROOT / "artifacts/evaluation"))
    )
    VERBOSE: bool = os.getenv("EVAL_VERBOSE", "true").lower() in ("true", "1", "yes")

    # Failure example selection
    FAILURE_EXAMPLE_TOP_K: int = int(os.getenv("EVAL_FAILURE_TOP_K", "5"))

    def __init__(self):
        """Initialize config with unique run_id and timestamped output directory."""
        self.timestamp = datetime.utcnow()
        self.run_id = self._generate_run_id()
        self.OUTPUT_DIR = self._create_run_directory()

    def _generate_run_id(self) -> str:
        """Generate unique run identifier: timestamp + short UUID."""
        ts = self.timestamp.strftime("%Y%m%dT%H%M%SZ")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{ts}_{short_uuid}"

    def _create_run_directory(self) -> Path:
        """Create timestamped subdirectory for this evaluation run."""
        iso_timestamp = self.timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = self.OUTPUT_BASE_DIR / iso_timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @property
    def model_id(self) -> str:
        """Return model identifier from path."""
        return self.MODEL_PATH.stem

    def validate_model_path(self) -> None:
        """
        Validate that the model file exists.
        
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.MODEL_PATH}\n"
                f"The evaluation module requires an existing model file.\n"
                f"Please ensure the model is placed at the specified path or set EVAL_MODEL_PATH environment variable."
            )

    def validate_annotations_path(self) -> None:
        """
        Validate that the annotations file exists.
        
        Raises:
            FileNotFoundError: If the annotations file does not exist.
        """
        if not self.ANNOTATIONS_PATH.exists():
            raise FileNotFoundError(
                f"Annotations not found at: {self.ANNOTATIONS_PATH}\n"
                f"Please ensure the annotations file exists or set EVAL_ANNOTATIONS_PATH environment variable."
            )

    def validate(self) -> None:
        """
        Validate all required paths exist before evaluation.
        
        Raises:
            FileNotFoundError: If any required file is missing.
        """
        self.validate_model_path()
        self.validate_annotations_path()
