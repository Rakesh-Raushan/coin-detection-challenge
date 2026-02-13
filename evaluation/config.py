# app/evaluation/config.py

import os
from pathlib import Path


class EvalConfig:
    # Dataset
    DATA_DIR: Path = Path(os.getenv("EVAL_DATA_DIR", "/Users/rraushan/projects/G42/secondroundtechnicalassessmentseniorengineermach/coin-dataset/"))
    ANNOTATIONS_PATH: Path = Path(
        os.getenv("EVAL_ANNOTATIONS_PATH", "/Users/rraushan/projects/G42/secondroundtechnicalassessmentseniorengineermach/annotations/_annotations.coco.json")
    )

    # Model
    MODEL_PATH: Path = Path(
        os.getenv("EVAL_MODEL_PATH", "artifacts/models/yolov8n.pt")
    )

    # Evaluation settings
    IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]  # 0.50:0.95
    CONF_THRESHOLD: float = float(os.getenv("EVAL_CONF_THRESHOLD", 0.25))

    # Output
    OUTPUT_DIR: Path = Path(os.getenv("EVAL_OUTPUT_DIR", "artifacts/evaluation"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
