"""
Export the best checkpoint from a training experiment to the production models folder.

Copies the best.pt from an experiment run into models/ at project root with a
descriptive name, making it available for deployment and production use.

Idea is to maintain clean separation between training artifacts (training/artifacts/)
and production models (models/).
"""

import argparse
import shutil
from pathlib import Path


TRAINING_ROOT = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = TRAINING_ROOT.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"  # Production models directory


def export_model(
    experiment_dir: Path,
    model_name: str = "yolov8n-coin-finetuned",
    format: str = "pt",
):
    """
    Export best checkpoint from experiment to production models/ directory.

    Args:
        experiment_dir: Path to the experiment directory.
        model_name: Name for the exported model (without extension).
        format: Export format — 'pt' copies as-is, 'onnx'/'torchscript' triggers conversion.
    """
    best_pt = experiment_dir / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"No best.pt found at {best_pt}. "
            f"Check that training completed and the experiment path is correct."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if format == "pt":
        dst = MODELS_DIR / f"{model_name}.pt"
        shutil.copy2(best_pt, dst)
        print(f"✓ Exported production model to {dst}")
        print(f"  Model ready for deployment!")
    else:
        # Use Ultralytics export for other formats
        from ultralytics import YOLO

        model = YOLO(str(best_pt))
        model.export(format=format)
        # Find the exported file
        exported = best_pt.parent / f"best.{format}"
        if exported.exists():
            dst = MODELS_DIR / f"{model_name}.{format}"
            shutil.move(str(exported), dst)
            print(f"Exported production model to {dst}")
        else:
            print(f"Export completed. Check {best_pt.parent} for output files.")

    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export best model from experiment")
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        help="Path to experiment directory (e.g., experiments/20260213T...)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n-coin-finetuned",
        help="Name for exported model (default: yolov8n-coin-finetuned)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "onnx", "torchscript"],
        help="Export format (default: pt)",
    )
    args = parser.parse_args()

    export_model(
        experiment_dir=args.experiment_dir,
        model_name=args.name,
        format=args.format,
    )
