"""
Fine-tune YOLOv8n on the coin detection dataset.

Loads training config from YAML, sets up experiment directory,
and runs Ultralytics training with experiment tracking.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


TRAINING_ROOT = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = TRAINING_ROOT.parent.resolve()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_experiment_dir(experiments_dir: Path, name: str | None = None) -> Path:
    """Create a timestamped experiment directory."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    exp_name = f"{timestamp}_{name}" if name else timestamp
    exp_dir = experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train(
    model_config_path: Path = TRAINING_ROOT / "configs" / "model.yaml",
    dataset_config_path: Path = TRAINING_ROOT / "configs" / "dataset.yaml",
    experiment_name: str | None = None,
    resume: str | None = None,
):
    """Run fine-tuning with given configuration."""
    # Load configs
    model_cfg = load_config(model_config_path)
    print(f"Loaded model config from {model_config_path}")

    # Create experiment directory
    experiments_dir = TRAINING_ROOT / "experiments"
    exp_dir = create_experiment_dir(experiments_dir, experiment_name)
    print(f"Experiment directory: {exp_dir}")

    # Save configs for reproducibility
    shutil.copy2(model_config_path, exp_dir / "model.yaml")
    shutil.copy2(dataset_config_path, exp_dir / "dataset.yaml")

    # Resolve dataset.yaml to absolute path (Ultralytics needs this)
    dataset_yaml_abs = dataset_config_path.resolve()

    # Extract model base and training params
    base_model = model_cfg.pop("model", "yolov8n.pt")

    # If resuming, load from checkpoint instead
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        model = YOLO(resume)
    else:
        # Check for pretrained model in artifacts or download
        local_model = PROJECT_ROOT / "artifacts" / "models" / base_model
        if local_model.exists():
            print(f"Loading pretrained model from {local_model}")
            model = YOLO(str(local_model))
        else:
            print(f"Downloading {base_model} from Ultralytics hub")
            model = YOLO(base_model)

    # Train
    results = model.train(
        data=str(dataset_yaml_abs),
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        **model_cfg,
    )

    # Save training summary
    summary = {
        "experiment": exp_dir.name,
        "base_model": base_model,
        "dataset_config": str(dataset_yaml_abs),
        "best_model": str(exp_dir / "train" / "weights" / "best.pt"),
        "last_model": str(exp_dir / "train" / "weights" / "last.pt"),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best model: {summary['best_model']}")
    print(f"  Last model: {summary['last_model']}")
    print(f"  Experiment: {exp_dir}")

    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on coin dataset")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=TRAINING_ROOT / "configs" / "model.yaml",
        help="Path to model/training config YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=TRAINING_ROOT / "configs" / "dataset.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name suffix (appended to timestamp)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    train(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        experiment_name=args.name,
        resume=args.resume,
    )
