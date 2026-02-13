import hashlib
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List


@dataclass
class ModelMetadata:
    name: str
    path: str
    framework: str
    input_size: List[int]
    git_commit: Optional[str] = None
    file_modified: Optional[str] = None
    file_size_mb: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class DatasetMetadata:
    name: str
    data_dir: str
    annotations_path: str
    num_images: int
    num_annotations: int
    annotations_hash: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def get_git_commit() -> Optional[str]:
    """Retrieve current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return None


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
    """Compute hash of a file."""
    if not file_path.exists():
        return None
    try:
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return f"{algorithm}:{hasher.hexdigest()[:16]}"
    except Exception:
        return None


def extract_model_metadata(
    model_path: Path,
    framework: str = "ultralytics",
    input_size: List[int] = [640, 640]
) -> ModelMetadata:
    """Extract metadata from model file."""
    path = Path(model_path)
    
    file_modified = None
    file_size_mb = None
    if path.exists():
        file_modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    
    return ModelMetadata(
        name=path.stem,
        path=str(path.absolute()),
        framework=framework,
        input_size=input_size,
        git_commit=get_git_commit(),
        file_modified=file_modified,
        file_size_mb=file_size_mb
    )


def extract_dataset_metadata(
    data_dir: Path,
    annotations_path: Path,
    coco_dict: dict
) -> DatasetMetadata:
    """Extract metadata from evaluation dataset."""
    num_images = len(coco_dict.get("images", []))
    num_annotations = len(coco_dict.get("annotations", []))
    
    return DatasetMetadata(
        name=data_dir.name,
        data_dir=str(data_dir.absolute()),
        annotations_path=str(annotations_path.absolute()),
        num_images=num_images,
        num_annotations=num_annotations,
        annotations_hash=compute_file_hash(annotations_path)
    )