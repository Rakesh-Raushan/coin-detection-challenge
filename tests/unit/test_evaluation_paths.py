"""
Verification test for evaluation module path configuration.
Ensures evaluation outputs align with architecture.
"""
from pathlib import Path
from evaluation.config import EvalConfig


def test_evaluation_uses_production_model():
    """Test that evaluation loads model from production models/ directory."""
    config = EvalConfig()

    assert config.MODEL_PATH.exists()
    assert "models" in str(config.MODEL_PATH)
    assert config.MODEL_PATH.name == "yolov8n-coin-finetuned.pt"
    # Should NOT be in old artifacts/models location
    assert "artifacts/models" not in str(config.MODEL_PATH)


def test_evaluation_outputs_to_training_artifacts():
    """Test that evaluation outputs to training/artifacts/evaluation."""
    config = EvalConfig()

    # Output directory should be under training/artifacts/evaluation
    assert "training" in str(config.OUTPUT_BASE_DIR)
    assert "artifacts" in str(config.OUTPUT_BASE_DIR)
    assert "evaluation" in str(config.OUTPUT_BASE_DIR)

    # Should have timestamped subdirectory
    assert config.OUTPUT_DIR.parent == config.OUTPUT_BASE_DIR


def test_evaluation_directory_structure():
    """Test that evaluation creates proper directory structure."""
    config = EvalConfig()

    # Verify the structure matches: training/artifacts/evaluation/TIMESTAMP
    parts = config.OUTPUT_DIR.parts

    # Find the indices of key parts
    training_idx = parts.index("training")

    # Verify order: .../training/artifacts/evaluation/TIMESTAMP
    assert parts[training_idx + 1] == "artifacts"
    assert parts[training_idx + 2] == "evaluation"
    # Last part should be timestamp (format: YYYY-MM-DDTHH-MM-SSZ)
    timestamp = parts[-1]
    assert "T" in timestamp
    assert "Z" in timestamp


def test_evaluation_paths_aligned():
    """Integration test: verify complete path alignment."""
    config = EvalConfig()

    # Production model in models/
    assert config.MODEL_PATH.parts[-2] == "models"

    # Evaluation outputs in training/artifacts/
    output_path_str = str(config.OUTPUT_BASE_DIR)
    assert "training/artifacts/evaluation" in output_path_str

    # Verify no legacy "artifacts/models" or "artifacts/evaluation" paths
    assert "/artifacts/models" not in str(config.MODEL_PATH)
    assert "/artifacts/evaluation" not in output_path_str or "/training/artifacts/evaluation" in output_path_str
