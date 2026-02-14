# Production Models

This directory contains **production-ready models only** - models that are deployed with the application. This can be replaced with model registry logic in real scenarios.

## Current Model

**File**: `yolov8n-coin-finetuned.pt`
**Version**: 1.0
**Framework**: YOLOv8 (Ultralytics)
**Size**: 6.0 MB
**Task**: Coin detection in images

### Model Metadata

- **Training Date**: 2024-02-13
- **Base Model**: YOLOv8n (nano)
- **Fine-tuning Dataset**: 191 images, 521 coin annotations
- **Input Size**: 640x640
- **Classes**: 1 (coin)

### Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5:0.95 | See evaluation artifacts |
| mAP@0.5 | See evaluation artifacts |
| Inference Time | ~50-150ms (CPU) |

### Usage

This model is loaded by the application via `app/core/config.py`:

```python
MODEL_PATH = BASE_DIR / "models" / "yolov8n-coin-finetuned.pt"
```

### Model Registry

For production deployments, consider using a proper model registry (MLflow, W&B, etc.) and fetching models at runtime rather than bundling in Docker images.

### Updating Models

To deploy a new model:

1. Train and export model using `training/scripts/export_model.py`
2. Copy new model to this directory
3. Update this README with new version and metrics
4. Run tests: `pytest tests/`
5. Rebuild Docker image

### Important Notes

- This directory is copied into Docker images during build
- Model files should be versioned externally (S3, model registry, etc.)
