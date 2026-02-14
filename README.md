# Coin Detection API

> Production-ready REST API for detecting and identifying circular objects (coins) in images with geometric analysis. The system combines YOLO-based object detection with deterministic geometric post-processing to derive precise coin properties without pixel-level-segmenation training.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](docs/07_testing_and_quality.md)

---

## What This Does

Upload an image containing coins → Get bounding boxes, centroids, radii, and segmentation masks for each coin.

**Functional Features**:
- Multi-instance object detection (handles 1-14+ coins per image)
- Unique persistent identifiers for each detected coin
- Geometric properties: bbox, centroid (ellipse center), radius
- Segmentation masks via elliptical fitting (handles slanted coins)
- Visualization endpoint with transparent mask overlay

**Engineering Quality**:
- Structured JSON logging
- Health checks
- 98% test coverage with unit + integration tests

---

## Quick Start

### Docker (Recommended)

```bash
# Clone and run
git clone <repo-url>
cd coin-detection-challenge
docker-compose up --build

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Local Development

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
make dev

# Run API
make run  # Starts at http://localhost:8000
```

**System Requirements**:
- Python 3.10+
- ~200MB RAM (model + runtime)
- Typical inference latency(CPU) ~50-150ms per image (depends on image size)

---

## API Usage

### Upload & Detect

```bash
curl -X POST "http://localhost:8000/api/v1/images" \
  -F "file=@coin_image.jpg"
```

**Response**:
```json
{
  "id": "a1b2c3d4",
  "filename": "a1b2c3d4.jpg",
  "uploaded_at": "2026-02-13T10:30:45Z",
  "coins": [
    {
      "id": "a1b2c3d4_coin_001",
      "center_x": 150.5,
      "center_y": 200.3,
      "radius": 45.0,
      "is_slanted": false,
      "bbox_x": 105.5,
      "bbox_y": 155.3,
      "bbox_w": 90.0,
      "bbox_h": 90.0,
      "confidence": 0.95
    }
  ]
}
```

### Retrieve Image Details

```bash
curl "http://localhost:8000/api/v1/images/a1b2c3d4"
```

### Get Coin Details

```bash
curl "http://localhost:8000/api/v1/coins/a1b2c3d4_coin_001"
```

### Visualize Detections

```bash
curl "http://localhost:8000/api/v1/images/a1b2c3d4/render" \
  --output visualization.png
```

**Output**: Original image with transparent red masks + green bboxes + coin IDs

### Health Check

```bash
curl "http://localhost:8000/api/v1/health"
```

**Response**:
```json
{
  "status": "healthy",
  "model": {
    "available": true,
    "status": "loaded"
  }
}
```

**Full API Reference**: OpenAPI docs at `/docs` (interactive, includes examples)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/images` | Upload image, triggers detection |
| `GET` | `/api/v1/images/{id}` | Get image details with all coins |
| `GET` | `/api/v1/coins/{id}` | Get specific coin properties |
| `GET` | `/api/v1/images/{id}/render` | Visualize masks on image |
| `GET` | `/api/v1/health` | Health check (model status) |

---

## Development Commands

```bash
make help         # Show all available commands
make dev          # Install dev dependencies
make test         # Run tests (98% coverage)
make test-cov     # Run tests with coverage report
make lint         # Check code style (ruff)
make format       # Auto-format code (black)
make docker-up    # Start Docker containers
make docker-down  # Stop Docker containers
```

---

## Model Information

**Current Production Model**: `yolov8n-coin-finetuned.pt`
- Base: YOLOv8 Nano (3.2M parameters, 6.0 MB)
- Task: Single-class object detection (coin)
- Performance: mAP@0.5:0.95 0.73, Counting accuracy 84.6% (on test set)
- Inference: ~50-150ms (CPU), Varies with image size

See eval report [`training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json`](training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json)

**See**: [`models/README.md`](models/README.md) for model card and metadata

---

## Project Structure

```
coin-detection-challenge/
├── app/                    # Application code
│   ├── api/                # FastAPI routes
│   ├── services/           # Detection & geometry logic
│   ├── core/               # Config, DB, logging
│   └── middleware/         # Request logging
├── models/                 # Production models only
├── training/               # Training pipeline (separate)
├── evaluation/             # Evaluation framework
├── tests/                  # Unit + integration tests (98% coverage)
├── docs/                   # Engineering documentation →
└── Dockerfile              # Production container
```

---

## Technical Documentation

> **For implementation details, design rationale, and engineering insights**, see [`/docs`](docs/).

### Documentation Structure

📂 **Engineering Narrative** (`/docs/` — 8 documents)

1. **[Problem Understanding](docs/01_problem_understanding.md)**: Dataset analysis, constraints, assumptions, trade-offs
2. **[Architecture Overview](docs/02_architecture_overview.md)**: System design, components, data flow, technology stack
3. **[Detection and Geometry](docs/03_detection_and_geometry.md)**: YOLO pipeline, ellipse fitting, mask generation, edge cases
4. **[Training Pipeline](docs/04_training_pipeline.md)**: Dataset management, hyperparameters, experiment tracking, model export
5. **[Evaluation Framework](docs/05_evaluation_framework.md)**: Metrics rationale, proxy validation, failure bucketing, schema design
6. **[MLOps and Deployment](docs/06_mlops_and_deployment.md)**: Containerization, observability, configuration, production path
7. **[Testing and Quality](docs/07_testing_and_quality.md)**: 98% coverage, unit vs. integration, fixtures, CI/CD
8. **[Design Decisions (ADRs)](docs/08_design_decisions.md)**: Key architectural choices with context and trade-offs

📄 **Domain-Specific READMEs**

- **[`training/README.md`](training/README.md)**: Training workflow, fine-tuning, data split strategy
- **[`models/README.md`](models/README.md)**: Model card, metadata, usage, versioning

📊 **Evaluation Deep Dive**

- **[`evaluation/eval_strategy.md`](evaluation/eval_strategy.md)**: Metric selection philosophy, schema design, proxy metrics rationale

---

## Configuration

**Environment Variables** (optional overrides):

```bash
# Model configuration
MODEL_PATH=models/yolov8n-coin-finetuned.pt
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45

# Storage
UPLOAD_DIR=data/uploads
DATABASE_URL=sqlite:///data/database.db  # Or postgresql://...

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

**See**: [`app/core/config.py`](app/core/config.py) for all settings

---

## License

MIT

---

## Acknowledgments

Built as part of technical assessment challenge 1

For detailed design rationale and trade-off analysis, see [Design Decisions (ADRs)](docs/08_design_decisions.md).
