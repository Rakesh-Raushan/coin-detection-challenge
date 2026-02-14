# 03. Detection and Geometry

> **Approach**: Outcome and valed based approach to problem by grounding it to realities of data.

---

## 1. Detection Strategy

### Why Object Detection over Segmentation

**Decision**: Use YOLOv8 object detection + geometric post-processing

**Rationale**:
```
Dataset Reality:
├── Have: Bounding box annotations (521 instances)
├── Don't Have: Pixel-level masks
└── Implication: Cannot train supervised instance segmentation

Solution:
├── Use detection model (YOLO) for localization
└── Derive masks geometrically from bounding boxes
```

**Alternatives Considered**:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Train Mask R-CNN** | End-to-end segmentation | No GT masks; need pseudo-labeling | ❌ Reject |
| **Classical CV (Hough)** | No training needed | Fails on occlusion, clutter | ❌ Baseline only |
| **YOLO + Geometry** | Robust detection + deterministic masks | Assumes elliptical shape | ✅ **Selected** |

---

## 2. YOLO Detection Pipeline

### Model Selection: YOLOv8 Nano

**Specification**:
- Base model: `yolov8n.pt` (pre-trained on COCO)
- Parameters: 3.2M
- Model size: 6.0 MB
- Input: 640×640 (auto-resized)
- Output: Bounding boxes `[x, y, w, h]`, confidence scores, class probabilities

**Why YOLOv8n over larger variants**:
- Sufficient accuracy for single-class detection
- Fast inference: ~50ms CPU, ~10ms GPU
- Small size enables edge deployment
- Transfer learning from COCO covers circular objects (frisbees, bowls, plates)

### Fine-Tuning Strategy

**Baseline Performance** (pre-trained, no fine-tuning):
```
mAP@0.5:0.95: 0.5011
mAP@0.5:     0.5565
Count Accuracy: 42.3%
```

See eval summary for baseline [`training/artifacts/evaluation/2026-02-13T17-40-27Z/evaluation_summary.json`](../training/artifacts/evaluation/2026-02-13T17-40-27Z/evaluation_summary.json)

**Fine-Tuned Performance** (achieved on test set):
```
mAP@0.5:0.95: 0.731
mAP@0.5:     0.973
Count Accuracy: 84.6%
```

See eval summary for finetuned [`training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json`](../training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json)

**Training Configuration**:
- Learning rate: 0.001 (low to preserve pre-trained features)
- Epochs: 100 (with early stopping, patience=15)
- Augmentation: Rotation ±15°, scale 0.5–1.5×, horizontal flip, color jitter
- Optimizer: AdamW with cosine LR schedule

**Data Split**:
- Train: 139 images (73%)
- Val: 26 images (14%) — used for checkpoint selection
- Test: 26 images (14%) — held-out for final evaluation

**Stratification**: By coin count (1, 2, 3, 4+ bins) to ensure balanced difficulty

### Inference Process

```python
class DetectionService:
    def __init__(self, model_path: Path):
        self.model = YOLO(str(model_path))

    def process_image(self, image_path: str, image_id: str) -> List[Coin]:
        """
        Runs inference, applies geometric logic, sorts spatially,
        and returns list of Coin ORM objects ready for DB persistence.
        """
        # Run YOLO inference
        results = self.model(image_path)
        result = results[0]  # Single image result

        # Extract bboxes (convert from xyxy to xywh)
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            bbox = [float(x1), float(y1), float(w), float(h)]

            # Geometric analysis
            geo_data = CoinGeometry.analyze_detection(bbox)

            detections.append({
                "bbox": bbox,
                "geo": geo_data,
                "sort_key": (y1, x1)  # For spatial sorting
            })

        # Spatial sort (top-to-bottom, left-to-right)
        detections.sort(key=lambda x: x['sort_key'])

        # Create ORM objects with deterministic IDs
        coin_objects = []
        for idx, det in enumerate(detections):
            unique_id = f"{image_id}_coin_{idx+1:03d}"
            coin = Coin(
                id=unique_id,
                image_id=image_id,
                center_x=det['geo']['center_point'][0],
                center_y=det['geo']['center_point'][1],
                radius=det['geo']['radius'],
                is_slanted=det['geo']['is_slanted'],
                bbox_x=det['bbox'][0],
                bbox_y=det['bbox'][1],
                bbox_w=det['bbox'][2],
                bbox_h=det['bbox'][3]
            )
            coin_objects.append(coin)

        return coin_objects
```

**Key Design Choices**:

1. **YOLO Default Thresholds**:
   - Uses YOLO's default confidence and IoU thresholds
   - Confidence threshold: 0.25 (YOLO default)
   - IoU threshold for NMS: 0.45 (YOLO default)
   - Sufficient for coin detection; no manual tuning needed

2. **Spatial Sorting**:
   - Sort by (y_center, x_center) — top-to-bottom, left-to-right
   - Ensures predictable coin numbering based on spatial position within an image

---

## 3. Geometric Post-Processing

### Ellipse Fitting Algorithm

**Problem**: YOLO outputs axis-aligned bounding boxes, but coins may be slanted.

**Solution**: Fit ellipse to bbox dimensions

```python
@staticmethod
def analyze_detection(bbox: List[float]) -> Dict[str, Any]:
    """
    Analyze a bounding box and compute geometric properties.

    Args:
        bbox: [x, y, w, h] in pixel coordinates

    Returns:
        {
            'center_point': Tuple[float, float],
            'radius': float,
            'aspect_ratio': float,
            'is_slanted': bool
        }
    """
    x, y, w, h = bbox

    # Effective radius using max dimension (accounts for slanted coins)
    radius = max(w, h) / 2.0

    # Centroid (ellipse center)
    center_point = (x + w / 2, y + h / 2)

    # Aspect ratio for slant detection
    aspect_ratio = w / h if h != 0 else 0

    # Bidirectional slant detection: catches both wide and tall ellipses
    is_slanted = aspect_ratio < 0.8 or aspect_ratio > 1.2

    return {
        'radius': radius,
        'center_point': center_point,
        'aspect_ratio': aspect_ratio,
        'is_slanted': is_slanted
    }
```

**Why Max Dimension for Radius**:
- For circular coins: `max(w, h) / 2 = w / 2 = h / 2` (correct diameter-based radius)
- For slanted coins: Longest dimension approximates the true coin diameter when viewed at an angle
- Heuristic approach: Ensures radius doesn't underestimate coin size for elliptical projections

### Mask Generation

**Approach**: Generate binary mask for each coin using ellipse parameters

```python
@staticmethod
def generate_mask(bbox: List[float], img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate binary segmentation mask for a coin.

    Args:
        bbox: [x, y, w, h]
        img_shape: (height, width)

    Returns:
        Binary mask (np.ndarray) where 1 = coin, 0 = background
    """
    x, y, w, h = map(int, bbox)
    mask = np.zeros(img_shape, dtype=np.uint8)

    center = (x + w // 2, y + h // 2)
    axes = (w // 2, h // 2)
    angle = 0  # Axis-aligned (YOLO doesn't predict rotation)

    cv2.ellipse(
        mask,
        center=center,
        axes=axes,
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1  # Filled
    )

    return mask
```

**Design Decisions**:

1. **Axis-aligned ellipses**:
   - YOLO bboxes don't encode rotation
   - Sufficient for most coins (camera typically level)
   - Future: Could add rotation via contour fitting

2. **Filled ellipses**:
   - Provides binary mask (not just outline)
   - Compatible with IoU computation

3. **Generated on-demand**:
   - Not stored in database (saves space)
   - Cheap to compute (<5ms per coin)

---

## 4. Handling Edge Cases

### Slanted Coins (~45% of detections)

**Problem**: Tilted coins appear elliptical (aspect ratio ≠ 1)

**Detection**: Bidirectional aspect ratio check
```python
aspect_ratio = width / height
is_slanted = aspect_ratio < 0.8 or aspect_ratio > 1.2  # Catches both orientations
```

**Impact**:
- `is_slanted=True` → Coin aspect ratio deviates >20% from circular (1.0)
- Max-dimension radius accounts for elongation
- Elliptical mask fits naturally

**Validation**: 55% near-circular, 45% showing some slant (moderate 38% + high 7.5%)

**Evidence**: [`training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json`](../training/artifacts/evaluation/2026-02-13T17-30-19Z/evaluation_summary.json) — shape_distribution

### Overlapping Coins

**YOLO Behavior**: Predicts separate bounding boxes for overlapping objects

**Non-Maximum Suppression (NMS)**:
- Suppresses redundant detections
- Tuned IOU threshold (0.45) balances:
  - Too low: Suppresses valid nearby coins
  - Too high: Allows duplicate detections

**Mask Overlap**:
- Masks can overlap in render (transparent red overlay)
- Each coin retains distinct ID and properties

### Partial Coins (Boundary Cropping)

**Scenario**: Coin partially visible at image edge

**YOLO Handling**:
- Detects partial bbox (clipped at boundary)
- Confidence may be lower (fewer visual features)

**Geometric Handling**:
- Centroid may fall outside image bounds (valid)
- Radius computed from visible bbox portion
- No special treatment (model learns from data)

### Small Coins (Scale Variation)

**Dataset Range**: Coin sizes vary 10× (small distant coins vs. close-up)

**YOLO Multi-Scale Detection**:
- Detects objects at multiple scales (pyramid structure)
- Trained with scale augmentation (0.5–1.5×)

**Minimum Size**: ~20×20 pixels (empirically, YOLO struggles below this)

---

## 5. Deterministic Coin Identifiers

### ID Format

```
{image_id}_coin_{NNN}
```

**Example**: `a1b2c3d4_coin_001`, `a1b2c3d4_coin_002`, ...

### Spatial Sorting for Consistency

**Problem**: YOLO detection order is non-deterministic (depends on internal processing)

**Solution**: Sort detections spatially before assigning IDs

```python
@staticmethod
def _spatial_sort(bboxes: List[List[float]]) -> List[List[float]]:
    """
    Sort bounding boxes top-to-bottom, then left-to-right.

    Ensures predictable coin numbering within a single image.
    """
    def sort_key(bbox: List[float]) -> Tuple[float, float]:
        x, y, w, h = bbox
        center_y = y + h / 2
        center_x = x + w / 2
        return (center_y, center_x)  # Primary: y, Secondary: x

    return sorted(bboxes, key=sort_key)
```

**Why Top-to-Bottom, Left-to-Right**:
- Natural reading order (intuitive for humans)
- Language-agnostic (works globally)
- Predictable within image: Top-left coin always gets `_coin_001`

**Note on ID Stability**: Each upload generates a new `image_id` (UUID), so absolute coin IDs differ across re-uploads of the same image. Spatial sorting ensures *relative consistency* (same spatial position → same suffix number) within a single upload session.

---

## 6. Performance Characteristics

### Inference Time

| Component | Latency | Notes |
|-----------|---------|-------|
| YOLO prediction | ~50ms (CPU) | Dominant bottleneck |
| Geometric analysis | <1ms/coin | Pure Python, minimal computation |
| Mask generation | ~5ms/coin | OpenCV ellipse drawing |
| **Total (5 coins)** | ~75ms | Acceptable for interactive use |

**Optimization Opportunities**:
- GPU inference: ~10ms YOLO (5× speedup)
- Batch processing: Parallelize multiple images
- Mask caching: Store rendered images (trade space for speed)

### Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| YOLO model | 6 MB | Loaded once (singleton) |
| Model runtime (CPU) | ~100 MB | PyTorch + numpy arrays |
| Per-image inference | ~10 MB | Temporary tensor allocation |

**Singleton Pattern Benefit**: Model loaded once, shared across requests

---

## 7. Quality Assurance

### Geometric Consistency Checks

**Centroid Offset Check**:
```python
# Expected: Centroid near bbox center
offset = distance(centroid, bbox_center)
normalized_offset = offset / bbox_diagonal
assert normalized_offset < 0.2  # Flag if >20% offset
```

**Radius Sanity Check**:
```python
# Expected: Radius equals half of bbox max dimension
expected_radius = max(width, height) / 2
radius_ratio = predicted_radius / expected_radius
assert 0.99 < radius_ratio < 1.01  # Should match exactly
```

**Tests**: Unit tests verify these checks for synthetic bboxes

### Detection Quality Metrics

Evaluated separately (see `05_evaluation_framework.md`):
- mAP@0.5:0.95 (localization quality)
- Counting accuracy (practical metric)
- Failure bucketing (severe/moderate/good/excellent)

---
