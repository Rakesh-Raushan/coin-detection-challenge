# Evaluation Strategy:

## 1. The Data Reality

When we examined the coin detection dataset, we encountered a fundamental constraint that shaped our entire evaluation approach: **we have bounding box annotations, but no ground truth geometry**.

Specifically, we observed:

- **Available**: COCO-format annotations with bounding boxes (`[x, y, width, height]`)
- **Missing**: Ground truth masks, centroids, radii, or any explicit circular geometry
- **Implication**: We cannot directly measure how well-detected circles align with actual coin boundaries

This is a common scenario in production ML systems. Annotation budgets are finite, and bounding boxes are significantly cheaper to produce than pixel-perfect segmentation masks. We needed an evaluation framework that acknowledges this reality rather than pretending it doesn't exist.

### What We Know vs. What We Assume

| Aspect | Status | Handling |
|--------|--------|----------|
| Object presence | ✓ Ground truth exists | Standard detection metrics (mAP) |
| Object count | ✓ Derivable from annotations | Direct comparison |
| Object location (coarse) | ✓ Bounding boxes available | IoU-based matching |
| Object shape (circle) | ✗ Assumed, not verified | Proxy metrics required |
| Object geometry (centroid, radius) | ✗ Not annotated | Self-consistency checks |

---

## 2. Metric Selection

Given the data reality above, we designed a three-tier metric system:

### Tier 1: Standard Detection Metrics

These are well-established and directly computable:

- **mAP@0.5:0.95**: The COCO-standard mean Average Precision across IoU thresholds from 0.5 to 0.95 in 0.05 increments. This gives us a robust measure of localization quality.
  
- **mAP@0.5**: A more lenient threshold that's useful for understanding baseline detection capability before fine localization.

- **Count Accuracy & MAE**: For coin counting applications, knowing how often we get the exact count right (and by how much we miss when we're wrong) is directly business-relevant.

### Tier 2: Geometric Consistency Proxies

This is where we had to get creative. Without ground truth geometry, we measure **self-consistency** instead:

#### Centroid Offset Ratio

```
centroid_offset_ratio = distance(predicted_centroid, bbox_center) / bbox_diagonal
```

**Why this metric?**

We observed that a well-fit circle should have its centroid near the center of the bounding box that contains it. If a detection algorithm is producing geometrically consistent outputs, the predicted centroid and the bbox center should be close.

- **Value near 0**: Centroid is well-centered within the detection
- **Value >> 0**: Centroid is offset, suggesting geometric inconsistency

This doesn't tell us if we're *correct*, but it tells us if we're *internally consistent*. Large offsets are a red flag worth investigating.

#### Radius Consistency Ratio

```
radius_consistency_ratio = predicted_radius / (min(bbox_width, bbox_height) / 2)
```

**Why this metric?**

For circular objects, the radius should relate predictably to the bounding box dimensions. A circle inscribed in a square bbox should have `radius ≈ min(w, h) / 2`.

- **Value ≈ 1.0**: Predicted radius matches bbox geometry
- **Value << 1.0**: Predicted radius is smaller than expected
- **Value >> 1.0**: Predicted radius exceeds bbox bounds (invalid)

We track both mean and P95 to understand typical behavior and outlier severity.

### Tier 3: Shape Diagnostics

We noticed that not all "coin" detections are actually circular. Some objects appear elliptical due to camera perspective, occlusion, or detection errors.

#### Ellipse Eccentricity

```
eccentricity = sqrt(1 - (minor_axis² / major_axis²))
```

We derive this from bbox dimensions (using width and height as proxies for ellipse axes).

**Bucketing thresholds**:

| Eccentricity | Category | Interpretation |
|--------------|----------|----------------|
| < 0.3 | Near-circular | Object appears round (expected for coins) |
| 0.3 – 0.6 | Moderately slanted | Slight perspective distortion or oval shape |
| > 0.6 | Highly slanted | Significant elongation (warrants investigation) |

These thresholds were set based on geometric interpretation:
- `e = 0` is a perfect circle
- `e = 0.3` corresponds to an ellipse where `b/a ≈ 0.95` (barely noticeable)
- `e = 0.6` corresponds to `b/a ≈ 0.8` (clearly oval)
- `e → 1` approaches a line segment

---

## 3. Bucketing Logic

We needed a way to prioritize which images require human review. Not all errors are equal—a single missed coin in a pile of 50 is different from missing all coins in a simple scene.

### Failure Severity Buckets

We defined four buckets based on observable characteristics:

#### Severe Failures

```python
if abs(pred_count - gt_count) >= 3 or recall == 0:
    return "severe"
```

**Rationale**: 
- Missing 3+ objects indicates systematic failure, not noise
- Zero recall means complete detection failure—the model saw nothing

These cases warrant immediate investigation and potential model retraining.

#### Moderate Failures

```python
if abs(pred_count - gt_count) in (1, 2):
    return "moderate"
```

**Rationale**:
- Off by 1-2 objects is common at detection boundaries
- Could be threshold sensitivity, occlusion, or edge cases
- Worth reviewing but not urgent

#### Good

```python
if pred_count == gt_count and mean_iou >= 0.5:
    return "good"
```

**Rationale**:
- Correct count shows the model understands the scene
- IoU ≥ 0.5 is the standard "correct detection" threshold
- Model is performing acceptably

#### Excellent

```python
if pred_count == gt_count and mean_iou >= 0.75:
    return "excellent"
```

**Rationale**:
- Correct count + tight localization
- IoU ≥ 0.75 indicates precise bounding boxes
- These are the cases we want to preserve during model updates

### Why These Specific Thresholds?

The `count_diff >= 3` threshold for severe failures comes from practical considerations:

1. **Statistical significance**: With typical images containing 5-20 coins, a difference of 3+ represents a 15-60% error rate.

2. **Debugging priority**: We can manually review severe failures in reasonable time if they're kept to a small percentage.

3. **User impact**: In downstream applications (e.g., counting coins for banking), missing 3+ objects is likely to trigger manual verification anyway.

The IoU thresholds (0.5 for "good", 0.75 for "excellent") align with PASCAL VOC and COCO conventions, making our results comparable to published benchmarks.

---

## 4. Schema Design

We split evaluation outputs into four distinct files, each serving a specific consumer:

### A. `evaluation_summary.json` — The Executive View

**Consumer**: Product managers, dashboards, CI/CD pipelines

**Design principle**: Maximum signal, minimum noise

```json
{
  "run_id": "...",
  "model_id": "...",
  "metrics": { ... },
  "failure_analysis": { ... },
  "shape_distribution": { ... }
}
```

**What we excluded**:
- Raw file paths (not actionable at executive level)
- Per-image breakdowns (too granular)
- Intermediate computations (implementation details)

This file should answer: "Is the model good enough to ship?" in under 5 seconds.

### B. `evaluation_provenance.json` — The Audit Trail

**Consumer**: MLOps, compliance, reproducibility workflows

**Design principle**: Everything needed to reproduce this exact evaluation

```json
{
  "timestamp": "...",
  "git_commit": "...",
  "model": { ... },
  "dataset": { ... },
  "config": { ... }
}
```

**Why separate from summary?**

Provenance data is stable (same model + data = same provenance) while metrics may vary with code changes. Separating them allows:
- Efficient diffing between runs
- Clear attribution when metrics change
- Compliance audits without metric noise

### C. `per_image_analysis.jsonl` — The Debug Log

**Consumer**: ML engineers diagnosing failures

**Design principle**: One line per image, grep-friendly

Format: JSON Lines (`.jsonl`)

```json
{"image_id": "123", "pred_count": 5, "gt_count": 7, "failure_bucket": "moderate", ...}
{"image_id": "124", "pred_count": 3, "gt_count": 3, "failure_bucket": "excellent", ...}
```

**Why JSONL instead of JSON?**

1. **Streaming**: Can process 100k images without loading all into memory
2. **Grep-able**: `grep '"failure_bucket": "severe"' per_image_analysis.jsonl`
3. **Append-friendly**: Can add images without rewriting the file
4. **Tool compatibility**: Works with `jq`, pandas, DuckDB natively

### D. `failure_examples.json` — The Visualization Queue

**Consumer**: Visualization scripts, error analysis notebooks

**Design principle**: Ready-to-use image IDs for each failure category

```json
{
  "severe_failure": ["id_1", "id_2", ...],
  "excellent": ["id_7", "id_9", ...]
}
```

**Why Top-K selection?**

- **Severe failures**: We want to see the worst cases first
- **Excellent cases**: We want to protect these during model changes (regression testing)
- **Configurable K**: Balances review time vs. coverage

---

## 5. Future Extensions

This framework is designed to accommodate growth:

1. **When GT geometry becomes available**: The proxy metrics can be replaced with direct comparison metrics while maintaining the same schema.

2. **Multi-class support**: The current single-class (coin) design can extend to category-specific metrics.

3. **Temporal tracking**: Run IDs and timestamped directories enable drift detection across deployment versions.

4. **A/B testing**: Summary schemas can be diffed programmatically to compare model variants.

---

## Conclusion

Our evaluation framework acknowledges the messy reality of production ML: imperfect annotations, proxy metrics, and the need for multiple audiences. By designing explicit schemas and documenting our reasoning, we create an evaluation system that's not just functional but *maintainable*.

The goal isn't perfect metrics—it's **actionable metrics** that help us improve the model and ship with confidence.
