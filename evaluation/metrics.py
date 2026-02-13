# app/evaluation/metrics.py
"""
Evaluation metrics module.

Implements detection metrics (mAP), counting metrics, geometric consistency proxies,
shape diagnostics, and failure severity bucketing for coin detection evaluation.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import math
import numpy as np

from evaluation.utils import compute_iou


# ─────────────────────────────────────────────────────────────────────────────────
# Data Classes for Metric Results
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class MAPResults:
    """Results from mAP computation."""
    map_50_95: float
    map_50: float
    per_threshold_ap: Dict[float, float] = field(default_factory=dict)


@dataclass
class CountingResults:
    """Results from counting metrics computation."""
    count_accuracy: float
    mean_absolute_count_error: float
    over_count_rate: float
    under_count_rate: float


@dataclass
class GeometricProxyResults:
    """
    Geometric consistency proxy metrics.
    
    NOTE: These are PROXY metrics because we do not have ground truth geometry
    (centroid, radius) annotations. We compute consistency ratios against the
    predicted bounding box as a self-consistency check.
    
    - Radius Consistency Ratio: How well the predicted radius matches bbox dimensions
    - Centroid Offset Ratio: How centered the predicted centroid is within the bbox
    """
    radius_consistency_mean: float
    radius_consistency_p95: float
    centroid_offset_mean: float
    centroid_offset_p95: float


@dataclass
class ShapeDistribution:
    """
    Shape distribution based on ellipse eccentricity.
    
    Eccentricity formula: sqrt(1 - (b² / a²)) where a >= b
    - Near-circular (< 0.3): Objects close to perfect circles
    - Moderately slanted (0.3 - 0.6): Slight elliptical deformation
    - Highly slanted (> 0.6): Significant elliptical shape
    """
    near_circular_pct: float
    moderately_slanted_pct: float
    highly_slanted_pct: float


@dataclass
class FailureAnalysis:
    """
    Failure severity bucketing for error analysis.
    
    Buckets:
    - Severe: |pred_count - gt_count| >= 3 OR recall = 0
    - Moderate: |pred_count - gt_count| is 1 or 2
    - Good: Exact count + mean IoU >= 0.5
    - Excellent: Exact count + mean IoU >= 0.75
    """
    severe_failure_pct: float
    moderate_failure_pct: float
    good_pct: float
    excellent_pct: float


# ─────────────────────────────────────────────────────────────────────────────────
# Detection Metrics (mAP)
# ─────────────────────────────────────────────────────────────────────────────────


def compute_map(results: List[Dict], iou_thresholds: List[float]) -> MAPResults:
    """
    Compute mean Average Precision (mAP) across IoU thresholds.
    
    Args:
        results: List of dicts with 'preds' and 'gts' keys.
            - preds: [{"bbox": [x1,y1,x2,y2], "score": float}]
            - gts: [{"bbox": [x1,y1,x2,y2]}]
        iou_thresholds: List of IoU thresholds (e.g., [0.5, 0.55, ..., 0.95])
    
    Returns:
        MAPResults with mAP@0.5:0.95, mAP@0.5, and per-threshold APs.
    """
    aps = []
    per_threshold_ap = {}

    for iou_thresh in iou_thresholds:
        all_scores = []
        all_matches = []
        total_gts = 0

        for item in results:
            preds = sorted(item["preds"], key=lambda x: x["score"], reverse=True)
            gts = item["gts"].copy()
            total_gts += len(gts)
            matched = set()

            for pred in preds:
                best_iou = 0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if idx in matched:
                        continue
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_thresh:
                    matched.add(best_gt_idx)
                    all_matches.append(1)
                else:
                    all_matches.append(0)

                all_scores.append(pred["score"])

        if total_gts == 0:
            ap = 0.0
        else:
            sorted_indices = np.argsort(-np.array(all_scores))
            matches = np.array(all_matches)[sorted_indices] if all_matches else np.array([])

            if len(matches) > 0:
                tp = np.cumsum(matches)
                fp = np.cumsum(1 - matches)
                recalls = tp / total_gts
                precisions = tp / (tp + fp + 1e-8)
                ap = float(np.trapz(precisions, recalls))
            else:
                ap = 0.0

        aps.append(ap)
        per_threshold_ap[iou_thresh] = ap

    return MAPResults(
        map_50_95=float(np.mean(aps)) if aps else 0.0,
        map_50=float(aps[0]) if aps else 0.0,
        per_threshold_ap=per_threshold_ap
    )


# ─────────────────────────────────────────────────────────────────────────────────
# Counting Metrics
# ─────────────────────────────────────────────────────────────────────────────────


def compute_counting_metrics(results: List[Dict]) -> CountingResults:
    """
    Compute counting accuracy and error metrics.
    
    Args:
        results: List of dicts with 'preds' and 'gts' keys.
    
    Returns:
        CountingResults with accuracy, MAE, and over/under-count rates.
    """
    if not results:
        return CountingResults(
            count_accuracy=0.0,
            mean_absolute_count_error=0.0,
            over_count_rate=0.0,
            under_count_rate=0.0
        )

    correct = 0
    over_count = 0
    under_count = 0
    absolute_errors = []

    for item in results:
        pred_count = len(item["preds"])
        gt_count = len(item["gts"])
        diff = pred_count - gt_count

        absolute_errors.append(abs(diff))

        if diff == 0:
            correct += 1
        elif diff > 0:
            over_count += 1
        else:
            under_count += 1

    total = len(results)
    return CountingResults(
        count_accuracy=correct / total,
        mean_absolute_count_error=float(np.mean(absolute_errors)),
        over_count_rate=over_count / total,
        under_count_rate=under_count / total
    )


# ─────────────────────────────────────────────────────────────────────────────────
# Geometric Consistency Proxy Metrics
# ─────────────────────────────────────────────────────────────────────────────────


def compute_geometric_proxies(results: List[Dict]) -> Tuple[GeometricProxyResults, List[Dict]]:
    """
    Compute geometric consistency proxy metrics.
    
    NOTE: These are PROXY metrics. We do not have ground truth geometry annotations
    (centroids, radii). Instead, we measure self-consistency between predicted
    geometry (if available) and the bounding box.
    
    - Radius Consistency Ratio: pred_radius / (min(bbox_w, bbox_h) / 2)
      Ideally close to 1.0 for circular objects fitting within the bbox.
    
    - Centroid Offset Ratio: distance(pred_centroid, bbox_center) / bbox_diagonal
      Ideally close to 0.0 for well-centered predictions.
    
    Args:
        results: List of dicts with 'preds' containing bbox and optional geometry.
    
    Returns:
        Tuple of (GeometricProxyResults, per_prediction_metrics)
    """
    radius_ratios = []
    centroid_offsets = []
    per_pred_metrics = []

    for item in results:
        for pred in item["preds"]:
            bbox = pred["bbox"]
            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox_diagonal = math.sqrt(bbox_w**2 + bbox_h**2)
            expected_radius = min(bbox_w, bbox_h) / 2

            # Get predicted geometry or estimate from bbox
            pred_centroid = pred.get("centroid", bbox_center)
            pred_radius = pred.get("radius", expected_radius)

            # Radius consistency ratio
            radius_ratio = pred_radius / expected_radius if expected_radius > 0 else 1.0
            radius_ratios.append(radius_ratio)

            # Centroid offset ratio
            dx = pred_centroid[0] - bbox_center[0]
            dy = pred_centroid[1] - bbox_center[1]
            offset_distance = math.sqrt(dx**2 + dy**2)
            offset_ratio = offset_distance / bbox_diagonal if bbox_diagonal > 0 else 0.0
            centroid_offsets.append(offset_ratio)

            per_pred_metrics.append({
                "radius_consistency_ratio": radius_ratio,
                "centroid_offset_ratio": offset_ratio
            })

    if not radius_ratios:
        return GeometricProxyResults(
            radius_consistency_mean=0.0,
            radius_consistency_p95=0.0,
            centroid_offset_mean=0.0,
            centroid_offset_p95=0.0
        ), []

    return GeometricProxyResults(
        radius_consistency_mean=float(np.mean(radius_ratios)),
        radius_consistency_p95=float(np.percentile(radius_ratios, 95)),
        centroid_offset_mean=float(np.mean(centroid_offsets)),
        centroid_offset_p95=float(np.percentile(centroid_offsets, 95))
    ), per_pred_metrics


# ─────────────────────────────────────────────────────────────────────────────────
# Shape Diagnostics (Ellipse Eccentricity)
# ─────────────────────────────────────────────────────────────────────────────────


def compute_ellipse_eccentricity(bbox: List[float]) -> float:
    """
    Compute ellipse eccentricity from bounding box.
    
    Formula: eccentricity = sqrt(1 - (b² / a²))
    Where a = semi-major axis, b = semi-minor axis (a >= b)
    
    Returns value in [0, 1):
    - 0: Perfect circle
    - Approaching 1: Highly elongated ellipse
    """
    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    a = max(w, h) / 2  # Semi-major axis
    b = min(w, h) / 2  # Semi-minor axis

    if a == 0:
        return 0.0

    ratio = (b / a) ** 2
    eccentricity = math.sqrt(1 - ratio) if ratio <= 1 else 0.0
    return eccentricity


def compute_shape_distribution(results: List[Dict]) -> Tuple[ShapeDistribution, List[float]]:
    """
    Compute shape distribution based on ellipse eccentricity.
    
    Buckets:
    - Near-circular: eccentricity < 0.3
    - Moderately slanted: 0.3 <= eccentricity <= 0.6
    - Highly slanted: eccentricity > 0.6
    
    Args:
        results: List of dicts with 'preds' containing bboxes.
    
    Returns:
        Tuple of (ShapeDistribution, list of eccentricity values)
    """
    eccentricities = []

    for item in results:
        for pred in item["preds"]:
            ecc = compute_ellipse_eccentricity(pred["bbox"])
            eccentricities.append(ecc)

    if not eccentricities:
        return ShapeDistribution(
            near_circular_pct=0.0,
            moderately_slanted_pct=0.0,
            highly_slanted_pct=0.0
        ), []

    total = len(eccentricities)
    near_circular = sum(1 for e in eccentricities if e < 0.3)
    moderately_slanted = sum(1 for e in eccentricities if 0.3 <= e <= 0.6)
    highly_slanted = sum(1 for e in eccentricities if e > 0.6)

    return ShapeDistribution(
        near_circular_pct=near_circular / total * 100,
        moderately_slanted_pct=moderately_slanted / total * 100,
        highly_slanted_pct=highly_slanted / total * 100
    ), eccentricities


# ─────────────────────────────────────────────────────────────────────────────────
# Failure Severity Bucketing
# ─────────────────────────────────────────────────────────────────────────────────


def compute_per_image_iou_stats(item: Dict) -> Tuple[float, float]:
    """
    Compute mean IoU and recall for a single image.
    
    Returns:
        Tuple of (mean_iou, recall)
    """
    preds = item["preds"]
    gts = item["gts"]

    if not preds or not gts:
        return 0.0, 0.0

    # Compute best IoU for each prediction
    ious = []
    matched_gts = set()

    for pred in sorted(preds, key=lambda x: x.get("score", 0), reverse=True):
        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(gts):
            if idx in matched_gts:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou > 0:
            matched_gts.add(best_gt_idx)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    recall = len(matched_gts) / len(gts) if gts else 0.0

    return mean_iou, recall


def classify_failure_bucket(
    pred_count: int,
    gt_count: int,
    mean_iou: float,
    recall: float
) -> str:
    """
    Classify image into failure severity bucket.
    
    Buckets:
    - Severe: |pred_count - gt_count| >= 3 OR recall = 0
    - Moderate: |pred_count - gt_count| is 1 or 2
    - Good: Exact count + mean IoU >= 0.5
    - Excellent: Exact count + mean IoU >= 0.75
    
    Returns:
        One of: "severe", "moderate", "good", "excellent"
    """
    count_diff = abs(pred_count - gt_count)

    # Severe failures
    if count_diff >= 3 or (recall == 0 and gt_count > 0):
        return "severe"

    # Moderate failures
    if count_diff in (1, 2):
        return "moderate"

    # Exact count - check IoU quality
    if count_diff == 0:
        if mean_iou >= 0.75:
            return "excellent"
        elif mean_iou >= 0.5:
            return "good"
        else:
            return "moderate"

    return "moderate"


def compute_failure_analysis(results: List[Dict]) -> Tuple[FailureAnalysis, Dict[str, List]]:
    """
    Compute failure severity distribution across all images.
    
    Args:
        results: List of dicts with 'image_id', 'preds', and 'gts'.
    
    Returns:
        Tuple of (FailureAnalysis, dict mapping bucket -> list of image_ids)
    """
    if not results:
        return FailureAnalysis(
            severe_failure_pct=0.0,
            moderate_failure_pct=0.0,
            good_pct=0.0,
            excellent_pct=0.0
        ), {"severe": [], "moderate": [], "good": [], "excellent": []}

    buckets = {"severe": [], "moderate": [], "good": [], "excellent": []}

    for item in results:
        image_id = item.get("image_id", "unknown")
        pred_count = len(item["preds"])
        gt_count = len(item["gts"])
        mean_iou, recall = compute_per_image_iou_stats(item)

        bucket = classify_failure_bucket(pred_count, gt_count, mean_iou, recall)
        buckets[bucket].append(image_id)

    total = len(results)
    return FailureAnalysis(
        severe_failure_pct=len(buckets["severe"]) / total * 100,
        moderate_failure_pct=len(buckets["moderate"]) / total * 100,
        good_pct=len(buckets["good"]) / total * 100,
        excellent_pct=len(buckets["excellent"]) / total * 100
    ), buckets


# ─────────────────────────────────────────────────────────────────────────────────
# Per-Image Analysis
# ─────────────────────────────────────────────────────────────────────────────────


def compute_per_image_analysis(item: Dict) -> Dict:
    """
    Compute comprehensive per-image analysis for debug output.
    
    Args:
        item: Dict with 'image_id', 'preds', and 'gts'.
    
    Returns:
        Dict with all per-image metrics for per_image_analysis.jsonl
    """
    image_id = str(item.get("image_id", "unknown"))
    pred_count = len(item["preds"])
    gt_count = len(item["gts"])
    count_diff = pred_count - gt_count

    mean_iou, recall = compute_per_image_iou_stats(item)
    failure_bucket = classify_failure_bucket(pred_count, gt_count, mean_iou, recall)

    # Compute per-prediction geometric metrics (wrap in list for helper functions)
    single_item_list = [{"preds": item["preds"], "gts": item["gts"]}]
    geo_results, per_pred_geo = compute_geometric_proxies(single_item_list)
    _, eccentricities = compute_shape_distribution(single_item_list)

    avg_eccentricity = float(np.mean(eccentricities)) if eccentricities else 0.0
    avg_centroid_offset = float(np.mean([p["centroid_offset_ratio"] for p in per_pred_geo])) if per_pred_geo else 0.0
    avg_radius_consistency = float(np.mean([p["radius_consistency_ratio"] for p in per_pred_geo])) if per_pred_geo else 0.0

    return {
        "image_id": image_id,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "count_diff": count_diff,
        "mean_iou": round(mean_iou, 4),
        "centroid_offset_ratio": round(avg_centroid_offset, 4),
        "radius_consistency_ratio": round(avg_radius_consistency, 4),
        "ellipse_eccentricity": round(avg_eccentricity, 4),
        "failure_bucket": failure_bucket
    }


# ─────────────────────────────────────────────────────────────────────────────────
# Aggregate Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class AllMetrics:
    """Container for all computed metrics."""
    map_results: MAPResults
    counting_results: CountingResults
    geometric_proxies: GeometricProxyResults
    shape_distribution: ShapeDistribution
    failure_analysis: FailureAnalysis
    failure_buckets: Dict[str, List]
    per_image_analyses: List[Dict]


def compute_all_metrics(results: List[Dict], iou_thresholds: List[float]) -> AllMetrics:
    """
    Compute all metrics in a single pass.
    
    Args:
        results: List of dicts with 'image_id', 'preds', and 'gts'.
        iou_thresholds: List of IoU thresholds for mAP computation.
    
    Returns:
        AllMetrics containing all computed metrics.
    """
    # Detection metrics
    map_results = compute_map(results, iou_thresholds)

    # Counting metrics
    counting_results = compute_counting_metrics(results)

    # Geometric consistency proxies
    geometric_proxies, _ = compute_geometric_proxies(results)

    # Shape distribution
    shape_distribution, _ = compute_shape_distribution(results)

    # Failure analysis
    failure_analysis, failure_buckets = compute_failure_analysis(results)

    # Per-image analysis
    per_image_analyses = [compute_per_image_analysis(item) for item in results]

    return AllMetrics(
        map_results=map_results,
        counting_results=counting_results,
        geometric_proxies=geometric_proxies,
        shape_distribution=shape_distribution,
        failure_analysis=failure_analysis,
        failure_buckets=failure_buckets,
        per_image_analyses=per_image_analyses
    )
