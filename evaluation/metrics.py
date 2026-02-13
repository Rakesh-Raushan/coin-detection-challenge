# app/evaluation/metrics.py

from typing import List, Dict
import numpy as np
from evaluation.utils import compute_iou


def compute_map(results: List[Dict], iou_thresholds: List[float]) -> Dict:
    """
    results: [
        {
            "preds": [{"bbox": [x1,y1,x2,y2], "score": float}],
            "gts": [{"bbox": [x1,y1,x2,y2]}]
        }
    ]
    """

    aps = []

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
            aps.append(0)
            continue

        # Precision-recall
        sorted_indices = np.argsort(-np.array(all_scores))
        matches = np.array(all_matches)[sorted_indices]

        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)

        recalls = tp / total_gts
        precisions = tp / (tp + fp + 1e-8)

        ap = np.trapz(precisions, recalls)
        aps.append(ap)

    return {
        "mAP@0.5:0.95": float(np.mean(aps)),
        "mAP@0.5": float(aps[0])
    }


def compute_count_accuracy(results: List[Dict]) -> float:
    correct = 0
    total = len(results)

    for item in results:
        if len(item["preds"]) == len(item["gts"]):
            correct += 1

    return correct / total if total > 0 else 0.0
