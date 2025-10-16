from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricResult:
    precision: Dict[str, float]
    recall: Dict[str, float]
    geo_error_mean: float
    geo_error_p95: float
    runtime_mean: float


def compute_geometry_error(pred: List[LineString], gt: List[LineString]) -> Tuple[float, float]:
    dists = []
    for p in pred:
        nearest = min((p.distance(g) for g in gt), default=np.inf)
        dists.append(nearest)
    if not dists:
        return float("inf"), float("inf")
    arr = np.array(dists)
    return float(arr.mean()), float(np.percentile(arr, 95))


def compute_pr(
    pred_labels: List[str],
    gt_labels: List[str],
    classes: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    precision = {}
    recall = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == g == cls)
        fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_labels, gt_labels) if g == cls and p != cls)
        precision[cls] = tp / (tp + fp + 1e-6)
        recall[cls] = tp / (tp + fn + 1e-6)
    return precision, recall
