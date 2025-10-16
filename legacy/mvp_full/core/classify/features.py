from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from core.common.datatypes import ClassifiedPrimitive, PrimitiveGeometry

LOGGER = logging.getLogger(__name__)


def compute_features(primitives: List[PrimitiveGeometry]) -> np.ndarray:
    """Derive geometric features."""
    feature_list: List[List[float]] = []
    for prim in primitives:
        length = float(np.linalg.norm(prim.points[-1, :2] - prim.points[0, :2]))
        width = float(np.linalg.norm(prim.points.max(axis=0)[:2] - prim.points.min(axis=0)[:2]))
        curvature = prim.attributes.get("curvature", 0.0)
        stripe_count = prim.attributes.get("stripe_count", 0.0)
        parallelism = prim.attributes.get("parallelism", 0.0)
        angle = _line_angle_deg(prim.points)
        intensity = prim.attributes.get("intensity_mean", 0.5)
        feature_list.append([length, width, curvature, stripe_count, parallelism, angle, intensity])
    return np.array(feature_list, dtype=np.float32)


def _line_angle_deg(points: np.ndarray) -> float:
    direction = points[-1, :2] - points[0, :2]
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def attach_probabilities(
    primitives: List[PrimitiveGeometry],
    labels: List[str],
    probs: np.ndarray,
    layer_map: Dict[str, str],
) -> List[ClassifiedPrimitive]:
    classified: List[ClassifiedPrimitive] = []
    for prim, label, prob in zip(primitives, labels, probs):
        layer = layer_map.get(label, "QC_REVIEW")
        classified.append(
            ClassifiedPrimitive(
                layer=layer, primitive=prim, class_name=label, probability=float(prob)
            )
        )
    return classified
