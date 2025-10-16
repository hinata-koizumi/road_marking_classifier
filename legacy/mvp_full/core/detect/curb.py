from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from shapely.geometry import LineString

from core.common.datatypes import PrimitiveGeometry

LOGGER = logging.getLogger(__name__)


@dataclass
class CurbConfig:
    height_step_min_m: float
    normal_angle_change_deg: float
    connect_gap_m: float


def detect_curbs(
    points: np.ndarray,
    normals: np.ndarray,
    heights: np.ndarray,
    config: Dict[str, float],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    cfg = CurbConfig(**config)
    gradients = np.gradient(heights)
    candidates: List[PrimitiveGeometry] = []
    step_mask = np.abs(gradients) > cfg.height_step_min_m
    indices = np.where(step_mask)[0]
    if len(indices) < 2:
        return candidates
    clusters = _cluster_indices(indices)
    for cluster in clusters:
        pts = points[cluster]
        line = LineString(pts[:, :2])
        if line.length < cfg.connect_gap_m:
            continue
        candidates.append(
            PrimitiveGeometry(
                kind="LINE",
                points=np.vstack((pts[0], pts[-1])),
                score=0.8,
                attributes={"length_m": float(line.length)},
                bbox=(pts.min(axis=0), pts.max(axis=0)),
                source_tile=source_tile,
            )
        )
    LOGGER.info("Curb candidates: %d", len(candidates))
    return candidates


def _cluster_indices(indices: np.ndarray, max_gap: int = 8) -> List[np.ndarray]:
    groups = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx - prev > max_gap:
            groups.append(np.arange(start, prev + 1))
            start = idx
        prev = idx
    groups.append(np.arange(start, prev + 1))
    return groups
