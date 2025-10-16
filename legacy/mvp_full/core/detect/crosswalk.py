from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from shapely.geometry import Polygon

from core.common.datatypes import PrimitiveGeometry

LOGGER = logging.getLogger(__name__)


@dataclass
class CrosswalkConfig:
    min_stripes: int
    stripe_spacing_m: float
    stripe_width_m: float
    orientation_tolerance_deg: float


def detect_crosswalks(
    bev_img: np.ndarray,
    bev_resolution: float,
    config: Dict[str, float],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    cfg = CrosswalkConfig(**config)
    binary = (bev_img > np.percentile(bev_img[bev_img > 0], 75)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[PrimitiveGeometry] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = sorted(rect[1])
        stripes = int(round(height * bev_resolution / cfg.stripe_spacing_m))
        if stripes < cfg.min_stripes:
            continue
        box = cv2.boxPoints(rect)
        world = np.array(
            [
                [
                    (pt[0] - bev_img.shape[1] / 2) * bev_resolution,
                    (bev_img.shape[0] / 2 - pt[1]) * bev_resolution,
                    0.0,
                ]
                for pt in box
            ]
        )
        polygon = Polygon(world[:, :2])
        if not polygon.is_valid or polygon.area < 0.5:
            continue
        candidates.append(
            PrimitiveGeometry(
                kind="LWPOLYLINE",
                points=world,
                score=1.0,
                attributes={"stripe_count": stripes, "width_m": width * bev_resolution},
                bbox=(world.min(axis=0), world.max(axis=0)),
                source_tile=source_tile,
            )
        )
    LOGGER.info("Crosswalk candidates: %d", len(candidates))
    return candidates
