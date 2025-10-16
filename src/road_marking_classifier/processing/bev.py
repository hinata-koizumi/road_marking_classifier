from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter

from ..types import PointCloud

LOGGER = logging.getLogger(__name__)


@dataclass
class BevResult:
    image: np.ndarray
    origin_xy: Tuple[float, float]
    resolution: float


def generate_bev(
    pc: PointCloud,
    mask: np.ndarray,
    resolution: float,
    roi_radius: float,
    apply_smoothing: bool = True,
) -> BevResult:
    """Rasterize intensity into a top-down BEV grid."""
    points = pc.points[mask]
    intensities = pc.intensities[mask]
    if len(points) == 0:
        raise ValueError("No points selected for BEV generation.")

    center = points[:, :2].mean(axis=0)
    min_xy = center - roi_radius
    max_xy = center + roi_radius
    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))
    LOGGER.info("BEV grid size %dx%d (res=%.3f)", width, height, resolution)

    grid = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros_like(grid)
    x_idx = ((points[:, 0] - min_xy[0]) / resolution).astype(int)
    y_idx = ((max_xy[1] - points[:, 1]) / resolution).astype(int)
    valid = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    x_idx, y_idx = x_idx[valid], y_idx[valid]
    np.add.at(grid, (y_idx, x_idx), intensities[valid])
    np.add.at(counts, (y_idx, x_idx), 1)
    nonzero = counts > 0
    grid[nonzero] /= counts[nonzero]
    if apply_smoothing:
        grid = median_filter(grid, size=3)
    return BevResult(image=grid, origin_xy=(min_xy[0], max_xy[1]), resolution=resolution)
