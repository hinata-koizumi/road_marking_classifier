from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d

from ..types import PointCloud

LOGGER = logging.getLogger(__name__)


def load_point_cloud(path: Path) -> PointCloud:
    """Load PCD point cloud into memory."""
    suffix = path.suffix.lower()
    LOGGER.info("Loading point cloud %s", path)
    if suffix == ".pcd":
        return _load_pcd(path)
    raise ValueError(f"Unsupported point cloud format: {suffix}")


def ensure_epsg(pc: PointCloud, epsg: Optional[int]) -> PointCloud:
    """Ensure EPSG is set; if not, use provided or fallback to default."""
    if pc.epsg:
        return pc
    if epsg:
        pc.epsg = epsg
        return pc
    raise ValueError("EPSG code missing and not provided via CLI/config.")


def _load_pcd(path: Path) -> PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    intensities = (
        np.asarray(pcd.colors).mean(axis=1) if pcd.has_colors() else np.ones(len(pcd.points))
    )
    metadata: Dict[str, float] = {}
    return PointCloud(
        points=np.asarray(pcd.points, dtype=np.float64),
        intensities=intensities.astype(np.float32),
        epsg=None,
        metadata=metadata,
        source_path=path,
    )
