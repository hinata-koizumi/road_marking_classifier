from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import laspy
import numpy as np
import open3d as o3d

from ..types import PointCloud

LOGGER = logging.getLogger(__name__)


def load_point_cloud(path: Path) -> PointCloud:
    """Load LAS/LAZ/PCD point cloud into memory."""
    suffix = path.suffix.lower()
    LOGGER.info("Loading point cloud %s", path)
    if suffix in {".las", ".laz"}:
        return _load_las(path)
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


def _load_las(path: Path) -> PointCloud:
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z), dtype=np.float64).T
    intensities = las.intensity.astype(np.float32)
    epsg = getattr(las.header, "epsg", None)
    if epsg is None:
        crs = las.header.parse_crs()
        if crs is not None:
            try:
                epsg = crs.to_epsg()
            except Exception:
                epsg = None
    metadata = {"scale": float(las.header.scales[0])}
    return PointCloud(points, intensities, epsg, metadata, source_path=path)


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
