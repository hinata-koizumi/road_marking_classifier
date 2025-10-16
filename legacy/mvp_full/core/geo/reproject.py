from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import laspy
import numpy as np
import open3d as o3d
from pyproj import CRS, Transformer

from core.common.datatypes import PointCloud

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


def _load_las(path: Path) -> PointCloud:
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z), dtype=np.float64).T
    intensities = las.intensity.astype(np.float32)
    epsg = las.header.epsg if las.header.epsg else None
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


def ensure_epsg(pc: PointCloud, epsg: Optional[int]) -> PointCloud:
    """Ensure EPSG is set; if not, use provided or fallback to default."""
    if pc.epsg:
        return pc
    if epsg:
        pc.epsg = epsg
        return pc
    raise ValueError("EPSG code missing and not provided via CLI/config.")


def reproject_points(
    pc: PointCloud,
    target_epsg: int,
    reference_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> PointCloud:
    """Reproject to target CRS; optionally apply rigid alignment."""
    if pc.epsg == target_epsg and not reference_pairs:
        return pc

    LOGGER.info("Reprojecting from EPSG %s to %s", pc.epsg, target_epsg)
    transformer = Transformer.from_crs(
        CRS.from_epsg(pc.epsg), CRS.from_epsg(target_epsg), always_xy=True
    )
    x, y, z = transformer.transform(pc.points[:, 0], pc.points[:, 1], pc.points[:, 2])
    reproj_points = np.vstack((x, y, z)).T

    if reference_pairs:
        LOGGER.info("Applying rigid alignment using %d control points", len(reference_pairs))
        reproj_points = _rigid_alignment(reproj_points, reference_pairs)

    return PointCloud(
        points=reproj_points,
        intensities=pc.intensities,
        epsg=target_epsg,
        metadata=pc.metadata,
        source_path=pc.source_path,
    )


def _rigid_alignment(
    points: np.ndarray,
    reference_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    src = np.vstack([p[0] for p in reference_pairs])
    dst = np.vstack([p[1] for p in reference_pairs])
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    aligned = (R @ (points - src_centroid).T).T + t
    rmse = np.sqrt(np.mean(np.sum((aligned[: len(dst)] - dst) ** 2, axis=1)))
    if rmse > 0.03:
        LOGGER.warning("Alignment RMSE %.3fm exceeds tolerance 3cm", rmse)
    return aligned


def run_pdal_reprojection(
    input_path: Path,
    output_path: Path,
    source_epsg: int,
    target_epsg: int,
) -> None:
    """Execute PDAL filters.reprojection via subprocess."""
    pipeline = {
        "pipeline": [
            str(input_path),
            {
                "type": "filters.reprojection",
                "in_srs": f"EPSG:{source_epsg}",
                "out_srs": f"EPSG:{target_epsg}",
            },
            str(output_path),
        ]
    }
    pipeline_path = output_path.with_suffix(".pdal.json")
    pipeline_path.write_text(json.dumps(pipeline, indent=2))
    LOGGER.info("Running PDAL pipeline %s", pipeline_path)
    subprocess.run(["pdal", "pipeline", str(pipeline_path)], check=True)
