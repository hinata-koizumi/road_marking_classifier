from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import open3d as o3d

from ..types import GroundPlane, PointCloud

LOGGER = logging.getLogger(__name__)


def voxel_downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    """Apply voxel downsampling to tame large point clouds."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.points))
    pcd.colors = o3d.utility.Vector3dVector(
        np.repeat(pc.intensities[:, None], 3, axis=1)
    )
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    LOGGER.info("Voxel downsample: %d -> %d points", len(pc.points), len(down.points))
    return PointCloud(
        points=np.asarray(down.points),
        intensities=np.asarray(down.colors)[:, 0],
        epsg=pc.epsg,
        metadata=pc.metadata,
        source_path=pc.source_path,
    )


def estimate_ground_plane(pc: PointCloud, distance_threshold: float = 0.05) -> GroundPlane:
    """Estimate a dominant road plane using RANSAC."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.points))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=2000,
    )
    normal = np.array(plane_model[:3])
    d = plane_model[3]
    inlier_mask = np.zeros(len(pc.points), dtype=bool)
    inlier_mask[inliers] = True
    plane_points = pc.points[inlier_mask]
    dist = np.abs(plane_points @ normal + d) / np.linalg.norm(normal)
    rms = float(np.sqrt(np.mean(dist**2)))
    LOGGER.info("Ground plane RMS %.3fm", rms)
    return GroundPlane(normal=normal, d=d, inliers=inlier_mask, rms_error=rms)


def height_intensity_masks(
    pc: PointCloud,
    ground: GroundPlane,
    height_limits: Tuple[float, float],
    roi_radius: float,
    intensity_near: float,
    intensity_far: float,
    intensity_far_range: float,
) -> np.ndarray:
    """Build a boolean mask for candidate road marking points."""
    normal = ground.normal / np.linalg.norm(ground.normal)
    distances = (pc.points @ normal + ground.d)
    mask_height = (distances >= height_limits[0]) & (distances <= height_limits[1])
    intensities_norm = pc.intensities / (pc.intensities.max() + 1e-6)
    reference_xy = pc.points[ground.inliers][:, :2].mean(axis=0)
    radial = np.linalg.norm(pc.points[:, :2] - reference_xy, axis=1)
    scaled = np.clip(radial / max(intensity_far_range, 1e-6), 0.0, 1.0)
    dynamic_thresh = intensity_near + scaled * (intensity_far - intensity_near)
    mask_intensity = intensities_norm > dynamic_thresh
    mask = mask_height & mask_intensity
    radial = np.linalg.norm(pc.points[:, :2] - reference_xy, axis=1)
    mask &= radial <= roi_radius
    LOGGER.debug("Mask selected %d / %d points", mask.sum(), len(mask))
    return mask
