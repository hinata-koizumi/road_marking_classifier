from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from .processing.bev import generate_bev
from .processing.classify import LAYER_MAP, classify_primitives
from .config import SimplePipelineConfig
from .processing.detect import detect_crosswalks, detect_lines_from_bev
from .io.pointcloud import ensure_epsg, load_point_cloud
from .processing.preprocess import estimate_ground_plane, height_intensity_masks, voxel_downsample
from .types import ClassifiedPrimitive, PointCloud, PrimitiveGeometry

LOGGER = logging.getLogger(__name__)


LINE_CONFIG: Dict[str, Dict[str, float]] = {
    "ppht": {
        "rho": 1.0,
        "theta_deg": 1.0,
        "threshold": 50,
        "min_line_length": 2.0,
        "max_line_gap": 0.8,
    },
    "reconnect": {
        "max_gap_m": 0.6,
        "angle_tol_deg": 5.0,
    },
    "nms": {
        "distance_m": 0.4,
        "angle_deg": 5.0,
        "overlap_ratio": 0.8,
    },
    "curvature": {
        "arc_min_length_m": 5.0,
    },
    "cluster": {
        "eps_m": 1.5,
        "min_samples": 2.0,
    },
}

CROSSWALK_CONFIG: Dict[str, float] = {
    "min_stripes": 3,
    "stripe_spacing_m": 0.5,
    "stripe_width_m": 0.4,
    "orientation_tolerance_deg": 12.0,
}


def run_pipeline(input_path: Path, output_path: Path, config: SimplePipelineConfig) -> Path:
    """
    Execute the simplified point-cloud → DXF pipeline.

    The routine focuses on the core steps (load → filter → BEV → detect → DXF)
    and omits the heavy MVP scaffolding. The result is a curated layer set
    containing ROAD_LINE, STOP_LINE, and CROSSWALK.
    """
    LOGGER.info("Running simple pipeline on %s", input_path)
    pc = load_point_cloud(input_path)
    pc = _ensure_epsg(pc, config)
    pc = voxel_downsample(pc, voxel_size=config.voxel_size_m)
    ground = estimate_ground_plane(pc)
    mask = height_intensity_masks(
        pc,
        ground,
        config.height_limits_m,
        config.roi_radius_m,
        config.intensity_near_threshold,
        config.intensity_far_threshold,
        config.intensity_far_range_m,
    )
    bev = generate_bev(
        pc=pc,
        mask=mask,
        resolution=config.bev_resolution_m,
        roi_radius=config.roi_radius_m,
        apply_smoothing=True,
    )
    line_config = _line_config(config)
    primitives = _detect_primitives(
        bev.image,
        bev.resolution,
        line_config,
        source_tile=input_path.stem,
    )
    classified = classify_primitives(
        primitives,
        config.stop_line_length_m,
        config.stop_line_distance_m,
        config.stop_line_angle_tolerance_deg,
    )
    _write_output(classified, output_path, config)
    LOGGER.info("DXF exported to %s", output_path)
    return output_path


def _ensure_epsg(pc: PointCloud, config: SimplePipelineConfig) -> PointCloud:
    if config.epsg is not None:
        return ensure_epsg(pc, config.epsg)
    if pc.epsg is None:
        LOGGER.warning("EPSG code missing; proceeding with raw coordinates.")
    return pc


def _detect_primitives(
    bev_image: np.ndarray,
    bev_resolution: float,
    line_config: Dict[str, Dict[str, float]],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    lines = detect_lines_from_bev(
        bev_image,
        bev_resolution,
        line_config,
        source_tile=source_tile,
    )
    crosswalks = detect_crosswalks(
        bev_image,
        bev_resolution,
        CROSSWALK_CONFIG,
        source_tile=source_tile,
    )
    return lines + crosswalks


def _write_output(
    primitives: List[ClassifiedPrimitive],
    output_path: Path,
    config: SimplePipelineConfig,
) -> None:
    from .io.dxf import write_dxf

    write_dxf(
        primitives,
        output_path,
        LAYER_MAP,
        dp_tolerance=config.douglas_peucker_m,
        timestamp_format=config.timestamp_format,
    )


def _line_config(config: SimplePipelineConfig) -> Dict[str, Dict[str, float]]:
    cfg = {key: value.copy() for key, value in LINE_CONFIG.items()}
    cluster_cfg = cfg.setdefault("cluster", {})
    cluster_cfg["eps_m"] = config.line_cluster_eps_m
    cluster_cfg["min_samples"] = config.line_cluster_min_samples
    return cfg
