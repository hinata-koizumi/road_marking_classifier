from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString

from core.common.datatypes import PrimitiveGeometry

LOGGER = logging.getLogger(__name__)


@dataclass
class LineDetectionConfig:
    rho: float
    theta_deg: float
    threshold: int
    min_line_length: float
    max_line_gap: float
    reconnect_max_gap: float
    reconnect_angle_deg: float
    nms_distance: float
    nms_angle_deg: float
    nms_overlap: float
    arc_min_length: float


def detect_lines_from_bev(
    bev_img: np.ndarray,
    bev_resolution: float,
    config: Dict[str, Dict[str, float]],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    """Run PPHT on BEV and consolidate dashed lines."""
    cfg = LineDetectionConfig(
        rho=config["ppht"]["rho"],
        theta_deg=config["ppht"]["theta_deg"],
        threshold=config["ppht"]["threshold"],
        min_line_length=config["ppht"]["min_line_length"],
        max_line_gap=config["ppht"]["max_line_gap"],
        reconnect_max_gap=config["reconnect"]["max_gap_m"],
        reconnect_angle_deg=config["reconnect"]["angle_tol_deg"],
        nms_distance=config["nms"]["distance_m"],
        nms_angle_deg=config["nms"]["angle_deg"],
        nms_overlap=config["nms"]["overlap_ratio"],
        arc_min_length=config["curvature"]["arc_min_length_m"],
    )

    edges = cv2.Canny((bev_img * 255).astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=cfg.rho,
        theta=np.deg2rad(cfg.theta_deg),
        threshold=cfg.threshold,
        minLineLength=cfg.min_line_length / bev_resolution,
        maxLineGap=cfg.max_line_gap / bev_resolution,
    )
    if lines is None:
        LOGGER.info("No lines detected.")
        return []

    primitives: List[PrimitiveGeometry] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        world = _bev_pixels_to_world((x1, y1, x2, y2), bev_resolution, bev_img.shape)
        length = np.linalg.norm(world[1] - world[0])
        primitives.append(
            PrimitiveGeometry(
                kind="LINE",
                points=np.array(world),
                score=1.0,
                attributes={"length_m": length},
                bbox=_bbox(world),
                source_tile=source_tile,
            )
        )

    merged = _merge_lines(primitives, cfg)
    LOGGER.info("Lines detected %d -> %d after merging", len(primitives), len(merged))
    return merged


def _bev_pixels_to_world(
    coords: Tuple[int, int, int, int], resolution: float, shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = coords
    width, height = shape[1], shape[0]

    def pix_to_world(x: int, y: int) -> np.ndarray:
        return np.array([(x - width / 2) * resolution, (height / 2 - y) * resolution, 0.0])

    return pix_to_world(x1, y1), pix_to_world(x2, y2)


def _bbox(points: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.vstack(points)
    return pts.min(axis=0), pts.max(axis=0)


def _merge_lines(
    primitives: List[PrimitiveGeometry],
    cfg: LineDetectionConfig,
) -> List[PrimitiveGeometry]:
    merged: List[PrimitiveGeometry] = []
    used = np.zeros(len(primitives), dtype=bool)
    for idx, prim in enumerate(primitives):
        if used[idx]:
            continue
        group = [prim]
        for jdx in range(idx + 1, len(primitives)):
            if used[jdx]:
                continue
            if _should_merge(prim, primitives[jdx], cfg):
                group.append(primitives[jdx])
                used[jdx] = True
        merged_geom = _refit_line(group)
        merged.append(merged_geom)
    return _nms(merged, cfg)


def _should_merge(a: PrimitiveGeometry, b: PrimitiveGeometry, cfg: LineDetectionConfig) -> bool:
    line_a = LineString(a.points[:, :2])
    line_b = LineString(b.points[:, :2])
    dist = line_a.distance(line_b)
    if dist > cfg.reconnect_max_gap:
        return False
    angle = _angle_between(a.points, b.points)
    return angle <= cfg.reconnect_angle_deg


def _angle_between(pa: np.ndarray, pb: np.ndarray) -> float:
    def direction(p):
        vec = p[1, :2] - p[0, :2]
        return vec / (np.linalg.norm(vec) + 1e-6)

    da = direction(pa)
    db = direction(pb)
    cos_theta = np.clip(np.dot(da, db), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _refit_line(group: List[PrimitiveGeometry]) -> PrimitiveGeometry:
    points = np.vstack([g.points for g in group])
    centroid = points.mean(axis=0)
    centered = points - centroid
    u, _, _ = np.linalg.svd(centered[:, :2], full_matrices=False)
    direction = u[:, 0]
    proj = centered[:, :2] @ direction
    min_pt = centroid[:2] + direction * proj.min()
    max_pt = centroid[:2] + direction * proj.max()
    world = np.array([[min_pt[0], min_pt[1], 0.0], [max_pt[0], max_pt[1], 0.0]])
    return PrimitiveGeometry(
        kind="LINE",
        points=world,
        score=float(np.mean([g.score for g in group])),
        attributes={"length_m": np.linalg.norm(world[1] - world[0])},
        bbox=_bbox(world),
        source_tile=group[0].source_tile,
    )


def _nms(prims: List[PrimitiveGeometry], cfg: LineDetectionConfig) -> List[PrimitiveGeometry]:
    keep: List[PrimitiveGeometry] = []
    for prim in prims:
        should_keep = True
        for existing in keep:
            if _angle_between(prim.points, existing.points) < cfg.nms_angle_deg:
                line_a = LineString(prim.points[:, :2])
                line_b = LineString(existing.points[:, :2])
                if line_a.distance(line_b) < cfg.nms_distance:
                    overlap = min(line_a.length, line_b.length) / max(line_a.length, line_b.length)
                    if overlap > cfg.nms_overlap:
                        should_keep = False
                        break
        if should_keep:
            keep.append(prim)
    return keep
