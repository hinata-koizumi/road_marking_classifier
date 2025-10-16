from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

try:
    from sklearn.cluster import DBSCAN
except ImportError:  # pragma: no cover - optional dependency
    DBSCAN = None

from ..types import PrimitiveGeometry

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
    cluster_eps: float
    cluster_min_samples: int


@dataclass
class CrosswalkConfig:
    min_stripes: int
    stripe_spacing_m: float
    stripe_width_m: float
    orientation_tolerance_deg: float


def detect_lines_from_bev(
    bev_img: np.ndarray,
    bev_resolution: float,
    config: Dict[str, Dict[str, float]],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    """Run PPHT on the BEV raster and consolidate dashed lines."""
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
        cluster_eps=config.get("cluster", {}).get("eps_m", 0.0),
        cluster_min_samples=int(config.get("cluster", {}).get("min_samples", 1)),
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
    clustered = _cluster_lines(merged, cfg)
    LOGGER.info(
        "Lines detected %d -> %d after merging, %d retained after clustering",
        len(primitives),
        len(merged),
        len(clustered),
    )
    return clustered


def detect_crosswalks(
    bev_img: np.ndarray,
    bev_resolution: float,
    config: Dict[str, float],
    source_tile: str,
) -> List[PrimitiveGeometry]:
    """Detect crosswalk-like blobs by looking for stripe patterns."""
    cfg = CrosswalkConfig(**config)
    nonzero = bev_img[bev_img > 0]
    if len(nonzero) == 0:
        return []
    binary = (bev_img > np.percentile(nonzero, 75)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[PrimitiveGeometry] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = sorted(rect[1])
        if width == 0 or height == 0:
            continue
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
        stripe_angle_deg, walkway_angle_deg = _rect_orientation(world[:, :2])
        center_point = polygon.centroid
        length_m = max(width, height) * bev_resolution
        width_m = min(width, height) * bev_resolution
        candidates.append(
            PrimitiveGeometry(
                kind="LWPOLYLINE",
                points=world,
                score=1.0,
                attributes={
                    "stripe_count": stripes,
                    "length_m": length_m,
                    "width_m": width_m,
                    "center_x": float(center_point.x),
                    "center_y": float(center_point.y),
                    "stripe_angle_deg": stripe_angle_deg,
                    "walkway_angle_deg": walkway_angle_deg,
                },
                bbox=(world.min(axis=0), world.max(axis=0)),
                source_tile=source_tile,
            )
        )
    LOGGER.info("Crosswalk candidates: %d", len(candidates))
    return candidates


def _rect_orientation(points_2d: np.ndarray) -> Tuple[float, float]:
    """Estimate stripe and walkway angles (deg) for a crosswalk rectangle."""
    if len(points_2d) < 2:
        return 0.0, 90.0
    centered = points_2d - points_2d.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    stripe_vec = vh[0]
    stripe_angle = float(np.degrees(np.arctan2(stripe_vec[1], stripe_vec[0])) % 180.0)
    walkway_angle = (stripe_angle + 90.0) % 180.0
    return stripe_angle, walkway_angle


def _bev_pixels_to_world(
    coords: Tuple[int, int, int, int],
    resolution: float,
    shape: Tuple[int, int],
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
    _, _, vh = np.linalg.svd(centered[:, :2], full_matrices=False)
    direction = vh[0]
    direction /= np.linalg.norm(direction) + 1e-6
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


def _cluster_lines(
    primitives: List[PrimitiveGeometry],
    cfg: LineDetectionConfig,
) -> List[PrimitiveGeometry]:
    if DBSCAN is None or cfg.cluster_eps <= 0 or cfg.cluster_min_samples <= 1:
        return primitives
    if not primitives:
        return primitives
    centers = np.array([prim.points[:, :2].mean(axis=0) for prim in primitives])
    clustering = DBSCAN(eps=cfg.cluster_eps, min_samples=cfg.cluster_min_samples)
    labels = clustering.fit_predict(centers)
    keep: List[PrimitiveGeometry] = []
    for label, prim in zip(labels, primitives):
        if label != -1:
            keep.append(prim)
    LOGGER.info(
        "Line clusters retained %d / %d primitives (eps=%.2f, min=%d)",
        len(keep),
        len(primitives),
        cfg.cluster_eps,
        cfg.cluster_min_samples,
    )
    return keep
