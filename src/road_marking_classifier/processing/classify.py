from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from shapely.geometry import Point, Polygon

from ..types import ClassifiedPrimitive, PrimitiveGeometry

LAYER_MAP: Dict[str, str] = {
    "road_line": "ROAD_LINE",
    "stop_line": "STOP_LINE",
    "crosswalk": "CROSSWALK",
}


def classify_primitives(
    primitives: Iterable[PrimitiveGeometry],
    stop_line_length_m: float,
    stop_line_distance_m: float,
    stop_line_angle_tolerance_deg: float,
) -> List[ClassifiedPrimitive]:
    """Attach simple semantic classes to geometric primitives."""
    primitive_list = list(primitives)
    crosswalk_info = _build_crosswalk_index(
        [p for p in primitive_list if p.kind == "LWPOLYLINE"]
    )
    classified: List[ClassifiedPrimitive] = []
    for primitive in primitive_list:
        if primitive.kind == "LWPOLYLINE":
            layer = LAYER_MAP["crosswalk"]
            class_name = "crosswalk"
            probability = min(1.0, primitive.score)
        else:
            if _is_stop_line(
                primitive,
                crosswalk_info,
                stop_line_distance_m,
                stop_line_angle_tolerance_deg,
            ):
                layer = LAYER_MAP["stop_line"]
                class_name = "stop_line"
                probability = 0.85
            else:
                length = _primitive_length(primitive)
                if length < stop_line_length_m:
                    layer = LAYER_MAP["stop_line"]
                    class_name = "stop_line"
                    probability = 0.7
                else:
                    layer = LAYER_MAP["road_line"]
                    class_name = "road_line"
                    probability = 0.8
        classified.append(
            ClassifiedPrimitive(
                layer=layer,
                primitive=primitive,
                class_name=class_name,
                probability=probability,
            )
        )
    return classified


def _primitive_length(primitive: PrimitiveGeometry) -> float:
    if "length_m" in primitive.attributes:
        return float(primitive.attributes["length_m"])
    if len(primitive.points) < 2:
        return 0.0
    return float(np.linalg.norm(primitive.points[-1] - primitive.points[0]))


def _build_crosswalk_index(
    crosswalks: Sequence[PrimitiveGeometry],
) -> List[Dict[str, object]]:
    index: List[Dict[str, object]] = []
    for cw in crosswalks:
        polygon = Polygon(cw.points[:, :2])
        if polygon.is_empty:
            continue
        stripe_angle = cw.attributes.get("stripe_angle_deg")
        walkway_angle = cw.attributes.get("walkway_angle_deg")
        if stripe_angle is None or walkway_angle is None:
            stripe_angle, walkway_angle = _fallback_angles(cw.points[:, :2])
        index.append(
            {
                "polygon": polygon,
                "stripe_angle": float(stripe_angle),
                "walkway_angle": float(walkway_angle),
                "center": np.array([polygon.centroid.x, polygon.centroid.y]),
            }
        )
    return index


def _is_stop_line(
    primitive: PrimitiveGeometry,
    crosswalk_info: Sequence[Dict[str, object]],
    distance_thresh_m: float,
    angle_tolerance_deg: float,
) -> bool:
    if not crosswalk_info:
        return False
    line_points = primitive.points[:, :2]
    center = line_points.mean(axis=0)
    direction = line_points[1] - line_points[0]
    line_angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    point = Point(center[0], center[1])
    for info in crosswalk_info:
        polygon: Polygon = info["polygon"]  # type: ignore[assignment]
        if polygon.distance(point) > distance_thresh_m:
            continue
        walkway_angle = info["walkway_angle"]  # type: ignore[assignment]
        if _angle_difference(line_angle, float(walkway_angle)) <= angle_tolerance_deg:
            return True
    return False


def _fallback_angles(points_2d: np.ndarray) -> tuple[float, float]:
    centered = points_2d - points_2d.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    stripe_vec = vh[0]
    stripe_angle = float(np.degrees(np.arctan2(stripe_vec[1], stripe_vec[0])) % 180.0)
    walkway_angle = (stripe_angle + 90.0) % 180.0
    return stripe_angle, walkway_angle


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)
