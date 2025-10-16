from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import ezdxf
import numpy as np

from ..types import ClassifiedPrimitive, LayerName

LOGGER = logging.getLogger(__name__)


def write_dxf(
    primitives: Iterable[ClassifiedPrimitive],
    output_path: Path,
    layer_map: Dict[str, LayerName],
    dp_tolerance: float,
    timestamp_format: str,
) -> None:
    """Serialize primitives into a DXF file with thin metadata."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for layer in set(layer_map.values()).union({"QC_REVIEW"}):
        if layer not in doc.layers:
            doc.layers.add(name=layer)
    timestamp = datetime.utcnow().strftime(timestamp_format)
    for prim in primitives:
        simplified = _douglas_peucker(prim.primitive.points[:, :2], dp_tolerance)
        if len(simplified) < 2:
            continue
        if prim.primitive.kind == "LINE":
            entity = msp.add_line(simplified[0], simplified[-1], dxfattribs={"layer": prim.layer})
        elif prim.primitive.kind == "ARC":
            center, radius, start_angle, end_angle = _fit_arc(simplified)
            entity = msp.add_arc(
                center=center,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                dxfattribs={"layer": prim.layer},
            )
        else:
            entity = msp.add_lwpolyline(
                simplified, dxfattribs={"layer": prim.layer, "closed": True}
            )
        data = {
            "class": prim.class_name,
            "score": round(prim.probability, 3),
            "source_tile": prim.primitive.source_tile,
            "timestamp": timestamp,
        }
        entity.set_xdata("RMC_METADATA", _pack_xdata(data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)
    LOGGER.info("DXF saved: %s", output_path)


def _douglas_peucker(points: np.ndarray, tolerance: float) -> List[List[float]]:
    if len(points) <= 2:
        return points.tolist()

    def recursive(pts):
        start, end = pts[0], pts[-1]
        line_vec = end - start
        if len(pts) <= 2:
            return [start, end]
        dists = np.abs(np.cross(line_vec, start - pts[1:-1])) / (
            np.linalg.norm(line_vec) + 1e-6
        )
        idx = np.argmax(dists)
        if dists[idx] > tolerance:
            left = recursive(pts[: idx + 2])
            right = recursive(pts[idx + 1 :])
            return left[:-1] + right
        return [start, end]

    simplified = recursive(points)
    return [[float(x), float(y)] for x, y in simplified]


def _fit_arc(points: np.ndarray):
    center = points.mean(axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    angles = np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return center.tolist(), radius, float(angles.min()), float(angles.max())


def _pack_xdata(data: Dict[str, object]) -> List:
    return [
        (1000, "class"),
        (1000, str(data["class"])),
        (1000, "score"),
        (1040, float(data["score"])),
        (1000, "source_tile"),
        (1000, str(data["source_tile"])),
        (1000, "timestamp"),
        (1000, str(data["timestamp"])),
    ]
