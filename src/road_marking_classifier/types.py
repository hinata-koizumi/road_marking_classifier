from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np


LayerName = Literal["ROAD_LINE", "STOP_LINE", "CROSSWALK", "QC_REVIEW"]
PrimitiveKind = Literal["LINE", "ARC", "LWPOLYLINE"]


@dataclass
class PointCloud:
    """In-memory representation of a point cloud."""

    points: np.ndarray  # shape (N, 3)
    intensities: np.ndarray  # shape (N,)
    epsg: Optional[int]
    metadata: Dict[str, float]
    source_path: Path

    def copy(self) -> "PointCloud":
        return PointCloud(
            points=self.points.copy(),
            intensities=self.intensities.copy(),
            epsg=self.epsg,
            metadata=self.metadata.copy(),
            source_path=self.source_path,
        )


@dataclass
class GroundPlane:
    """Road surface plane model."""

    normal: np.ndarray  # shape (3,)
    d: float
    inliers: np.ndarray  # boolean mask
    rms_error: float


@dataclass
class PrimitiveGeometry:
    """Detected geometric primitive before classification."""

    kind: PrimitiveKind
    points: np.ndarray  # world coordinates
    score: float
    attributes: Dict[str, float]
    bbox: Tuple[np.ndarray, np.ndarray]
    source_tile: str


@dataclass
class ClassifiedPrimitive:
    """Primitive with assigned semantic class."""

    layer: LayerName
    primitive: PrimitiveGeometry
    class_name: str
    probability: float
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


def stack_points(
    primitives: Iterable[PrimitiveGeometry],
) -> np.ndarray:
    return np.vstack([p.points for p in primitives if len(p.points) > 0])
