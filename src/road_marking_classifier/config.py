from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimplePipelineConfig:
    """Configuration values for the simplified end-to-end pipeline."""

    epsg: Optional[int] = None
    voxel_size_m: float = 0.04
    roi_radius_m: float = 35.0
    bev_resolution_m: float = 0.05
    height_limits_m: Tuple[float, float] = (-0.1, 0.25)
    intensity_near_threshold: float = 0.28
    intensity_far_threshold: float = 0.45
    intensity_far_range_m: float = 35.0
    stop_line_length_m: float = 6.0
    stop_line_distance_m: float = 5.0
    stop_line_angle_tolerance_deg: float = 20.0
    line_cluster_eps_m: float = 1.5
    line_cluster_min_samples: int = 2
    douglas_peucker_m: float = 0.02
    timestamp_format: str = "%Y-%m-%dT%H:%M:%SZ"
