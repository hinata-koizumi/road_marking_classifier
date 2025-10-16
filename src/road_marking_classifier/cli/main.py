from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import SimplePipelineConfig
from ..pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal pipeline: PCD input → color-coded DXF output."
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input PCD path.")
    parser.add_argument("--out", dest="output_path", required=True, help="Output DXF path.")
    parser.add_argument("--epsg", type=int, help="Override EPSG code when missing in source.")
    parser.add_argument("--voxel", type=float, default=0.04, help="Voxel size in meters.")
    parser.add_argument("--roi", type=float, default=35.0, help="ROI radius in meters.")
    parser.add_argument("--bev", type=float, default=0.05, help="BEV resolution in meters.")
    parser.add_argument(
        "--intensity-near",
        dest="intensity_near",
        type=float,
        default=0.28,
        help="Normalized intensity threshold near the sensor.",
    )
    parser.add_argument(
        "--intensity-far",
        dest="intensity_far",
        type=float,
        default=0.45,
        help="Normalized intensity threshold at the far range.",
    )
    parser.add_argument(
        "--intensity-range",
        dest="intensity_range",
        type=float,
        default=35.0,
        help="Range (m) where the far intensity threshold applies.",
    )
    parser.add_argument(
        "--stop-line",
        dest="stop_line_length",
        type=float,
        default=6.0,
        help="Lines shorter than this are labelled STOP_LINE.",
    )
    parser.add_argument(
        "--stop-dist",
        dest="stop_line_distance",
        type=float,
        default=5.0,
        help="Distance (m) from crosswalk to treat a line as STOP_LINE.",
    )
    parser.add_argument(
        "--stop-angle",
        dest="stop_line_angle",
        type=float,
        default=20.0,
        help="Angular tolerance (deg) for STOP_LINE vs crosswalk orientation.",
    )
    parser.add_argument(
        "--dp",
        dest="douglas_peucker",
        type=float,
        default=0.02,
        help="Douglas-Peucker tolerance for DXF simplification.",
    )
    parser.add_argument(
        "--line-cluster-eps",
        dest="line_cluster_eps",
        type=float,
        default=1.5,
        help="DBSCAN epsilon (m) for line clustering noise removal.",
    )
    parser.add_argument(
        "--line-cluster-min",
        dest="line_cluster_min",
        type=int,
        default=2,
        help="DBSCAN min samples for line clustering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = SimplePipelineConfig(
        epsg=args.epsg,
        voxel_size_m=args.voxel,
        roi_radius_m=args.roi,
        bev_resolution_m=args.bev,
        intensity_near_threshold=args.intensity_near,
        intensity_far_threshold=args.intensity_far,
        intensity_far_range_m=args.intensity_range,
        stop_line_length_m=args.stop_line_length,
        stop_line_distance_m=args.stop_line_distance,
        stop_line_angle_tolerance_deg=args.stop_line_angle,
        line_cluster_eps_m=args.line_cluster_eps,
        line_cluster_min_samples=args.line_cluster_min,
        douglas_peucker_m=args.douglas_peucker,
    )
    run_pipeline(Path(args.input_path), Path(args.output_path), config)


if __name__ == "__main__":
    main()
