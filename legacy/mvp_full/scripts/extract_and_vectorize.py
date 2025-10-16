from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from core.bev.orthointensity import generate_bev
from core.classify.features import attach_probabilities, compute_features
from core.classify.lgbm import HybridClassifier
from core.common.datatypes import ClassifiedPrimitive
from core.common.logging import configure_logging
from core.detect.crosswalk import detect_crosswalks
from core.detect.curb import detect_curbs
from core.detect.lines import detect_lines_from_bev
from core.geo.reproject import ensure_epsg, load_point_cloud, reproject_points
from core.pre.filters import (
    estimate_ground_plane,
    height_intensity_masks,
    statistical_outlier_removal,
    voxel_downsample,
)
from io.dxf.writer import write_dxf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract road markings and write DXF.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--epsg", type=int, required=False)
    parser.add_argument("--roi", type=float, default=None)
    parser.add_argument("--bev", dest="bev_resolution", type=float, default=None)
    parser.add_argument("--mode", choices=["auto", "precision", "speed"], default="auto")
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--override", type=str, help="JSON string to override config.")
    return parser.parse_args()


def load_config(path: Path, overrides: Optional[str]) -> Dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        cfg.update(json.loads(overrides))
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), args.override)
    configure_logging(cfg["logging"]["level"], Path(cfg["logging"]["file"]))
    logger = logging.getLogger(__name__)
    start_time = time.time()

    pc = load_point_cloud(Path(args.input_path))
    pc = ensure_epsg(pc, cfg["epsg_default"])
    pc = reproject_points(pc, args.epsg or cfg["epsg_default"])
    pc = voxel_downsample(pc, voxel_size=0.04 if args.mode == "speed" else 0.03)
    pc = statistical_outlier_removal(pc)
    ground = estimate_ground_plane(pc, distance_threshold=0.05)
    mask = height_intensity_masks(
        pc,
        ground,
        (cfg["height_limits_m"]["lower"], cfg["height_limits_m"]["upper"]),
        cfg["intensity_thresholds"],
        args.roi or cfg["roi_radius_m"],
    )
    bev = generate_bev(
        pc=pc,
        mask=mask,
        resolution=args.bev_resolution or cfg["bev_resolution_m"],
        roi_radius=args.roi or cfg["roi_radius_m"],
    )
    detection_cfg = yaml.safe_load(Path("configs/detection.yaml").read_text())
    lines = detect_lines_from_bev(
        bev.image,
        bev.resolution,
        detection_cfg["lines"],
        source_tile=Path(args.input_path).stem,
    )
    crosswalks = detect_crosswalks(
        bev.image,
        bev.resolution,
        detection_cfg["crosswalk"],
        source_tile=Path(args.input_path).stem,
    )
    curbs = detect_curbs(
        points=pc.points[mask],
        normals=np.zeros_like(pc.points[mask]),
        heights=pc.points[mask][:, 2],
        config=detection_cfg["curb"],
        source_tile=Path(args.input_path).stem,
    )
    primitives = lines + crosswalks + curbs
    features = compute_features(primitives)
    classifier_cfg = yaml.safe_load(Path("configs/classifier.yaml").read_text())
    classifier = HybridClassifier(classifier_cfg["lgbm"])
    if Path(classifier_cfg["lgbm"]["model_path"]).exists():
        classifier.load(Path(classifier_cfg["lgbm"]["model_path"]))
    else:
        seed_labels = (
            ["road_line"] * len(lines)
            + ["crosswalk"] * len(crosswalks)
            + ["curb"] * len(curbs)
        )
        classifier.load_or_init(features, seed_labels)
    labels, probs = classifier.predict(features)
    layer_map = cfg["output"]["dxf_layer_map"]
    classified: list[ClassifiedPrimitive] = attach_probabilities(primitives, labels, probs, layer_map)
    output_path = Path(args.output_path)
    write_dxf(
        classified,
        output_path,
        layer_map,
        cfg["output"]["douglas_peucker_m"],
        cfg["output"]["timestamp_format"],
    )
    elapsed = time.time() - start_time
    logger.info("Pipeline completed in %.2fs", elapsed)


if __name__ == "__main__":
    main()
