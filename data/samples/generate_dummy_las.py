from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np


def generate_dummy_las(out_path: Path, epsg: int) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.epsg = epsg
    points = []
    intensities = []
    for x in np.linspace(0, 100, 500):
        for lane_offset in (-1.5, 0, 1.5):
            points.append([x, lane_offset, 0.0])
            intensities.append(200)
    for y in np.linspace(-3, 3, 80):
        points.append([50.0, y, 0.0])
        intensities.append(220)
    for stripe in range(5):
        y0 = -2 + stripe * 1.5
        for x in np.linspace(45, 55, 50):
            for y in np.linspace(y0, y0 + 0.8, 10):
                points.append([x, y, 0.0])
                intensities.append(210)
    for x in np.linspace(0, 100, 200):
        points.append([x, -4.0, 0.15])
        points.append([x, 4.0, 0.15])
        intensities.extend([180, 180])
    arr = np.array(points)
    las = laspy.LasData(header)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.intensity = np.array(intensities, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_path)
    print(f"Dummy LAS written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsg", type=int, default=6677)
    args = parser.parse_args()
    generate_dummy_las(args.out, args.epsg)


if __name__ == "__main__":
    main()
