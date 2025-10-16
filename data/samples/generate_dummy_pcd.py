from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def generate_dummy_pcd(out_path: Path) -> None:
    """Create a synthetic road marking scene and save it as PCD."""
    points = []
    colors = []

    # Road surface (dark gray)
    for x in np.linspace(0.0, 100.0, 240):
        for y in np.linspace(-6.0, 6.0, 120):
            z = np.random.normal(0.0, 0.005)
            points.append([x, y, z])
            colors.append([0.25, 0.25, 0.25])

    # Lane markings (white)
    for lane_offset in (-2.5, 0.0, 2.5):
        for x in np.linspace(0.0, 100.0, 200):
            if int(x) % 10 < 5:
                points.append([x, lane_offset, 0.01])
                colors.append([0.95, 0.95, 0.95])

    # Stop line
    for y in np.linspace(-3.0, 3.0, 80):
        points.append([70.0, y, 0.01])
        colors.append([0.95, 0.95, 0.95])

    # Crosswalk stripes
    for stripe_idx in range(5):
        stripe_y0 = -2.0 + stripe_idx * 1.5
        for x in np.linspace(60.0, 65.0, 120):
            for y in np.linspace(stripe_y0, stripe_y0 + 1.0, 40):
                points.append([x, y, 0.01])
                colors.append([0.95, 0.95, 0.95])

    # Curbs (yellow)
    for x in np.linspace(0.0, 100.0, 200):
        points.append([x, -6.0, 0.12])
        colors.append([0.9, 0.8, 0.1])
        points.append([x, 6.0, 0.12])
        colors.append([0.9, 0.8, 0.1])

    arr_points = np.asarray(points, dtype=np.float64)
    arr_colors = np.asarray(colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr_points)
    pcd.colors = o3d.utility.Vector3dVector(arr_colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Dummy PCD written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate_dummy_pcd(args.out)


if __name__ == "__main__":
    main()
