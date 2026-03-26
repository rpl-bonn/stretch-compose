#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 2D occupancy map from a high-res scene.ply by slicing at a laser height.

- Assumes Stretch lidar plane ~ 6.5 inches above ground (0.1651 m).
- Slices with ±1 inch tolerance (±0.0254 m).
- Works whether your PLY coords are ground-referenced or base_link-referenced.
- Outputs map.pgm + map.yaml for ROS map_server/AMCL.

Usage:
  python make_laser_map.py --ply scene.ply --out map \
    --resolution 0.02 --mode ground   # auto-estimate floor z
  # or if your PLY origin is base_link and you know base_link height above ground:
  python make_laser_map.py --ply scene.ply --out map --mode base_link --base_link_to_floor 0.095
"""

import argparse
import math
import os
from dataclasses import dataclass
import numpy as np
from plyfile import PlyData

try:
  from scipy.ndimage import binary_dilation
  _HAS_SCIPY = True
except Exception:
  _HAS_SCIPY = False
from PIL import Image

INCH = 0.0254

@dataclass
class Params:
  laser_height_m: float = 7.5 * INCH
  band_half_thickness_m: float = 1.0 * INCH
  resolution_m: float = 0.01
  occupancy_hit_threshold: int = 10
  dilate_pixels: int = 1
  padding_m: float = 0.20  # pad map bounds so walls at edges aren't clipped

def load_vertices_xyz(ply_path: str) -> np.ndarray:
  p = PlyData.read(ply_path)
  v = p['vertex']
  # Support common field names
  xs = np.asarray(v['x'], dtype=np.float64)
  ys = np.asarray(v['y'], dtype=np.float64)
  zs = np.asarray(v['z'], dtype=np.float64)
  xyz = np.column_stack([xs, ys, zs])
  return xyz

def estimate_floor_z(xyz: np.ndarray) -> float:
  # Histogram z and pick the lowest prominent mode as floor
  z = xyz[:, 2]
  finite = np.isfinite(z)
  z = z[finite]
  if z.size == 0:
    raise ValueError("No finite Z values in point cloud.")
  bins = np.arange(z.min(), z.max() + 0.01, 0.01)  # 1 cm bins
  hist, edges = np.histogram(z, bins=bins)
  # Require a modest count to avoid tiny spurious minima
  count_thresh = max(100, int(0.001 * z.size))
  candidates = np.where(hist > count_thresh)[0]
  if candidates.size == 0:
    # Fallback: use 2nd percentile as "floor"
    return float(np.percentile(z, 2.0))
  floor_bin = candidates.min()
  floor_z = (edges[floor_bin] + edges[floor_bin + 1]) * 0.5
  return float(floor_z)

def slice_points(xyz: np.ndarray, z_center: float, band_half_thickness: float) -> np.ndarray:
  z = xyz[:, 2]
  mask = (z >= (z_center - band_half_thickness)) & (z <= (z_center + band_half_thickness))
  return xyz[mask, :2]  # return XY only

def grid_from_points(xy: np.ndarray, res: float, padding_m: float):
  if xy.shape[0] == 0:
    raise ValueError("Slice produced 0 points. Check heights/mode parameters.")
  min_xy = xy.min(axis=0) - padding_m
  max_xy = xy.max(axis=0) + padding_m
  size = ((max_xy - min_xy) / res)
  width = int(math.ceil(size[0]))
  height = int(math.ceil(size[1]))
  # ROS maps are in image coords with origin at bottom-left; we’ll store row-major (y-up) then flip for PGM
  occ = np.zeros((height, width), dtype=np.uint8)  # 0=free (white later), we’ll mark occupied as 1
  # Populate occupied
  idx = np.floor((xy - min_xy) / res).astype(int)
  idx[:, 0] = np.clip(idx[:, 0], 0, width - 1)
  idx[:, 1] = np.clip(idx[:, 1], 0, height - 1)
  occ[idx[:, 1], idx[:, 0]] = 1
  return occ, min_xy, (width, height)

def postprocess_occ(occ: np.ndarray, dilate_px: int) -> np.ndarray:
  if dilate_px > 0:
    if _HAS_SCIPY:
      structure = np.ones((1 + 2 * dilate_px, 1 + 2 * dilate_px), dtype=bool)
      occ = binary_dilation(occ.astype(bool), structure=structure).astype(np.uint8)
    else:
      # Simple square dilation without scipy
      k = 2 * dilate_px + 1
      pad = dilate_px
      padded = np.pad(occ, pad, mode='constant')
      out = np.zeros_like(padded)
      for dy in range(k):
        for dx in range(k):
          out = np.maximum(out, padded[dy:dy+occ.shape[0]+2*pad-dilate_px*2, dx:dx+occ.shape[1]+2*pad-dilate_px*2])
      occ = out[pad:-pad, pad:-pad]
  return occ

def save_pgm_yaml(occ01: np.ndarray, res: float, origin_xy: np.ndarray, out_stem: str):
  # Convert to ROS PGM: 0=occupied (black), 254=free (white)
  pgm = np.full_like(occ01, 254, dtype=np.uint8)
  pgm[occ01 > 0] = 0
  # Flip vertically so origin (x=min_x, y=min_y) maps to bottom-left in image coords
  pgm_to_save = np.flipud(pgm)
  im = Image.fromarray(pgm_to_save, mode='L')
  pgm_path = out_stem + ".pgm"
  im.save(pgm_path)
  yaml_path = out_stem + ".yaml"
  # YAML origin is [origin_x, origin_y, yaw]; yaw=0 since grid axes align with world XY
  with open(yaml_path, "w") as f:
    f.write(f"image: {os.path.basename(pgm_path)}\n")
    f.write(f"resolution: {res:.6f}\n")
    f.write(f"origin: [{origin_xy[0]:.6f}, {origin_xy[1]:.6f}, 0.0]\n")
    f.write("occupied_thresh: 0.65\n")
    f.write("free_thresh: 0.196\n")
    f.write("negate: 0\n")
  print(f"Wrote: {pgm_path}, {yaml_path}")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ply", required=True, help="Input scene PLY")
  ap.add_argument("--out", required=True, help="Output map path stem (no extension)")
  ap.add_argument("--resolution", type=float, default=Params.resolution_m)
  ap.add_argument("--band", type=float, default=Params.band_half_thickness_m, help="Half thickness around laser plane [m]")
  ap.add_argument("--laser_height", type=float, default=Params.laser_height_m, help="Laser plane height above ground [m]")
  ap.add_argument("--mode", choices=["ground", "base_link"], default="ground",
                  help="ground: estimate floor z and add laser_height; base_link: z=laser_height assumes origin at ground+base_link_to_floor offset")
  ap.add_argument("--base_link_to_floor", type=float, default=0.0,
                  help="If mode=base_link, how high base_link is above floor [m]")
  ap.add_argument("--dilate_px", type=int, default=Params.dilate_pixels)
  ap.add_argument("--padding", type=float, default=Params.padding_m)
  args = ap.parse_args()

  xyz = load_vertices_xyz(args.ply)

  if args.mode == "ground":
    floor_z = estimate_floor_z(xyz)
    z_center = floor_z + args.laser_height
  else:
    # Origin at base_link: laser plane z = base_link_to_floor + laser_height
    z_center = args.base_link_to_floor + args.laser_height

  xy = slice_points(xyz, z_center, args.band)
  if xy.shape[0] == 0:
    raise SystemExit(f"No points in slice. z_center={z_center:.3f} m, band ±{args.band:.3f} m")

  occ, min_xy, _ = grid_from_points(xy, args.resolution, args.padding)
  occ = postprocess_occ(occ, args.dilate_px)
  save_pgm_yaml(occ, args.resolution, min_xy, args.out)

  print(f"Slice z_center={z_center:.3f} m, points={xy.shape[0]}, resolution={args.resolution} m")

if __name__ == "__main__":
  main()
