#!/usr/bin/env python3
"""
Depth Anything 3 batch demo for SAV-style exports.

Layout (example):
  <sequence_root>/00000/rgb.png
  <sequence_root>/00001/rgb.png
  ...

For each five-digit frame folder, runs DA3 on rgb.png, writes da3_depth.npy in that folder,
then writes <parent>/<sequence_name>_da3_depth.mp4 using a consistent depth colormap across
all frames (optionally RGB | depth per frame).

Usage:
  python demo.py --sequence /path/to/sav_000/sav_000001
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

FRAME_DIR_PATTERN = re.compile(r"^\d{5}$")
RGB_CANDIDATES = ("rgb.png", "rgb.jpg", "image.png", "frame.png")


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Same implementation as Any4D ``scripts/demo_sav.py`` (depth colormap + inverse-depth scaling).
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


def list_frame_dirs(sequence_root: Path) -> list[Path]:
    dirs = [
        p
        for p in sequence_root.iterdir()
        if p.is_dir() and FRAME_DIR_PATTERN.match(p.name)
    ]
    return sorted(dirs, key=lambda p: int(p.name))


def find_rgb(frame_dir: Path) -> Path | None:
    for name in RGB_CANDIDATES:
        p = frame_dir / name
        if p.is_file():
            return p
    return None


def global_depth_viz_range_demo_sav(planes: list[np.ndarray]) -> tuple[float, float]:
    """Match ``demo_sav.py`` video block: flatten valid depths, then 2/98 pct on inverse depth."""
    valid_chunks = [d[d > 0] for d in planes if np.any(d > 0)]
    if not valid_chunks:
        return 0.0, 1.0
    all_depths = np.concatenate(valid_chunks)
    global_min = float(np.percentile(1 / all_depths, 2))
    global_max = float(np.percentile(1 / all_depths, 98))

    return global_min, global_max


def load_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_video_frame(
    rgb_path: Path | None,
    depth: np.ndarray,
    global_min: float,
    global_max: float,
    side_by_side: bool,
) -> np.ndarray:
    depth_vis = visualize_depth(depth, depth_min=global_min, depth_max=global_max)
    if not side_by_side or rgb_path is None:
        return depth_vis
    rgb = load_rgb_u8(rgb_path)
    h, w = depth_vis.shape[:2]
    rgb_r = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([rgb_r, depth_vis], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sequence",
        type=Path,
        default=Path("/home/share/public_nas/Dataset/3D_scene/sam4d-test-data/output/weihan/sav/sav_000/sav_000001"),
        help="Path to one sequence folder (contains 00000, 00001, ... subfolders).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Hugging Face model id for DepthAnything3.from_pretrained",
    )
    p.add_argument("--fps", type=float, default=10.0, help="FPS for the output depth video")
    p.add_argument(
        "--side-by-side",
        action="store_true",
        help="Each video frame is RGB | depth colormap (RGB resized to depth resolution).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inference when da3_depth.npy already exists (still included in the video).",
    )
    p.add_argument("--process-res", type=int, default=504, help="DA3 internal processing resolution")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sequence_root = args.sequence.expanduser().resolve()
    if not sequence_root.is_dir():
        print(f"Not a directory: {sequence_root}", file=sys.stderr)
        return 1

    frame_dirs = list_frame_dirs(sequence_root)
    if not frame_dirs:
        print(f"No five-digit frame folders under {sequence_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available; running on CPU (slow).", file=sys.stderr)

    print(f"Loading model {args.model!r} on {device} …")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)

    depths_ordered: list[np.ndarray] = []
    rgb_paths_ordered: list[Path | None] = []

    for frame_dir in frame_dirs:
        rgb_path = find_rgb(frame_dir)
        if rgb_path is None:
            print(f"Skip {frame_dir.name}: no rgb image ({', '.join(RGB_CANDIDATES)})", file=sys.stderr)
            continue

        out_npy = frame_dir / "da3_depth.npy"
        if args.skip_existing and out_npy.is_file():
            depth = np.load(out_npy)
            if depth.ndim != 2:
                print(f"Skip {frame_dir.name}: unexpected da3_depth.npy shape {depth.shape}", file=sys.stderr)
                continue
        else:
            pred = model.inference(
                [str(rgb_path)],
                process_res=args.process_res,
            )
            depth = np.asarray(pred.depth[0], dtype=np.float32)
            np.save(out_npy, depth)
            print(f"{frame_dir.name}: wrote {out_npy}")

        depths_ordered.append(depth)
        rgb_paths_ordered.append(rgb_path if args.side_by_side else None)

    if not depths_ordered:
        print("No frames processed.", file=sys.stderr)
        return 1

    valid_chunks = [d[d > 0] for d in depths_ordered if np.any(d > 0)]
    if not valid_chunks:
        return 0.0, 1.0
    all_depths = np.concatenate(valid_chunks)
    global_min = float(np.percentile(1 / all_depths, 2))
    global_max = float(np.percentile(1 / all_depths, 98))

    frames = [
        build_video_frame(rgb_paths_ordered[i], depths_ordered[i], global_min, global_max, args.side_by_side)
        for i in range(len(depths_ordered))
    ]

    parent = sequence_root.parent
    video_path = sequence_root / f"da3_depth.mp4"
    os.makedirs(parent, exist_ok=True)
    # Same encoding path as Any4D ``demo_sav.py`` (imageio.v3, stacked uint8 RGB frames)
    iio.imwrite(str(video_path), np.stack(frames), fps=args.fps)
    print(f"Wrote {video_path} ({len(frames)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
