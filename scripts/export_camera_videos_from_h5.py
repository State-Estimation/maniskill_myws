#!/usr/bin/env python
"""
Export per-trajectory camera videos from a ManiSkill trajectory H5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def _normalize_path(path: str) -> str:
    return path.lstrip("/")


def _h5_get(root, path: str):
    cur = root
    for part in _normalize_path(path).split("/"):
        cur = cur[part]
    return cur


def _to_uint8_hwc(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"Expected frames with 4 dims [T,H,W,C] or [T,C,H,W], got {arr.shape}")
    if arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description="Export one mp4 per trajectory from a camera stream stored in H5.")
    p.add_argument("--h5-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--camera-key", type=str, default="obs/sensor_data/base_camera/rgb")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--suffix", type=str, default="base_view")
    args = p.parse_args()

    h5_path = Path(args.h5_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        traj_ids = sorted(k for k in f.keys() if k.startswith("traj_"))
        if not traj_ids:
            raise SystemExit(f"No traj_* groups found in {h5_path}")

        print(f"H5: {h5_path}")
        print(f"Trajectories: {len(traj_ids)}")
        print(f"Camera key: {args.camera_key}")
        print(f"Output dir: {out_dir}")

        for traj_id in traj_ids:
            frames = _to_uint8_hwc(np.asarray(_h5_get(f[traj_id], args.camera_key)))
            out_path = out_dir / f"{traj_id}.{args.suffix}.mp4"
            writer = imageio.get_writer(out_path.as_posix(), fps=args.fps)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            print(f"saved {out_path.name} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
