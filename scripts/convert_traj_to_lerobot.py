#!/usr/bin/env python
"""
Convert ManiSkill RecordEpisode trajectories (.h5) into a LeRobot dataset compatible with openpi.

This mirrors openpi's LIBERO conversion example:
  openpi/examples/libero/convert_libero_data_to_lerobot.py

We store the following features:
  - image: HWC uint8
  - wrist_image: HWC uint8
  - state: 1D float32 (any length; openpi can pad)
  - actions: 7D float32 (pd_ee_delta_pose)
  - task: str (used as prompt when `prompt_from_task=True`)

Example:
  conda activate mani_skill
  python scripts/convert_traj_to_lerobot.py \\
    --h5-glob "data/demos/**/*.h5" \\
    --repo-id "local/maniskill_turn_globe_valve" \\
    --image-key "obs/sensors/base_camera/rgb" \\
    --wrist-image-key "obs/sensors/wrist_camera/rgb" \\
    --state-keys "obs/extra/tcp_pose" \\
    --actions-key "actions" \\
    --task "turn the globe valve"
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _h5_get(root: Any, path: str) -> Any:
    cur: Any = root
    for part in path.split("/"):
        cur = cur[part]
    return cur


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")
    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _build_state(obs_root: Any, t: int, state_keys: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for k in state_keys:
        a = np.asarray(_h5_get(obs_root, k))[t]
        parts.append(np.asarray(a, dtype=np.float32).reshape(-1))
    if not parts:
        raise ValueError("state_keys is empty")
    return np.concatenate(parts, axis=0)


def _infer_task_for_h5(h5_path: Path, *, mode: str, fixed_task: str) -> str:
    """
    Infer per-file task string for multi-task training.

    - fixed: always use fixed_task
    - filename: use file stem (without extension)
    - json_env_id: read sibling .json (RecordEpisode metadata) and use env_info.env_id
    """
    if mode == "fixed":
        return fixed_task
    if mode == "filename":
        return h5_path.stem
    if mode == "json_env_id":
        json_path = h5_path.with_suffix(".json")
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
                env_id = meta.get("env_info", {}).get("env_id")
                if isinstance(env_id, str) and env_id:
                    return env_id
            except Exception:
                pass
        # fallback
        return h5_path.stem
    raise ValueError(f"Unknown --task-from: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-dir",
        type=str,
        default=None,
        help="Directory containing .h5 files (optionally nested). If set, we will scan for '**/*.h5'.",
    )
    parser.add_argument(
        "--h5-glob",
        type=str,
        default=None,
        help="Glob for .h5 files (e.g. 'data/demos/**/*.h5'). If both --h5-dir and --h5-glob are given, --h5-glob wins.",
    )
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--robot-type", type=str, default="panda")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-key", type=str, required=True, help="H5 path inside traj group (no leading slash)")
    parser.add_argument("--wrist-image-key", type=str, required=True)
    parser.add_argument(
        "--state-keys",
        type=str,
        nargs="+",
        required=True,
        help="One or more H5 paths inside traj group to concat into a 1D state vector.",
    )
    parser.add_argument("--actions-key", type=str, default="actions", help="H5 dataset path inside traj group.")
    parser.add_argument("--task", type=str, default="do something")
    parser.add_argument(
        "--task-from",
        type=str,
        default="fixed",
        choices=["fixed", "filename", "json_env_id"],
        help="How to populate the LeRobot 'task' field. "
        "'json_env_id' reads the sibling RecordEpisode .json's env_info.env_id (recommended for multi-task).",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    try:
        import h5py
    except Exception as e:  # pragma: no cover
        raise SystemExit("Please `pip install h5py` in your active conda env.") from e

    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Please install LeRobot in your active conda env. "
            "If you already use openpi training, you likely have it. "
            "Otherwise: `pip install lerobot`."
        ) from e

    if args.h5_glob:
        pattern = args.h5_glob
    elif args.h5_dir:
        pattern = str(Path(args.h5_dir).expanduser().resolve() / "**" / "*.h5")
    else:
        raise SystemExit("Need one of: --h5-glob or --h5-dir")

    h5_files = sorted(glob.glob(pattern, recursive=True))
    if not h5_files:
        raise SystemExit(f"No files match: {pattern}")

    # Determine shapes from first file/traj.
    with h5py.File(h5_files[0], "r") as f:
        trajs = sorted([k for k in f.keys() if k.startswith("traj_")])
        if not trajs:
            raise SystemExit(f"No traj_* found in {h5_files[0]}")
        g = f[trajs[0]]
        obs_g = g["obs"]
        img0 = _to_uint8_hwc(np.asarray(_h5_get(g, args.image_key))[0])
        wimg0 = _to_uint8_hwc(np.asarray(_h5_get(g, args.wrist_image_key))[0])
        state0 = _build_state(obs_g, 0, args.state_keys)
        act0 = np.asarray(_h5_get(g, args.actions_key))[0].reshape(-1).astype(np.float32)

    # Create dataset.
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features={
            "image": {"dtype": "image", "shape": tuple(img0.shape), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": tuple(wimg0.shape), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (int(state0.shape[0]),), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (int(act0.shape[0]),), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Write episodes.
    for h5_path in h5_files:
        h5_path = Path(h5_path)
        task_str = _infer_task_for_h5(h5_path, mode=args.task_from, fixed_task=args.task)
        with h5py.File(str(h5_path), "r") as f:
            trajs = sorted([k for k in f.keys() if k.startswith("traj_")])
            for tk in trajs:
                g = f[tk]
                obs_g = g["obs"]
                actions = np.asarray(_h5_get(g, args.actions_key), dtype=np.float32)
                T = int(actions.shape[0])

                imgs = np.asarray(_h5_get(g, args.image_key))
                wimgs = np.asarray(_h5_get(g, args.wrist_image_key))
                # obs groups are typically length T+1; we align with actions length T.
                for t in range(T):
                    dataset.add_frame(
                        {
                            "image": _to_uint8_hwc(imgs[t]),
                            "wrist_image": _to_uint8_hwc(wimgs[t]),
                            "state": _build_state(obs_g, t, args.state_keys),
                            "actions": actions[t],
                            "task": task_str,
                        }
                    )
                dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["maniskill", args.robot_type],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print("Saved LeRobot dataset to:", str(output_path))


if __name__ == "__main__":
    main()


