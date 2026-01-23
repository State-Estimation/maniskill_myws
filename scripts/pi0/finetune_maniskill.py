#!/usr/bin/env python
"""
Fine-tune openpi π0/π0.5 on a ManiSkill dataset (converted to LeRobot) following the LIBERO recipe.

This script is intended to be run in the **openpi uv environment** (training env), not the ManiSkill env.

Prereqs:
1) Convert your ManiSkill trajectories (.h5) to a LeRobot dataset (repo_id), e.g. with:
   python /home/sisyphus/Projects/maniskill_myws/scripts/convert_traj_to_lerobot.py ...
2) Make sure you can load the LeRobot dataset via that repo_id from within your openpi env.

Usage (recommended, inside openpi repo):
  cd /home/sisyphus/Projects/openpi
  uv run python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/finetune_maniskill.py \\
    --openpi-root /home/sisyphus/Projects/openpi \\
    --config pi05_libero \\
    --repo-id local/maniskill_myws_turn_globe_valve \\
    --exp-name ms_pi05_v1 \\
    --assets-base-dir /home/sisyphus/Projects/maniskill_myws/assets_openpi \\
    --checkpoint-base-dir /home/sisyphus/Projects/maniskill_myws/checkpoints_openpi \\
    --overwrite

Notes:
- We compute norm stats into <assets-base-dir>/<config>/<repo-id>/norm_stats.json, matching openpi's loader.
- Training itself is launched by calling openpi's `scripts/train.py` with tyro overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpi.transforms as transforms


def _ensure_openpi_importable(openpi_root: str | None) -> Path:
    try:
        import openpi  # noqa: F401

        return Path(openpi.__file__).resolve().parents[2]
    except Exception:
        pass

    root = openpi_root or os.environ.get("OPENPI_ROOT")
    if not root:
        raise ModuleNotFoundError(
            "No module named 'openpi'. Run this script inside the openpi uv environment, "
            "or pass --openpi-root /path/to/openpi, or set OPENPI_ROOT."
        )
    root_path = Path(root).expanduser().resolve()
    src = (root_path / "src").as_posix()
    if src not in sys.path:
        sys.path.insert(0, src)
    import openpi  # noqa: F401

    return root_path


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute_norm_stats_for_repo(
    *,
    openpi_root: Path,
    config_name: str,
    repo_id: str,
    assets_base_dir: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_frames: int | None = None,
) -> Path:
    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    base_cfg = _config.get_config(config_name)
    if batch_size is None:
        batch_size = base_cfg.batch_size
    if num_workers is None:
        num_workers = base_cfg.num_workers

    # Override assets dir and dataset repo id (without mutating openpi registry).
    data_factory = base_cfg.data
    if not hasattr(data_factory, "repo_id"):
        raise ValueError(f"Config {config_name} does not have a LeRobot-style repo_id field in its data config.")
    data_factory = dataclasses.replace(data_factory, repo_id=repo_id)
    cfg = dataclasses.replace(base_cfg, assets_base_dir=assets_base_dir, data=data_factory)

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.repo_id is None:
        raise ValueError("data_config.repo_id is None after override; please check your --repo-id.")

    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    if num_batches <= 0:
        raise ValueError(f"Not enough frames to compute stats: len={len(dataset)}, batch_size={batch_size}")

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    ).torch_loader

    keys = ["state", "actions"]
    stats = {k: normalize.RunningStats() for k in keys}
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        for k in keys:
            stats[k].update(np.asarray(batch[k]))

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    out_dir = cfg.assets_dirs / data_config.repo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(out_dir, norm_stats)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--openpi-root", type=str, default=None, help="Path to openpi repo (optional if openpi installed).")

    p.add_argument("--config", type=str, default="pi05_libero", help="openpi config name (pi0_libero / pi05_libero)")
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id (local/... or hf_user/...)")
    p.add_argument("--exp-name", type=str, required=True)

    p.add_argument("--assets-base-dir", type=str, required=True, help="Where to write norm stats for this finetune run.")
    p.add_argument("--checkpoint-base-dir", type=str, required=True, help="Where to write checkpoints for this finetune run.")

    p.add_argument("--batch-size", type=int, default=None, help="Override batch size for training and norm stats.")
    p.add_argument("--num-workers", type=int, default=None, help="Override num_workers for norm stats computation.")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames for norm stats (optional).")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb-enabled", action="store_true", default=False)
    p.add_argument(
        "--only-norm-stats",
        action="store_true",
        help="Compute norm stats and exit without launching training.",
    )
    p.add_argument(
        "--force-norm-stats",
        action="store_true",
        help="Recompute norm stats even if an existing file is found.",
    )

    args = p.parse_args()

    openpi_root = _ensure_openpi_importable(args.openpi_root)

    stats_dir = Path(args.assets_base_dir) / args.config / args.repo_id
    norm_stats_path = stats_dir / "norm_stats.json"
    if norm_stats_path.exists() and not args.force_norm_stats:
        print("✓ Using existing norm stats:", str(norm_stats_path))
    else:
        stats_dir = compute_norm_stats_for_repo(
            openpi_root=openpi_root,
            config_name=args.config,
            repo_id=args.repo_id,
            assets_base_dir=args.assets_base_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_frames=args.max_frames,
        )
        print("✓ Wrote norm stats to:", str(stats_dir))

    if args.only_norm_stats:
        return

    train_py = openpi_root / "scripts" / "train.py"
    cmd = [
        sys.executable,
        str(train_py),
        args.config,
        "--exp-name",
        args.exp_name,
        "--assets_base_dir",
        args.assets_base_dir,
        "--checkpoint_base_dir",
        args.checkpoint_base_dir,
        "--data.repo_id",
        args.repo_id,
    ]
    if args.wandb_enabled:
        cmd += ["--wandb_enabled"]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.resume:
        cmd += ["--resume"]

    print("Launching training:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


