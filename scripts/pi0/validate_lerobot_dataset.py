#!/usr/bin/env python
"""
Offline sanity-check for ManiSkill -> LeRobot conversion against openpi's expectations.

Runs in the openpi (uv) environment. It:
  - loads an openpi TrainConfig (e.g. pi05_libero)
  - overrides repo_id / assets_base_dir (so it finds your norm stats)
  - builds openpi dataloader
  - prints a compact summary of shapes/dtypes/ranges
  - optionally runs a single model loss forward pass (slow; will JIT/compile)

Example:
  cd third_party/openpi
  uv run python ../../scripts/pi0/validate_lerobot_dataset.py \
    --openpi-root . \
    --config pi05_libero \
    --repo-id local/maniskill_myws_multitask \
    --assets-base-dir ../../assets_openpi \
    --num-batches 1 \
    --save-images ../../outputs/validate_samples
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


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


def _tree_summary(x: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(x, dict):
        for k, v in x.items():
            out.extend(_tree_summary(v, prefix + f"{k}/"))
        return out
    arr = np.asarray(x)
    stats = ""
    if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
        a = arr.astype(np.float32, copy=False)
        stats = f" min={float(np.nanmin(a)):.3g} max={float(np.nanmax(a)):.3g} mean={float(np.nanmean(a)):.3g}"
    out.append(f"{prefix[:-1]}: shape={tuple(arr.shape)} dtype={arr.dtype}{stats}")
    return out


def _maybe_save_image(path: Path, image_any: Any) -> None:
    try:
        import imageio.v3 as iio
    except Exception:
        return
    arr = np.asarray(image_any)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        path.parent.mkdir(parents=True, exist_ok=True)
        # best-effort conversion to uint8
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                a = np.clip(arr, 0.0, 1.0)
                arr = (a * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        iio.imwrite(path, arr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--openpi-root", type=str, default=None)
    p.add_argument("--config", type=str, default="pi05_libero")
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--assets-base-dir", type=str, required=True, help="Must match where norm stats were written.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-batches", type=int, default=1)
    p.add_argument("--skip-norm-stats", action="store_true", help="Debug only; training expects stats.")
    p.add_argument("--save-images", type=str, default=None, help="Directory to write a few sample images.")
    p.add_argument("--run-model-forward", action="store_true", help="Run one compute_loss forward pass (JAX).")
    args = p.parse_args()

    _ensure_openpi_importable(args.openpi_root)

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    base_cfg = _config.get_config(args.config)
    data_factory = dataclasses.replace(base_cfg.data, repo_id=args.repo_id)
    cfg = dataclasses.replace(base_cfg, assets_base_dir=args.assets_base_dir, data=data_factory)
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=args.batch_size)

    print("Config:", cfg.name)
    print("repo_id:", args.repo_id)
    print("assets_base_dir:", args.assets_base_dir)
    print("action_horizon:", cfg.model.action_horizon)
    print("batch_size:", cfg.batch_size)

    loader = _data_loader.create_data_loader(cfg, shuffle=False, num_batches=args.num_batches, skip_norm_stats=args.skip_norm_stats)
    it = iter(loader)

    for bi in range(args.num_batches):
        obs, actions = next(it)
        print(f"\n=== batch {bi} ===")
        # Observation is an openpi Observation object
        obs_dict = obs.to_dict()
        print("-- observation:")
        for line in _tree_summary(obs_dict):
            print(" ", line)
        print("-- actions:")
        for line in _tree_summary(actions):
            print(" ", line)

        if args.save_images:
            out_dir = Path(args.save_images)
            # common keys: obs.images is a dict of camera views after repack/model transforms
            images = obs_dict.get("images", {})
            if isinstance(images, dict):
                for k, v in images.items():
                    _maybe_save_image(out_dir / f"batch{bi}_{k}.png", v)

    if args.run_model_forward:
        import jax

        # build model and run a single loss forward pass
        rng = jax.random.key(0)
        model = cfg.model.create(rng)
        model.train()
        obs, actions = next(iter(loader))
        loss = np.asarray(model.compute_loss(rng, obs, actions, train=False))
        print("\nmodel.compute_loss output:", dict(shape=loss.shape, dtype=loss.dtype, mean=float(loss.mean())))


if __name__ == "__main__":
    main()

