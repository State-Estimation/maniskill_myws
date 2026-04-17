#!/usr/bin/env python
"""
π0 / π0.5 inference server (websocket) wrapper for maniskill_myws.

This mirrors the "start model server first" workflow used in test_any_policy/GraspVLA,
but uses openpi's websocket server protocol (openpi-client compatible).

Why this exists:
- Keep openpi as an external dependency (no fork / no edits in openpi repo required)
- Allow setting XLA_FLAGS before importing JAX to work around some GPU toolchain issues

Server usage (GPU machine / policy env):
  conda activate <openpi_env>
  python scripts/pi0/serve.py --config pi05_libero --checkpoint gs://openpi-assets/checkpoints/pi05_libero --port 8000

Client usage (ManiSkill env):
  python scripts/pi0/run_pi0_remote.py --server ws://<ip>:8000 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import logging
import os
from pathlib import Path


def _apply_xla_flags(*, safe: bool, extra: str | None) -> None:
    parts: list[str] = []
    if safe:
        # Conservative: disable autotuning (helps avoid some ptxas crashes during fusion tuning).
        parts.append("--xla_gpu_autotune_level=0")
        # Avoid large upfront GPU reservation, which can cause the process to be killed on smaller GPUs.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if extra:
        parts.append(extra.strip())
    if not parts:
        return
    add = " ".join(p for p in parts if p)
    cur = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = (cur + " " + add).strip() if cur else add


def _resolve_local_checkpoint_dir(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    from openpi.shared import download as _download

    return Path(_download.maybe_download(str(checkpoint))).resolve()


def _infer_asset_id_from_checkpoint(checkpoint: str) -> str | None:
    checkpoint_dir = _resolve_local_checkpoint_dir(checkpoint)
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.exists():
        return None

    candidates = sorted(assets_dir.glob("**/norm_stats.json"))
    if not candidates:
        return None

    asset_ids = sorted({path.parent.relative_to(assets_dir).as_posix() for path in candidates})
    if len(asset_ids) == 1:
        return asset_ids[0]

    logging.warning(
        "Found multiple norm_stats.json files under %s: %s. "
        "Pass --repo-id or --norm-stats explicitly to disambiguate.",
        assets_dir,
        ", ".join(asset_ids),
    )
    return None


def _override_data_identity(cfg, *, repo_id: str | None, asset_id: str | None):
    if repo_id is None and asset_id is None:
        return cfg
    if not hasattr(cfg, "data") or not dataclasses.is_dataclass(cfg.data):
        raise ValueError(f"Config {cfg.name} does not expose a dataclass-like data config for override.")

    updates: dict[str, object] = {}
    if repo_id is not None:
        if not hasattr(cfg.data, "repo_id"):
            raise ValueError(f"Config {cfg.name} does not expose data.repo_id for override.")
        updates["repo_id"] = repo_id
    if asset_id is not None:
        if not hasattr(cfg.data, "assets") or not dataclasses.is_dataclass(cfg.data.assets):
            raise ValueError(f"Config {cfg.name} does not expose data.assets.asset_id for override.")
        updates["assets"] = dataclasses.replace(cfg.data.assets, asset_id=asset_id)

    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, **updates))


def _load_norm_stats_override(path_str: str):
    from openpi.shared import normalize as _normalize

    norm_stats_path = Path(path_str)
    if norm_stats_path.is_file():
        return _normalize.deserialize_json(norm_stats_path.read_text())
    return _normalize.load(norm_stats_path)


def main() -> None:
    faulthandler.enable(all_threads=True)

    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=str, default="pi05_libero", help="openpi config name (e.g. pi0_libero/pi05_libero)")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir (local path or gs://...)")
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional dataset repo_id override. If omitted, serve.py will try to infer it from checkpoint assets.",
    )
    p.add_argument("--default-prompt", type=str, default=None)
    p.add_argument(
        "--norm-stats",
        type=str,
        default=None,
        help="Optional path to a norm stats directory or norm_stats.json file to use instead of checkpoint assets.",
    )
    p.add_argument("--record", action="store_true")
    # XLA safety knobs (must be applied before importing openpi/JAX).
    p.add_argument("--xla-safe", action="store_true", help="set conservative XLA_FLAGS to reduce GPU autotuning issues")
    p.add_argument("--xla-flags", type=str, default=None, help='append to XLA_FLAGS (e.g. "--xla_gpu_autotune_level=0")')

    args = p.parse_args()

    _apply_xla_flags(safe=bool(args.xla_safe), extra=args.xla_flags)
    if os.environ.get("XLA_FLAGS"):
        logging.info("XLA_FLAGS=%s", os.environ["XLA_FLAGS"])
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"):
        logging.info("XLA_PYTHON_CLIENT_PREALLOCATE=%s", os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"])

    # Lazy imports after XLA_FLAGS is set.
    import jax
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as openpi_config

    logging.info("JAX default backend: %s", jax.default_backend())
    logging.info("JAX devices: %s", jax.devices())
    if jax.default_backend() == "cpu":
        logging.warning(
            "JAX is running on CPU only. Large checkpoints like pi0/pi0.5 may be very slow or get killed "
            "by host memory pressure if no GPU backend is available."
        )

    cfg = openpi_config.get_config(args.config)
    inferred_asset_id = _infer_asset_id_from_checkpoint(args.checkpoint)
    repo_id = args.repo_id or inferred_asset_id
    if inferred_asset_id:
        logging.info("Using checkpoint asset identity: %s", inferred_asset_id)
    if repo_id or inferred_asset_id:
        cfg = _override_data_identity(cfg, repo_id=repo_id, asset_id=inferred_asset_id or repo_id)
    logging.info("Resolved data identity: config=%s repo_id=%s", args.config, getattr(cfg.data, "repo_id", None))

    norm_stats = None
    if args.norm_stats:
        norm_stats = _load_norm_stats_override(args.norm_stats)
        logging.info("Loaded norm stats override from %s", args.norm_stats)
    logging.info("Creating policy from checkpoint %s", args.checkpoint)
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )
    logging.info("Policy created successfully")
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    logging.info("Serving %s from %s on ws://%s:%d", args.config, args.checkpoint, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
