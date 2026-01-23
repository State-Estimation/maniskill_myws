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
import logging
import os
import sys
from pathlib import Path


def _apply_xla_flags(*, safe: bool, extra: str | None) -> None:
    parts: list[str] = []
    if safe:
        # Conservative: disable autotuning (helps avoid some ptxas crashes during fusion tuning).
        parts.append("--xla_gpu_autotune_level=0")
    if extra:
        parts.append(extra.strip())
    if not parts:
        return
    add = " ".join(p for p in parts if p)
    cur = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = (cur + " " + add).strip() if cur else add


def _ensure_openpi_importable(openpi_root: str | None) -> None:
    """
    Ensure `import openpi` works.

    This script is meant to run inside the *policy* environment (uv/conda) that already has openpi deps installed.
    If openpi itself isn't installed as a package in that env, we can still import it by adding
    <openpi_root>/src to PYTHONPATH.
    """
    try:
        import openpi  # noqa: F401

        return
    except Exception:
        pass

    root = openpi_root or os.environ.get("OPENPI_ROOT")
    if not root:
        raise ModuleNotFoundError(
            "No module named 'openpi'.\n"
            "Start the server inside your openpi uv env (recommended), OR pass --openpi-root /path/to/openpi, "
            "OR set OPENPI_ROOT=/path/to/openpi."
        )
    src = (Path(root).expanduser().resolve() / "src").as_posix()
    if src not in sys.path:
        sys.path.insert(0, src)

    # Re-check.
    import openpi  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=str, default="pi05_libero", help="openpi config name (e.g. pi0_libero/pi05_libero)")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir (local path or gs://...)")
    p.add_argument("--default-prompt", type=str, default=None)
    p.add_argument(
        "--norm-stats",
        type=str,
        default=None,
        help="Optional path to norm_stats.json to use instead of checkpoint assets.",
    )
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--openpi-root",
        type=str,
        default=None,
        help="Optional path to openpi repo (adds <openpi_root>/src to PYTHONPATH if openpi isn't installed). "
        "Alternative: set OPENPI_ROOT.",
    )

    # XLA safety knobs (must be applied before importing openpi/JAX).
    p.add_argument("--xla-safe", action="store_true", help="set conservative XLA_FLAGS to reduce GPU autotuning issues")
    p.add_argument("--xla-flags", type=str, default=None, help='append to XLA_FLAGS (e.g. "--xla_gpu_autotune_level=0")')

    args = p.parse_args()

    _apply_xla_flags(safe=bool(args.xla_safe), extra=args.xla_flags)
    if os.environ.get("XLA_FLAGS"):
        logging.info("XLA_FLAGS=%s", os.environ["XLA_FLAGS"])

    _ensure_openpi_importable(args.openpi_root)

    # Lazy imports after XLA_FLAGS is set.
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as openpi_config

    cfg = openpi_config.get_config(args.config)
    norm_stats = None
    if args.norm_stats:
        from openpi.shared import normalize as _normalize

        norm_stats = _normalize.load(Path(args.norm_stats))
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("Serving %s from %s on ws://%s:%d", args.config, args.checkpoint, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()


