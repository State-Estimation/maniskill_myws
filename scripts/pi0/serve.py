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
  cd third_party/openpi && uv run python ../../scripts/pi0/serve.py --config pi0_maniskill --checkpoint ../../checkpoints_openpi/pi0_maniskill/<exp>/<step> --port 8000

Client usage (ManiSkill env):
  python scripts/pi0/run_pi0_remote.py --server ws://<ip>:8000 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import hashlib
import logging
import os
from pathlib import Path
import stat
import struct
import time


_INFERENCE_SEED_KEY = "__maniskill_inference_seed__"
_INFERENCE_SEED_CAPABILITY = "maniskill_deterministic_inference_seed_v1"
_FROZEN_LATENT_PROTOCOL = "maniskill_frozen_pi0_action_suffix_mean_v1"
_FROZEN_LATENT_KEY = "frozen_pi0_latent"
_FROZEN_LATENT_DIM = 1024
_SAFE_LATENT_PROTOCOL = "safe_pi0_pre_velocity_diff2_horizon2_concat_v1"
_SAFE_LATENT_KEY = "safe_pi0_pre_velocity"
_SAFE_LATENT_DIM = 4 * 1024
_CONTENT_HASH_SCHEMA = b"maniskill_policy_content_tree_sha256_v1\0"
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True, slots=True)
class _ContentEntry:
    relative_path: bytes
    kind: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclasses.dataclass(frozen=True, slots=True)
class _ContentIdentity:
    path: Path
    sha256: str
    file_count: int
    total_bytes: int
    snapshot: tuple[_ContentEntry, ...]


def _entry_from_stat(relative_path: bytes, kind: str, value: os.stat_result) -> _ContentEntry:
    return _ContentEntry(
        relative_path=relative_path,
        kind=kind,
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _snapshot_content_path(path: Path) -> tuple[_ContentEntry, ...]:
    """Snapshot regular files and directories without following nested symlinks."""

    try:
        root_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"Policy content path is missing or unreadable: {path}") from error

    if stat.S_ISREG(root_stat.st_mode):
        return (_entry_from_stat(os.fsencode(path.name), "file", root_stat),)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"Policy content path must be a regular file or directory: {path}")

    entries = [_entry_from_stat(b"", "directory", root_stat)]
    pending = [(path, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: os.fsencode(item.name))
        except OSError as error:
            raise RuntimeError(
                f"Policy content directory changed or became unreadable: {directory}"
            ) from error

        child_directories: list[tuple[Path, Path]] = []
        for child in children:
            child_path = Path(child.path)
            relative_path = relative_directory / child.name
            relative_bytes = os.fsencode(relative_path.as_posix())
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError(
                    f"Policy content entry changed or became unreadable: {child_path}"
                ) from error
            if stat.S_ISREG(child_stat.st_mode):
                entries.append(_entry_from_stat(relative_bytes, "file", child_stat))
            elif stat.S_ISDIR(child_stat.st_mode):
                entries.append(_entry_from_stat(relative_bytes, "directory", child_stat))
                child_directories.append((child_path, relative_path))
            else:
                raise RuntimeError(
                    "Policy content trees may contain only regular files and directories; "
                    f"refusing symlink or special entry: {child_path}"
                )
        pending.extend(reversed(child_directories))

    return tuple(sorted(entries, key=lambda entry: (entry.relative_path, entry.kind)))


def _same_file_identity(left: _ContentEntry, right: _ContentEntry) -> bool:
    return (
        left.kind == right.kind
        and left.device == right.device
        and left.inode == right.inode
        and left.mode == right.mode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
        and left.ctime_ns == right.ctime_ns
    )


def _hash_regular_file(
    path: Path,
    expected: _ContentEntry,
    digest: "hashlib._Hash",
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"Policy content file changed or became unreadable: {path}") from error

    try:
        before = _entry_from_stat(expected.relative_path, "file", os.fstat(descriptor))
        if not _same_file_identity(expected, before):
            raise RuntimeError(f"Policy content file changed before hashing: {path}")
        bytes_read = 0
        while True:
            block = os.read(descriptor, _HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
            bytes_read += len(block)
        after = _entry_from_stat(expected.relative_path, "file", os.fstat(descriptor))
        if bytes_read != expected.size or not _same_file_identity(before, after):
            raise RuntimeError(f"Policy content file changed while hashing: {path}")
    finally:
        os.close(descriptor)


def _hash_content_path(path: str | Path) -> _ContentIdentity:
    """Hash one file or a directory tree using unambiguous relative-path framing."""

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Policy content path does not exist: {path}") from error
    snapshot = _snapshot_content_path(resolved)
    digest = hashlib.sha256()
    digest.update(_CONTENT_HASH_SCHEMA)
    digest.update(struct.pack(">Q", len(snapshot)))
    file_count = 0
    total_bytes = 0
    for entry in snapshot:
        relative_path = entry.relative_path
        digest.update(b"F" if entry.kind == "file" else b"D")
        digest.update(struct.pack(">Q", len(relative_path)))
        digest.update(relative_path)
        if entry.kind == "file":
            digest.update(struct.pack(">Q", entry.size))
            file_path = resolved if len(snapshot) == 1 else resolved / os.fsdecode(relative_path)
            _hash_regular_file(file_path, entry, digest)
            file_count += 1
            total_bytes += entry.size

    current_snapshot = _snapshot_content_path(resolved)
    if current_snapshot != snapshot:
        raise RuntimeError(f"Policy content tree changed while hashing: {resolved}")
    return _ContentIdentity(
        path=resolved,
        sha256=digest.hexdigest(),
        file_count=file_count,
        total_bytes=total_bytes,
        snapshot=snapshot,
    )


def _assert_content_unchanged(identity: _ContentIdentity) -> None:
    if _snapshot_content_path(identity.path) != identity.snapshot:
        raise RuntimeError(
            f"Policy content changed after hashing and before server startup: {identity.path}"
        )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _content_identity_metadata(
    checkpoint_identity: _ContentIdentity,
    external_norm_stats_identity: _ContentIdentity | None,
) -> dict[str, str]:
    result = {
        "resolved_checkpoint": str(checkpoint_identity.path),
        "checkpoint_content_sha256": checkpoint_identity.sha256,
    }
    if external_norm_stats_identity is not None:
        result["norm_stats_content_sha256"] = external_norm_stats_identity.sha256
    return result


def _request_seeded_policy(policy, jax):
    """Apply a request-local JAX seed without advancing shared server RNG state."""

    class RequestSeededPolicy:
        @property
        def metadata(self):
            return policy.metadata

        def infer(self, obs: dict):
            request = dict(obs)
            inference_seed = request.pop(_INFERENCE_SEED_KEY, None)
            if inference_seed is None:
                return policy.infer(request)
            if not hasattr(policy, "_rng"):
                raise RuntimeError(
                    "Deterministic request seeds currently require a JAX OpenPI policy"
                )
            previous_rng = policy._rng
            policy._rng = jax.random.key(int(inference_seed))
            try:
                return policy.infer(request)
            finally:
                policy._rng = previous_rng

    return RequestSeededPolicy()


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
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        try:
            resolved = checkpoint_path.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(f"Checkpoint path cannot be resolved: {checkpoint}") from error
    else:
        from openpi.shared import download as _download

        downloaded = _download.maybe_download(str(checkpoint))
        try:
            resolved = Path(downloaded).expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeError(
                f"Downloaded checkpoint path cannot be resolved: {downloaded}"
            ) from error

    if not resolved.is_dir():
        raise RuntimeError(f"Checkpoint must resolve to a directory: {resolved}")
    return resolved


def _infer_asset_id_from_checkpoint(checkpoint_dir: Path) -> str | None:
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


def _load_norm_stats_override(path: Path):
    from openpi.shared import normalize as _normalize

    if path.is_file():
        return _normalize.deserialize_json(path.read_text(encoding="utf-8"))
    return _normalize.load(path)


def main() -> None:
    faulthandler.enable(all_threads=True)

    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--config",
        type=str,
        default="pi0_maniskill",
        help=(
            "openpi config name (e.g. pi0_maniskill/pi05_maniskill/"
            "pi05_maniskill_quantile/pi0_maniskill_legacy)"
        ),
    )
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
    latent_group = p.add_mutually_exclusive_group()
    latent_group.add_argument(
        "--frozen-action-latent",
        action="store_true",
        help=(
            "return the frozen final-denoise Pi0 action-suffix hidden mean "
            "alongside each action chunk"
        ),
    )
    latent_group.add_argument(
        "--safe-pre-velocity-latent",
        action="store_true",
        help=(
            "return SAFE's pre-velocity action tokens using official concat-2 "
            "selection over diffusion steps and action horizon (4096 floats)"
        ),
    )
    # XLA safety knobs (must be applied before importing openpi/JAX).
    p.add_argument("--xla-safe", action="store_true", help="set conservative XLA_FLAGS to reduce GPU autotuning issues")
    p.add_argument("--xla-flags", type=str, default=None, help='append to XLA_FLAGS (e.g. "--xla_gpu_autotune_level=0")')

    args = p.parse_args()

    _apply_xla_flags(safe=bool(args.xla_safe), extra=args.xla_flags)
    if os.environ.get("XLA_FLAGS"):
        logging.info("XLA_FLAGS=%s", os.environ["XLA_FLAGS"])
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"):
        logging.info("XLA_PYTHON_CLIENT_PREALLOCATE=%s", os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"])

    checkpoint_dir = _resolve_local_checkpoint_dir(args.checkpoint)
    logging.info(
        "Hashing all checkpoint files at %s once for policy identity; this startup "
        "read includes assets/norm stats and may read about 12 GB for a pi0 checkpoint",
        checkpoint_dir,
    )
    checkpoint_hash_started = time.monotonic()
    checkpoint_identity = _hash_content_path(checkpoint_dir)
    logging.info(
        "Checkpoint content identity: sha256=%s files=%d bytes=%d elapsed_s=%.2f",
        checkpoint_identity.sha256,
        checkpoint_identity.file_count,
        checkpoint_identity.total_bytes,
        time.monotonic() - checkpoint_hash_started,
    )

    resolved_norm_stats: Path | None = None
    external_norm_stats_identity: _ContentIdentity | None = None
    if args.norm_stats:
        try:
            resolved_norm_stats = Path(args.norm_stats).expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeError(
                f"Norm stats override does not exist or cannot be resolved: {args.norm_stats}"
            ) from error
        if not _is_within(resolved_norm_stats, checkpoint_dir):
            logging.info(
                "Hashing external norm stats override at %s for policy identity",
                resolved_norm_stats,
            )
            norm_hash_started = time.monotonic()
            external_norm_stats_identity = _hash_content_path(resolved_norm_stats)
            logging.info(
                "External norm stats content identity: sha256=%s files=%d bytes=%d "
                "elapsed_s=%.2f",
                external_norm_stats_identity.sha256,
                external_norm_stats_identity.file_count,
                external_norm_stats_identity.total_bytes,
                time.monotonic() - norm_hash_started,
            )

    # Lazy imports after XLA_FLAGS is set.
    import jax
    from openpi.models import gemma as _gemma
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
    frozen_action_latent = bool(args.frozen_action_latent)
    safe_pre_velocity_latent = bool(args.safe_pre_velocity_latent)
    if frozen_action_latent or safe_pre_velocity_latent:
        action_expert_variant = getattr(cfg.model, "action_expert_variant", None)
        if action_expert_variant is None:
            raise ValueError(
                "Latent output requires a Pi0 model with an action expert"
            )
        actual_latent_dim = _gemma.get_config(action_expert_variant).width
        if actual_latent_dim != _FROZEN_LATENT_DIM:
            raise ValueError(
                "Frozen action latent protocol dimension mismatch: "
                f"action expert {action_expert_variant!r} has width "
                f"{actual_latent_dim}, expected {_FROZEN_LATENT_DIM}"
            )
    inferred_asset_id = _infer_asset_id_from_checkpoint(checkpoint_dir)
    repo_id = args.repo_id or inferred_asset_id
    if inferred_asset_id:
        logging.info("Using checkpoint asset identity: %s", inferred_asset_id)
    if repo_id or inferred_asset_id:
        cfg = _override_data_identity(cfg, repo_id=repo_id, asset_id=inferred_asset_id or repo_id)
    logging.info("Resolved data identity: config=%s repo_id=%s", args.config, getattr(cfg.data, "repo_id", None))

    norm_stats = None
    if resolved_norm_stats is not None:
        norm_stats = _load_norm_stats_override(resolved_norm_stats)
        logging.info("Loaded norm stats override from %s", resolved_norm_stats)
    logging.info("Creating policy from resolved checkpoint %s", checkpoint_dir)
    policy = policy_config.create_trained_policy(
        cfg,
        str(checkpoint_dir),
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        return_frozen_action_latent=frozen_action_latent,
        return_safe_pre_velocity=safe_pre_velocity_latent,
    )
    _assert_content_unchanged(checkpoint_identity)
    if external_norm_stats_identity is not None:
        _assert_content_unchanged(external_norm_stats_identity)
    logging.info("Policy created successfully")
    policy_metadata = dict(policy.metadata)
    maniskill_policy_identity = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "repo_id": repo_id,
        "norm_stats_override": args.norm_stats,
        "resolved_norm_stats_override": (
            str(resolved_norm_stats) if resolved_norm_stats is not None else None
        ),
        "default_prompt": args.default_prompt,
        **_content_identity_metadata(
            checkpoint_identity,
            external_norm_stats_identity,
        ),
    }
    policy_metadata["maniskill_policy_identity"] = maniskill_policy_identity
    if hasattr(policy, "_rng"):
        policy_metadata["inference_seed_protocol"] = _INFERENCE_SEED_CAPABILITY
    if frozen_action_latent:
        policy_metadata.update(
            frozen_latent_protocol=_FROZEN_LATENT_PROTOCOL,
            frozen_latent_key=_FROZEN_LATENT_KEY,
            frozen_latent_shape=[_FROZEN_LATENT_DIM],
            frozen_latent_dtype="float32",
            frozen_latent_source="pi0_final_denoise_action_suffix_tokens",
            frozen_latent_pooling="mean_over_action_horizon",
        )
    if safe_pre_velocity_latent:
        safe_pred_horizon = int(getattr(cfg.model, "action_horizon", 0))
        if safe_pred_horizon < 2:
            raise ValueError(
                "SAFE concat-2 requires a Pi0 action horizon of at least two"
            )
        policy_metadata.update(
            safe_latent_protocol=_SAFE_LATENT_PROTOCOL,
            safe_latent_key=_SAFE_LATENT_KEY,
            safe_latent_shape=[_SAFE_LATENT_DIM],
            safe_latent_dtype="float32",
            safe_latent_source="pi0_action_expert_pre_velocity_tokens",
            safe_latent_diffusion_selection="concat-2_first_last",
            safe_latent_horizon_selection="concat-2_first_last",
            safe_latent_pooling="none",
            safe_latent_pred_horizon=safe_pred_horizon,
        )
    policy = _request_seeded_policy(policy, jax)
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    logging.info(
        "Serving %s from %s on ws://%s:%d",
        args.config,
        checkpoint_dir,
        args.host,
        args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
