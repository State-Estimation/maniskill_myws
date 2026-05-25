from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


DEFAULT_OFFLINE_DIR = "dataset/Pi0_rollout_OpenSafeDoor-v2"
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_src_path() -> None:
    src_path = repo_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def prepare_runtime(seed: int, *, register_envs: bool = True):
    ensure_src_path()
    import torch

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if register_envs:
        import maniskill_myws

        maniskill_myws.register()
    return torch


def to_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def as_scalar(x) -> float:
    return float(to_numpy(x).reshape(-1)[0])


def as_done(x) -> bool:
    return bool(to_numpy(x).reshape(-1)[0])


def normalize_render_mode(render_mode: str | None) -> str | None:
    if render_mode is None:
        return None
    if render_mode.lower() in {"none", "null", ""}:
        return None
    return render_mode


def maybe_render(env, render_mode: str | None, step: int, every: int, path_visualizer=None) -> None:
    if render_mode is None:
        return
    if every <= 0 or step % every != 0:
        return
    try:
        if path_visualizer is not None:
            path_visualizer.show_used()
        env.render()
    except Exception as e:
        print(f"render warning at step {step}: {e}")
    finally:
        if path_visualizer is not None:
            path_visualizer.hide_used()


def action_bounds(action_space, action_dim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return (-1.0,) * action_dim, (1.0,) * action_dim
    low_arr = np.asarray(low, dtype=np.float32).reshape(-1)[:action_dim]
    high_arr = np.asarray(high, dtype=np.float32).reshape(-1)[:action_dim]
    if not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(high_arr)):
        return (-1.0,) * action_dim, (1.0,) * action_dim
    return tuple(float(x) for x in low_arr), tuple(float(x) for x in high_arr)


def resolve_torch_device(device_arg: str):
    import torch

    device = torch.device(device_arg if device_arg == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    cuda_info = {}
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        cuda_info = dict(
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            device_count=torch.cuda.device_count(),
            current_device=current_device,
            current_device_name=torch.cuda.get_device_name(current_device),
        )
    return device, cuda_info


def add_offline_replay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offline-h5-dir", type=str, default=DEFAULT_OFFLINE_DIR)
    parser.add_argument("--offline-h5-glob", type=str, default=None)
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--actions-key", type=str, default="actions")
    parser.add_argument("--reward-key", type=str, default="rewards")
    parser.add_argument("--reward-from-success", action="store_true")
    parser.add_argument("--include-failed-offline", action="store_true")
    parser.add_argument("--offline-base-action-mode", choices=["action", "zero"], default="action")
    parser.add_argument("--max-offline-transitions", type=int, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--use-visual-rl", action="store_true")
    parser.add_argument("--rl-image-keys", type=str, nargs="+", default=None)
    parser.add_argument("--rl-image-size", type=int, default=128)


def add_sac_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=10.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--visual-latent-dim", type=int, default=256)


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-enabled", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="maniskill-pld")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-dir", type=str, default=None)


def load_offline_replay(args) -> tuple[Any, list[Path], list[str]]:
    from maniskill_myws.pld.h5_replay import find_h5_files, load_h5_replay

    root = repo_root()
    h5_dir = str((root / args.offline_h5_dir).resolve()) if args.offline_h5_dir else None
    h5_files = find_h5_files(h5_dir=h5_dir, h5_glob=args.offline_h5_glob)
    if not h5_files:
        raise SystemExit(f"No H5 files found under {h5_dir or args.offline_h5_glob}")

    image_keys = args.rl_image_keys or [args.image_key, args.wrist_image_key]
    print(
        "loading offline replay:",
        dict(
            files=len(h5_files),
            visual=bool(args.use_visual_rl),
            image_keys=image_keys if args.use_visual_rl else None,
            image_size=args.rl_image_size if args.use_visual_rl else None,
            max_transitions=args.max_offline_transitions,
        ),
        flush=True,
    )
    offline_data = load_h5_replay(
        h5_files,
        state_keys=args.state_keys,
        actions_key=args.actions_key,
        reward_key=args.reward_key,
        success_only=not args.include_failed_offline,
        reward_from_success=args.reward_from_success,
        base_action_mode=args.offline_base_action_mode,
        max_transitions=args.max_offline_transitions,
        mc_return_gamma=args.gamma,
        image_keys=image_keys if args.use_visual_rl else None,
        image_size=args.rl_image_size if args.use_visual_rl else None,
    )
    print(
        "offline replay:",
        dict(
            files=len(h5_files),
            transitions=offline_data.size,
            state_dim=offline_data.state_dim,
            action_dim=offline_data.action_dim,
            image_shape=offline_data.image_shape,
            mc_return_min=round(float(np.min(offline_data.mc_returns)), 4),
            mc_return_max=round(float(np.max(offline_data.mc_returns)), 4),
        ),
        flush=True,
    )
    if args.use_visual_rl and offline_data.image_shape is None:
        raise SystemExit("--use-visual-rl was set, but no visual observations were loaded")
    return offline_data, h5_files, image_keys


def make_offline_buffer(offline_data, *, batch_size: int):
    from maniskill_myws.pld.replay_buffer import ReplayBuffer

    image_shape = offline_data.image_shape
    capacity = max(int(offline_data.size), int(batch_size))
    buffer = ReplayBuffer(capacity, offline_data.state_dim, offline_data.action_dim, image_shape=image_shape)
    buffer.add_offline_data(offline_data)
    return buffer


def init_wandb(
    args,
    *,
    output_dir: Path,
    offline_files: int,
    offline_transitions: int,
    state_dim: int,
    action_dim: int,
    action_low: tuple[float, ...],
    action_high: tuple[float, ...],
):
    if not args.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as e:
        raise SystemExit(
            "wandb is not installed in the current environment. "
            "Install it or run without --wandb-enabled."
        ) from e

    wandb_dir = (
        Path(args.wandb_dir).expanduser().resolve()
        if args.wandb_dir
        else output_dir.resolve()
    )
    wandb_dir.mkdir(parents=True, exist_ok=True)

    config = dict(vars(args))
    config.update(
        output_dir=str(output_dir.resolve()),
        wandb_dir=str(wandb_dir),
        offline_files=int(offline_files),
        offline_transitions=int(offline_transitions),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        action_low=list(action_low),
        action_high=list(action_high),
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=config,
    )
    wandb.define_metric("offline_update")
    wandb.define_metric("offline/*", step_metric="offline_update")
    wandb.define_metric("env_step")
    wandb.define_metric("train/*", step_metric="env_step")
    wandb.define_metric("episode/*", step_metric="env_step")
    wandb.define_metric("warmup/*", step_metric="env_step")
    wandb.define_metric("checkpoint/*", step_metric="env_step")
    run_id = getattr(run, "id", None)
    if run_id:
        (output_dir / "wandb_run_id.txt").write_text(f"{run_id}\n")
    return wandb


def wandb_log(wandb_mod, payload: dict[str, object]) -> None:
    if wandb_mod is None:
        return
    clean_payload: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            clean_payload[key] = value.item()
        else:
            clean_payload[key] = value
    wandb_mod.log(clean_payload)
