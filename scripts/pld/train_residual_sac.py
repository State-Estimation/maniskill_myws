#!/usr/bin/env python
"""
Train a lightweight PLD residual SAC specialist for a ManiSkill task.

This is Stage 1 of PLD in this workspace:
  - load successful replay trajectories as B_offline
  - freeze/use a base policy for online actions
  - train a state-only residual Gaussian actor with SAC

For a faithful PLD run, use --base-policy remote_openpi and point --server at a
running pi0/openpi policy server. For quick code-path checks, --base-policy zero
can train from the offline replay and collect zero-prior online data.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_OFFLINE_DIR = "dataset/Replayed_traj_data_openSafeDoor2"
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _as_scalar(x) -> float:
    arr = _to_numpy(x)
    return float(arr.reshape(-1)[0])


def _as_done(x) -> bool:
    arr = _to_numpy(x)
    return bool(arr.reshape(-1)[0])


def _normalize_render_mode(render_mode: str | None) -> str | None:
    if render_mode is None:
        return None
    if render_mode.lower() in {"none", "null", ""}:
        return None
    return render_mode


def _maybe_render(env, render_mode: str | None, step: int, every: int, path_visualizer=None) -> None:
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


def _action_bounds(action_space, action_dim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return (-1.0,) * action_dim, (1.0,) * action_dim
    low_arr = np.asarray(low, dtype=np.float32).reshape(-1)[:action_dim]
    high_arr = np.asarray(high, dtype=np.float32).reshape(-1)[:action_dim]
    if not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(high_arr)):
        return (-1.0,) * action_dim, (1.0,) * action_dim
    return tuple(float(x) for x in low_arr), tuple(float(x) for x in high_arr)


def _init_wandb(
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


def _wandb_log(wandb_mod, payload: dict[str, object]) -> None:
    if wandb_mod is None:
        return
    clean_payload: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            clean_payload[key] = value.item()
        else:
            clean_payload[key] = value
    wandb_mod.log(clean_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="state_dict")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Optional ManiSkill render mode, e.g. 'human' for realtime viewing.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Call env.render() every N env steps when --render-mode is set.",
    )
    parser.add_argument(
        "--visualize-tcp-path",
        action="store_true",
        help="Draw TCP path markers in the viewer while rendering.",
    )
    parser.add_argument(
        "--path-every",
        type=int,
        default=2,
        help="Add one TCP path marker every N env steps when visualization is enabled.",
    )
    parser.add_argument("--path-max-points", type=int, default=500)
    parser.add_argument("--path-radius", type=float, default=0.008)
    parser.add_argument("--tcp-pose-key", type=str, default="extra/tcp_pose")
    parser.add_argument(
        "--base-chunk-max-actions",
        type=int,
        default=16,
        help="Maximum number of base-policy chunk actions to visualize in blue.",
    )
    parser.add_argument(
        "--base-chunk-position-scale",
        type=float,
        default=0.1,
        help=(
            "Meters per normalized action unit when projecting the base policy's "
            "future action chunk into TCP positions."
        ),
    )
    parser.add_argument(
        "--base-path-color",
        type=str,
        default="0.05,0.35,1.0,1.0",
        help="RGBA color for base-policy path markers.",
    )
    parser.add_argument(
        "--residual-path-color",
        type=str,
        default="1.0,0.28,0.02,1.0",
        help="RGBA color for residual-policy path markers.",
    )
    parser.add_argument("--offline-h5-dir", type=str, default=DEFAULT_OFFLINE_DIR)
    parser.add_argument("--offline-h5-glob", type=str, default=None)
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--actions-key", type=str, default="actions")
    parser.add_argument("--reward-key", type=str, default="rewards")
    parser.add_argument("--reward-from-success", action="store_true")
    parser.add_argument("--include-failed-offline", action="store_true")
    parser.add_argument("--offline-base-action-mode", choices=["action", "zero"], default="action")
    parser.add_argument("--max-offline-transitions", type=int, default=None)

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="zero")
    parser.add_argument("--server", type=str, default=None, help="ws://host:port for remote_openpi")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument(
        "--use-visual-rl",
        action="store_true",
        help="Train the residual RL actor/critic from RGB cameras plus proprio state.",
    )
    parser.add_argument(
        "--rl-image-keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Observation image keys for residual RL. Defaults to --image-key and "
            "--wrist-image-key."
        ),
    )
    parser.add_argument(
        "--rl-image-size",
        type=int,
        default=128,
        help="Square image size stored in visual replay when --use-visual-rl is set.",
    )
    parser.add_argument("--visual-latent-dim", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=250_000)
    parser.add_argument(
        "--offline-pretrain-method",
        choices=["sac", "calql", "none"],
        default="sac",
        help=(
            "Offline warm-start method before online residual RL. "
            "'calql' updates only the critics with Cal-QL and leaves the "
            "residual actor randomly initialized."
        ),
    )
    parser.add_argument("--offline-pretrain-updates", type=int, default=1_000)
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=250_000)
    parser.add_argument("--offline-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-episodes", type=int, default=100)
    parser.add_argument(
        "--warmup-buffer-path",
        type=str,
        default=None,
        help=(
            "Optional .npz snapshot of the online buffer after base-policy warmup. "
            "If the file exists, it is loaded and warmup is skipped; otherwise "
            "the warmup buffer is saved there once warmup episodes complete."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=25_000)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help=(
            "SAC target entropy. Defaults to a scale-aware value: "
            "-action_dim + action_dim * log(action_scale)."
        ),
    )
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=10.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--cql-alpha", type=float, default=0.0)
    parser.add_argument("--calql-alpha", type=float, default=5.0)
    parser.add_argument("--calql-n-actions", type=int, default=10)
    parser.add_argument("--calql-temp", type=float, default=1.0)
    parser.add_argument(
        "--calql-no-importance-sample",
        action="store_true",
        help="Disable Cal-QL's importance-sampled CQL objective.",
    )
    parser.add_argument(
        "--calql-backup-entropy",
        action="store_true",
        help="Include SAC entropy in Cal-QL critic pretraining TD backups.",
    )
    parser.add_argument(
        "--otf-backup-actions",
        type=int,
        default=0,
        help=(
            "Number of residual-policy candidate actions to sample for OTF TD backup. "
            "0 disables OTF backup; with the default base candidate enabled, N uses N+1 "
            "total candidates."
        ),
    )
    parser.add_argument(
        "--otf-rollout-actions",
        type=int,
        default=0,
        help=(
            "Number of residual-policy candidate actions to sample for OTF online rollout. "
            "0 keeps the standard stochastic residual policy; with the default base "
            "candidate enabled, N uses N+1 total candidates."
        ),
    )
    parser.add_argument(
        "--otf-no-base-candidate",
        action="store_true",
        help="Do not include the unedited base-policy action as an OTF candidate.",
    )
    parser.add_argument(
        "--otf-backup-entropy",
        action="store_true",
        help=(
            "Use SAC soft values Q - alpha log pi for OTF TD backup. By default "
            "OTF follows the paper pseudocode and backs up the selected hard Q."
        ),
    )

    parser.add_argument("--output-dir", type=str, default="outputs/pld/OpenSafeDoor-v2")
    parser.add_argument("--checkpoint-name", type=str, default="residual_sac.pt")
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
        help="wandb logging mode when --wandb-enabled is set.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default=None,
        help="Optional directory for local wandb files. Defaults to --output-dir.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.pld.h5_replay import find_h5_files, load_h5_replay
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.path_visualizer import TCPPathVisualizer, parse_rgba
    from maniskill_myws.pld.replay_buffer import ReplayBuffer, sample_offline_online
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    from maniskill_myws.pld.state import ImageAdapter, StateAdapter
    from maniskill_myws.task_prompts import get_task_prompt

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maniskill_myws.register()
    rl_image_keys = args.rl_image_keys or [args.image_key, args.wrist_image_key]

    offline_dir = str((repo_root / args.offline_h5_dir).resolve())
    h5_files = find_h5_files(h5_dir=offline_dir, h5_glob=args.offline_h5_glob)
    if not h5_files:
        raise SystemExit(f"No H5 files found under {offline_dir}")
    print(
        "loading offline replay:",
        dict(
            files=len(h5_files),
            visual=bool(args.use_visual_rl),
            image_keys=rl_image_keys if args.use_visual_rl else None,
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
        image_keys=rl_image_keys if args.use_visual_rl else None,
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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    action_dim = offline_data.action_dim
    env = None
    if args.total_env_steps > 0:
        render_mode = _normalize_render_mode(args.render_mode)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            render_mode=render_mode,
        )
        low, high = _action_bounds(env.action_space, action_dim)
    else:
        render_mode = None
        low, high = (-1.0,) * action_dim, (1.0,) * action_dim

    image_shape = offline_data.image_shape if args.use_visual_rl else None
    offline_capacity = (
        max(int(offline_data.size), int(args.batch_size))
        if args.use_visual_rl
        else int(args.buffer_capacity)
    )
    offline_buffer = ReplayBuffer(
        offline_capacity, offline_data.state_dim, action_dim, image_shape=image_shape
    )
    offline_buffer.add_offline_data(offline_data)
    online_buffer = ReplayBuffer(
        args.buffer_capacity, offline_data.state_dim, action_dim, image_shape=image_shape
    )
    warmup_buffer_path = (
        Path(args.warmup_buffer_path).expanduser().resolve()
        if args.warmup_buffer_path
        else None
    )
    loaded_warmup_episode_idx = 0
    loaded_warmup_env_steps = 0
    warmup_buffer_saved = warmup_buffer_path is None
    if warmup_buffer_path is not None and warmup_buffer_path.exists():
        warmup_meta = online_buffer.load(warmup_buffer_path)
        loaded_warmup_episode_idx = int(
            warmup_meta.get("episode_idx", warmup_meta.get("warmup_episodes", args.warmup_episodes))
        )
        loaded_warmup_env_steps = int(
            warmup_meta.get("env_steps", len(online_buffer))
        )
        warmup_buffer_saved = True
        print(
            "loaded warmup buffer:",
            dict(
                path=str(warmup_buffer_path),
                transitions=len(online_buffer),
                episode_idx=loaded_warmup_episode_idx,
                env_steps=loaded_warmup_env_steps,
            ),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name
    wandb_run = _init_wandb(
        args,
        output_dir=output_dir,
        offline_files=len(h5_files),
        offline_transitions=offline_data.size,
        state_dim=offline_data.state_dim,
        action_dim=action_dim,
        action_low=low,
        action_high=high,
    )
    _wandb_log(
        wandb_run,
        dict(
            offline_update=0,
            **{
                "offline/files": len(h5_files),
                "offline/transitions": offline_data.size,
                "offline/state_dim": offline_data.state_dim,
                "offline/action_dim": action_dim,
                "offline/mc_return_min": float(np.min(offline_data.mc_returns)),
                "offline/mc_return_max": float(np.max(offline_data.mc_returns)),
            },
        ),
    )

    agent = ResidualSAC(
        SACConfig(
            state_dim=offline_data.state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            actor_lr=args.lr,
            critic_lr=args.lr,
            alpha_lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            action_scale=args.action_scale,
            target_entropy=args.target_entropy,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            actor_update_interval=2,
            grad_clip_norm=args.grad_clip_norm,
            cql_alpha=args.cql_alpha,
            calql_alpha=args.calql_alpha,
            calql_n_actions=args.calql_n_actions,
            calql_temp=args.calql_temp,
            calql_importance_sample=not args.calql_no_importance_sample,
            calql_backup_entropy=args.calql_backup_entropy,
            otf_backup_actions=args.otf_backup_actions,
            otf_include_base_action=not args.otf_no_base_candidate,
            otf_backup_entropy=args.otf_backup_entropy,
            visual_encoder="resnet10" if args.use_visual_rl else "none",
            image_shape=image_shape,
            visual_latent_dim=args.visual_latent_dim,
            action_low=low,
            action_high=high,
        ),
        device=device,
    )

    metrics: dict[str, float] = {}
    if args.offline_pretrain_method != "none":
        for i in range(args.offline_pretrain_updates):
            batch = offline_buffer.sample(args.batch_size)
            if args.offline_pretrain_method == "calql":
                metrics = agent.pretrain_critic_calql(batch)
            else:
                metrics = agent.update(batch)
            if (i + 1) % max(1, args.log_every) == 0 or (i + 1) == args.offline_pretrain_updates:
                print(f"offline_{args.offline_pretrain_method}_update", i + 1, metrics)
                _wandb_log(
                    wandb_run,
                    dict(
                        offline_update=i + 1,
                        **{
                            "offline/pretrain_method": args.offline_pretrain_method,
                            **{f"offline/{k}": float(v) for k, v in metrics.items()},
                        },
                    ),
                )
    if args.total_env_steps <= 0:
        agent.save(ckpt_path)
        _wandb_log(
            wandb_run,
            dict(
                env_step=0,
                **{"checkpoint/offline_pretrained": 1.0},
            ),
        )
        print("saved offline-pretrained checkpoint:", ckpt_path)
        if wandb_run is not None:
            wandb_run.finish()
        return

    try:
        if loaded_warmup_env_steps >= args.total_env_steps:
            raise SystemExit(
                f"--total-env-steps ({args.total_env_steps}) must exceed loaded warmup "
                f"env steps ({loaded_warmup_env_steps})."
            )

        assert env is not None
        prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
        if wandb_run is not None:
            wandb_run.config.update({"prompt": prompt}, allow_val_change=True)
            if wandb_run.run is not None:
                wandb_run.run.summary["offline/files"] = len(h5_files)
                wandb_run.run.summary["offline/transitions"] = offline_data.size
                wandb_run.run.summary["warmup/buffer_path"] = (
                    str(warmup_buffer_path) if warmup_buffer_path is not None else ""
                )
        base_policy = make_base_policy(
            args.base_policy,
            action_space=env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.resize,
        )
        state_adapter = StateAdapter(args.state_keys)
        image_adapter = (
            ImageAdapter(
                rl_image_keys,
                image_size=args.rl_image_size,
                image_shape=image_shape,
            )
            if args.use_visual_rl
            else None
        )
        max_episode_steps = args.max_episode_steps or getattr(env.unwrapped, "max_episode_steps", None)
        if max_episode_steps is None and getattr(env, "spec", None) is not None:
            max_episode_steps = getattr(env.spec, "max_episode_steps", None)
        max_episode_steps = int(max_episode_steps or 500)
        path_visualizer = None
        if args.visualize_tcp_path:
            path_visualizer = TCPPathVisualizer(
                env=env,
                max_points=args.path_max_points,
                radius=args.path_radius,
                base_color=parse_rgba(args.base_path_color),
                residual_color=parse_rgba(args.residual_path_color),
                tcp_pose_key=args.tcp_pose_key,
            )

        obs, _ = env.reset(seed=args.seed + loaded_warmup_episode_idx)
        base_policy.reset()
        state = state_adapter(obs)
        image = image_adapter(obs) if image_adapter is not None else None
        base_action = base_policy.act(obs)
        if path_visualizer is not None:
            path_visualizer.clear()
            path_visualizer.set_base_prediction_from_chunk(
                obs,
                base_policy.planned_chunk(),
                position_scale=args.base_chunk_position_scale,
                max_actions=args.base_chunk_max_actions,
            )
        _maybe_render(env, render_mode, 0, args.render_every, path_visualizer)
        episode_idx = loaded_warmup_episode_idx
        episode_return = 0.0
        episode_steps = 0
        recent_successes: deque[float] = deque(maxlen=50)
        recent_returns: deque[float] = deque(maxlen=50)
        recent_lengths: deque[float] = deque(maxlen=50)
        t0 = time.time()

        _wandb_log(
            wandb_run,
            dict(
                env_step=loaded_warmup_env_steps,
                **{
                    "warmup/loaded": float(loaded_warmup_env_steps > 0),
                    "warmup/transitions": len(online_buffer),
                    "warmup/episode_idx": loaded_warmup_episode_idx,
                },
            ),
        )

        for env_step in range(loaded_warmup_env_steps + 1, args.total_env_steps + 1):
            if path_visualizer is not None:
                path_visualizer.set_base_prediction_from_chunk(
                    obs,
                    base_policy.planned_chunk(),
                    position_scale=args.base_chunk_position_scale,
                    max_actions=args.base_chunk_max_actions,
                )
            _maybe_render(env, render_mode, env_step, args.render_every, path_visualizer)
            if episode_idx < args.warmup_episodes:
                action_source = "base"
                action = base_action
            else:
                action_source = "residual"
                if args.otf_rollout_actions > 0:
                    action_source = "otf_residual"
                    action = agent.select_action_otf(
                        state,
                        base_action,
                        n_actions=args.otf_rollout_actions,
                        images=image,
                        include_base_action=not args.otf_no_base_candidate,
                    )
                else:
                    action = agent.select_action(
                        state, base_action, images=image, deterministic=False
                    )

            next_obs, reward, terminated, truncated, info = env.step(action)
            if (
                path_visualizer is not None
                and args.path_every > 0
                and env_step % args.path_every == 0
                and action_source in {"residual", "otf_residual"}
            ):
                path_visualizer.add_from_obs(next_obs, "residual")
            done = _as_done(terminated) or _as_done(truncated)
            next_state = state_adapter(next_obs)
            next_image = image_adapter(next_obs) if image_adapter is not None else None
            next_base_action = np.zeros_like(base_action) if done else base_policy.act(next_obs)
            online_buffer.add(
                state,
                action,
                base_action,
                _as_scalar(reward),
                next_state,
                next_base_action,
                done,
                images=image,
                next_images=next_image,
            )
            episode_return += _as_scalar(reward)
            episode_steps += 1

            for _ in range(max(0, int(args.updates_per_env_step))):
                batch = sample_offline_online(
                    offline_buffer,
                    online_buffer,
                    args.batch_size,
                    offline_fraction=args.offline_fraction,
                )
                metrics = agent.update(batch)

            if done or episode_steps >= max_episode_steps:
                success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                current_episode_idx = episode_idx
                is_warmup_episode = current_episode_idx < args.warmup_episodes
                recent_successes.append(float(success))
                recent_returns.append(float(episode_return))
                recent_lengths.append(float(episode_steps))
                print(
                    "episode",
                    dict(
                        idx=current_episode_idx,
                        env_step=env_step,
                        steps=episode_steps,
                        ret=round(episode_return, 4),
                        success=success,
                        online_buffer=len(online_buffer),
                    ),
                )
                _wandb_log(
                    wandb_run,
                    dict(
                        env_step=env_step,
                        **{
                            "episode/index": current_episode_idx,
                            "episode/steps": episode_steps,
                            "episode/return": float(episode_return),
                            "episode/success": float(success),
                            "episode/is_warmup": float(is_warmup_episode),
                            "episode/online_buffer": len(online_buffer),
                            "episode/success_rate_50": float(np.mean(recent_successes)),
                            "episode/return_mean_50": float(np.mean(recent_returns)),
                            "episode/steps_mean_50": float(np.mean(recent_lengths)),
                        },
                    ),
                )
                episode_idx += 1
                if (
                    not warmup_buffer_saved
                    and warmup_buffer_path is not None
                    and episode_idx >= args.warmup_episodes
                ):
                    warmup_buffer_path.parent.mkdir(parents=True, exist_ok=True)
                    online_buffer.save(
                        warmup_buffer_path,
                        env_id=args.env_id,
                        episode_idx=episode_idx,
                        env_steps=env_step,
                        warmup_episodes=args.warmup_episodes,
                        base_policy=args.base_policy,
                        seed=args.seed,
                    )
                    warmup_buffer_saved = True
                    print(
                        "saved warmup buffer:",
                        dict(
                            path=str(warmup_buffer_path),
                            transitions=len(online_buffer),
                            episode_idx=episode_idx,
                            env_steps=env_step,
                        ),
                    )
                    _wandb_log(
                        wandb_run,
                        dict(
                            env_step=env_step,
                            **{
                                "warmup/saved": 1.0,
                                "warmup/transitions": len(online_buffer),
                                "warmup/episode_idx": episode_idx,
                            },
                        ),
                    )
                obs, _ = env.reset(seed=args.seed + episode_idx)
                base_policy.reset()
                state = state_adapter(obs)
                image = image_adapter(obs) if image_adapter is not None else None
                base_action = base_policy.act(obs)
                if path_visualizer is not None:
                    path_visualizer.clear()
                    path_visualizer.set_base_prediction_from_chunk(
                        obs,
                        base_policy.planned_chunk(),
                        position_scale=args.base_chunk_position_scale,
                        max_actions=args.base_chunk_max_actions,
                    )
                _maybe_render(env, render_mode, env_step, args.render_every, path_visualizer)
                episode_return = 0.0
                episode_steps = 0
            else:
                obs = next_obs
                state = next_state
                image = next_image
                base_action = next_base_action

            if env_step % max(1, args.log_every) == 0:
                elapsed_s = round(time.time() - t0, 1)
                print(
                    "train",
                    dict(
                        env_step=env_step,
                        updates=agent.total_updates,
                        elapsed_s=elapsed_s,
                        metrics=metrics,
                    ),
                )
                train_payload = dict(
                    env_step=env_step,
                    **{
                        "train/updates": agent.total_updates,
                        "train/elapsed_s": elapsed_s,
                        "train/online_buffer": len(online_buffer),
                        "train/offline_buffer": len(offline_buffer),
                        "train/episodes_finished": episode_idx,
                    },
                )
                if elapsed_s > 0:
                    train_payload["train/env_steps_per_s"] = float(env_step) / float(elapsed_s)
                if recent_successes:
                    train_payload["train/success_rate_50"] = float(np.mean(recent_successes))
                train_payload.update({f"train/{k}": float(v) for k, v in metrics.items()})
                _wandb_log(wandb_run, train_payload)
            if env_step % max(1, args.save_every) == 0:
                step_ckpt = output_dir / f"residual_sac_step_{env_step}.pt"
                agent.save(step_ckpt)
                _wandb_log(
                    wandb_run,
                    dict(
                        env_step=env_step,
                        **{"checkpoint/saved": 1.0},
                    ),
                )

        agent.save(ckpt_path)
        _wandb_log(
            wandb_run,
            dict(
                env_step=args.total_env_steps,
                **{"checkpoint/final": 1.0},
            ),
        )
        if wandb_run is not None and wandb_run.run is not None:
            wandb_run.run.summary["checkpoint/final_path"] = str(ckpt_path)
            wandb_run.run.summary["checkpoint/final_env_step"] = args.total_env_steps
        print("saved:", ckpt_path)
    finally:
        if env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
