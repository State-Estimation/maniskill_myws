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
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_OFFLINE_DIR = "dataset/Replayed_traj_data_openSafeDoor2"
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_scalar(x) -> float:
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def _as_done(x) -> bool:
    arr = np.asarray(x)
    return bool(arr.reshape(-1)[0])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="state_dict")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
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
    parser.add_argument("--resize", type=int, default=224)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=250_000)
    parser.add_argument("--offline-pretrain-updates", type=int, default=1_000)
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=250_000)
    parser.add_argument("--offline-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=25_000)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--cql-alpha", type=float, default=0.0)

    parser.add_argument("--output-dir", type=str, default="outputs/pld/OpenSafeDoor-v2")
    parser.add_argument("--checkpoint-name", type=str, default="residual_sac.pt")
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.pld.h5_replay import find_h5_files, load_h5_replay
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.replay_buffer import ReplayBuffer, sample_offline_online
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    from maniskill_myws.pld.state import StateAdapter
    from maniskill_myws.task_prompts import get_task_prompt

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maniskill_myws.register()

    offline_dir = str((repo_root / args.offline_h5_dir).resolve())
    h5_files = find_h5_files(h5_dir=offline_dir, h5_glob=args.offline_h5_glob)
    if not h5_files:
        raise SystemExit(f"No H5 files found under {offline_dir}")
    offline_data = load_h5_replay(
        h5_files,
        state_keys=args.state_keys,
        actions_key=args.actions_key,
        reward_key=args.reward_key,
        success_only=not args.include_failed_offline,
        reward_from_success=args.reward_from_success,
        base_action_mode=args.offline_base_action_mode,
        max_transitions=args.max_offline_transitions,
    )
    print(
        "offline replay:",
        dict(files=len(h5_files), transitions=offline_data.size, state_dim=offline_data.state_dim),
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    action_dim = offline_data.action_dim
    env = None
    if args.total_env_steps > 0:
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            render_mode=None,
        )
        low, high = _action_bounds(env.action_space, action_dim)
    else:
        low, high = (-1.0,) * action_dim, (1.0,) * action_dim

    offline_buffer = ReplayBuffer(args.buffer_capacity, offline_data.state_dim, action_dim)
    offline_buffer.add_offline_data(offline_data)
    online_buffer = ReplayBuffer(args.buffer_capacity, offline_data.state_dim, action_dim)

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
            actor_update_interval=2,
            grad_clip_norm=args.grad_clip_norm,
            cql_alpha=args.cql_alpha,
            action_low=low,
            action_high=high,
        ),
        device=device,
    )

    for i in range(args.offline_pretrain_updates):
        metrics = agent.update(offline_buffer.sample(args.batch_size))
        if (i + 1) % max(1, args.log_every) == 0:
            print("offline_update", i + 1, metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name
    if args.total_env_steps <= 0:
        agent.save(ckpt_path)
        print("saved offline-pretrained checkpoint:", ckpt_path)
        return

    assert env is not None
    prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
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
    max_episode_steps = args.max_episode_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_episode_steps is None and getattr(env, "spec", None) is not None:
        max_episode_steps = getattr(env.spec, "max_episode_steps", None)
    max_episode_steps = int(max_episode_steps or 500)

    obs, _ = env.reset(seed=args.seed)
    base_policy.reset()
    state = state_adapter(obs)
    base_action = base_policy.act(obs)
    episode_idx = 0
    episode_return = 0.0
    episode_steps = 0
    t0 = time.time()

    for env_step in range(1, args.total_env_steps + 1):
        if episode_idx < args.warmup_episodes:
            action = base_action
        else:
            action = agent.select_action(state, base_action, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = _as_done(terminated) or _as_done(truncated)
        next_state = state_adapter(next_obs)
        next_base_action = np.zeros_like(base_action) if done else base_policy.act(next_obs)
        online_buffer.add(
            state,
            action,
            base_action,
            _as_scalar(reward),
            next_state,
            next_base_action,
            done,
        )
        episode_return += _as_scalar(reward)
        episode_steps += 1

        for _ in range(args.updates_per_env_step):
            batch = sample_offline_online(
                offline_buffer,
                online_buffer,
                args.batch_size,
                offline_fraction=args.offline_fraction,
            )
            metrics = agent.update(batch)

        if done or episode_steps >= max_episode_steps:
            success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            print(
                "episode",
                dict(
                    idx=episode_idx,
                    env_step=env_step,
                    steps=episode_steps,
                    ret=round(episode_return, 4),
                    success=success,
                    online_buffer=len(online_buffer),
                ),
            )
            episode_idx += 1
            obs, _ = env.reset(seed=args.seed + episode_idx)
            base_policy.reset()
            state = state_adapter(obs)
            base_action = base_policy.act(obs)
            episode_return = 0.0
            episode_steps = 0
        else:
            obs = next_obs
            state = next_state
            base_action = next_base_action

        if env_step % max(1, args.log_every) == 0:
            print(
                "train",
                dict(
                    env_step=env_step,
                    updates=agent.total_updates,
                    elapsed_s=round(time.time() - t0, 1),
                    metrics=metrics,
                ),
            )
        if env_step % max(1, args.save_every) == 0:
            agent.save(output_dir / f"residual_sac_step_{env_step}.pt")

    agent.save(ckpt_path)
    env.close()
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()
