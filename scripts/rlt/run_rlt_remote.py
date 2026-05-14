#!/usr/bin/env python
"""Run ManiSkill with openpi-RLT features and optional online actor refinement."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


def _infer_control_dt(env, fallback_hz: float = 20.0) -> tuple[float, float]:
    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = fallback_hz
    control_freq = float(control_freq)
    if control_freq <= 0:
        control_freq = fallback_hz
    return 1.0 / control_freq, control_freq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-server", type=str, required=True, help="openpi-RLT websocket, e.g. ws://127.0.0.1:8000")
    parser.add_argument("--actor-url", type=str, default=None, help="optional actor_service URL, e.g. http://127.0.0.1:9101")
    parser.add_argument("--replay-url", type=str, default=None, help="optional replay_manager URL, e.g. http://127.0.0.1:9102")
    parser.add_argument("--env-id", type=str, default="TurnGlobeValve-v1")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--act-dim", type=int, default=7)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--z-dim", type=int, default=2048)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--actor-deterministic", action="store_true", default=True)
    parser.add_argument("--no-fallback-to-ref", action="store_true")
    parser.add_argument("--collection-phase", type=str, default="online")
    parser.add_argument("--output-root", type=str, default="outputs/rlt")
    parser.add_argument("--save-trajectory", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.openpi_bridge.obs_to_openpi import ObsAdapter
    from maniskill_myws.rlt_bridge import ChunkTransitionBuilder
    from maniskill_myws.rlt_bridge import RLTOnlineChunkPolicy
    from maniskill_myws.rlt_bridge import ReplayClient

    maniskill_myws.register()

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    obs, info = env.reset(seed=args.seed)

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    if max_steps is None:
        max_steps = 200

    prompt = args.prompt
    if prompt is None:
        if hasattr(env.unwrapped, "DEFAULT_TASK_PROMPT"):
            prompt = env.unwrapped.DEFAULT_TASK_PROMPT
        else:
            raise SystemExit(f"Env '{args.env_id}' has no DEFAULT_TASK_PROMPT; pass --prompt.")

    state_keys = args.state_keys if args.state_keys else ["agent/qpos", "agent/qvel", "extra/tcp_pose"]
    adapter = ObsAdapter(
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=state_keys,
        prompt=prompt,
    )
    policy = RLTOnlineChunkPolicy(
        feature_server=args.feature_server,
        actor_url=args.actor_url,
        obs_adapter=adapter,
        act_dim=args.act_dim,
        chunk_len=args.chunk_len,
        z_dim=args.z_dim,
        proprio_dim=args.proprio_dim,
        actor_deterministic=args.actor_deterministic,
        fallback_to_ref=not args.no_fallback_to_ref,
    )
    policy.reset(episode_id=args.seed)

    replay = None if args.replay_url is None else ReplayClient(args.replay_url)
    builder = ChunkTransitionBuilder(
        chunk_len=args.chunk_len,
        action_dim=args.act_dim,
        collection_phase=args.collection_phase,
    )

    real_time_dt = None
    if args.real_time:
        real_time_dt, control_freq = _infer_control_dt(env)
        print("real-time pacing:", dict(control_freq_hz=control_freq, target_dt_s=real_time_dt))

    should_save = bool(args.save_trajectory)
    out_dir = None
    if should_save:
        out_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

    traj_actions: list[np.ndarray] = []
    traj_rewards: list[float] = []
    fallback_count = 0
    actor_versions: list[int] = []

    try:
        for step in range(max_steps):
            step_wall_start = time.perf_counter()

            if replay is not None and builder.plan is not None and not policy.has_pending_actions():
                next_features = policy.features_for_obs(obs)
                replay.add_transition(builder.to_transition(next_features=next_features, done=False, success=False))
                builder.reset()

            act = policy.act(obs, step_id=step)
            plan = policy.last_plan
            if plan is None:
                raise RuntimeError("RLT policy did not produce a plan.")
            if builder.plan is None:
                builder.begin(plan)
                fallback_count += int(plan.actor_result.used_fallback)
                actor_versions.append(int(plan.actor_result.actor_param_version))

            cursor = min(len(builder.actions), plan.features.ref_chunk.shape[0] - 1)
            ref_action = plan.features.ref_chunk[cursor, : args.act_dim]

            if args.render_mode is not None:
                try:
                    env.render()
                except Exception:
                    pass

            next_obs, reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)
            success = bool(info.get("success", False))
            traj_actions.append(np.asarray(act, dtype=np.float32))
            traj_rewards.append(float(reward))

            builder.append_step(
                action=act,
                ref_action=ref_action,
                reward=float(reward),
                source=int(plan.actor_result.source),
            )

            obs = next_obs
            if replay is not None and (builder.ready() or done):
                next_features = policy.features_for_obs(obs)
                replay.add_transition(builder.to_transition(next_features=next_features, done=done, success=success))
                builder.reset()

            if done:
                break
            if real_time_dt is not None:
                sleep_s = real_time_dt - (time.perf_counter() - step_wall_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        env.close()

    if args.save_trajectory and out_dir is not None:
        np.savez_compressed(
            out_dir / "trajectory.npz",
            actions=np.stack(traj_actions) if traj_actions else np.zeros((0, args.act_dim), dtype=np.float32),
            rewards=np.asarray(traj_rewards, dtype=np.float32),
            actor_versions=np.asarray(actor_versions, dtype=np.int32),
            prompt=prompt,
            env_id=args.env_id,
        )

    print(
        "done:",
        dict(
            steps=len(traj_actions),
            reward_sum=float(np.sum(traj_rewards)) if traj_rewards else 0.0,
            success=bool(info.get("success", False)),
            fallback_count=fallback_count,
            actor_version_last=actor_versions[-1] if actor_versions else -1,
            trajectory_dir=None if out_dir is None else str(out_dir),
        ),
    )


if __name__ == "__main__":
    main()
