#!/usr/bin/env python
"""
Collect frozen base-policy rollouts in ManiSkill RecordEpisode format.

This is the PLD Algorithm 1 initialization dataset:
  D_offline = successful and failed trials collected by pi_b.

The output is standard ManiSkill `.h5 + .json`, with rgb observations, Panda
wrist camera, actions in the configured control mode (standard: pd_joint_pos),
and the same keys expected by docs/maniskill_dataset_standard.md and
scripts/convert_traj_to_lerobot.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


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


def _as_done(x) -> bool:
    return bool(_to_numpy(x).reshape(-1)[0])


def _as_success(info: dict) -> bool:
    if "success" not in info:
        return False
    return bool(_to_numpy(info["success"]).reshape(-1)[0])


def _normalize_render_mode(render_mode: str | None) -> str | None:
    if render_mode is None:
        return None
    if render_mode.lower() in {"none", "null", ""}:
        return None
    return render_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument(
        "--robot-uids",
        type=str,
        default="panda_wristcam",
        help="Robot UID passed to gym.make. Use 'none' to let the env choose its default.",
    )
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--num-successes", type=int, default=50)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Stop after this many attempts even if --num-successes is not reached.",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--rollout-window-steps",
        type=int,
        default=50,
        help=(
            "Group this many environment steps into one throughput/logging window. "
            "Remote OpenPI chunk caching is owned by the policy client so collection "
            "matches run_pi0_remote_multi_seed.py exactly."
        ),
    )
    parser.add_argument(
        "--log-every-windows",
        type=int,
        default=1,
        help="Print live throughput every N rollout windows; 0 disables window logs.",
    )
    parser.add_argument(
        "--discard-failed",
        action="store_true",
        help="Discard failed episodes. By default failures are retained for critic training.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi")
    parser.add_argument("--server", type=str, default=None, help="ws://host:port for remote_openpi")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument(
        "--env-device",
        type=str,
        default=None,
        help="Optional ManiSkill environment device, e.g. 'cuda:1'.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/Pi0_rollout_OpenSafeDoor-v2_pd_joint_pos",
        help="Directory for RecordEpisode .h5 + .json outputs.",
    )
    parser.add_argument("--trajectory-name", type=str, default="pi0_base_policy")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()
    if args.rollout_window_steps <= 0:
        raise SystemExit("--rollout-window-steps must be positive")
    if args.log_every_windows < 0:
        raise SystemExit("--log-every-windows must be non-negative")

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode

    import maniskill_myws
    from maniskill_myws.pld.env_device import apply_env_device_kwargs
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.task_prompts import get_task_prompt

    maniskill_myws.register()

    render_mode = _normalize_render_mode(args.render_mode)
    reward_mode = None if args.reward_mode.lower() in {"none", "null", ""} else args.reward_mode
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.save_video else render_mode,
    )
    # Match the fast multi-seed evaluator: overriding the loop bound alone is
    # insufficient because Gym's TimeLimit would still truncate at the task's
    # registered horizon (notably 200 steps for TakeSafetyHook-v1).
    if args.max_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_steps
    if args.robot_uids.lower() not in {"none", "null", ""}:
        env_kwargs["robot_uids"] = args.robot_uids
    apply_env_device_kwargs(env_kwargs, args.env_device)
    env = gym.make(args.env_id, **env_kwargs)
    env = RecordEpisode(
        env,
        output_dir=str(Path(args.output_dir)),
        trajectory_name=args.trajectory_name,
        save_trajectory=True,
        save_video=bool(args.save_video),
        video_fps=args.video_fps,
        record_env_state=True,
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(action_low)) or not np.all(np.isfinite(action_high)):
        raise ValueError("Replay collection requires finite action-space bounds")

    prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    action_dim = int(np.prod(env.action_space.shape))
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

    max_steps = args.max_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    max_steps = int(max_steps or 500)
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(args.num_successes, args.num_successes * 5)

    print(
        "collector",
        dict(
            env_id=args.env_id,
            control_mode=args.control_mode,
            action_dim=action_dim,
            max_steps=max_steps,
            rollout_window_steps=args.rollout_window_steps,
            target_successes=args.num_successes,
            max_attempts=max_attempts,
            keep_failed=bool(args.keep_failed or not args.discard_failed),
        ),
        flush=True,
    )

    successes = 0
    attempts = 0
    last_keep = False
    clipped_action_steps = 0
    total_env_steps = 0
    total_rollout_windows = 0
    collection_start = time.perf_counter()
    keep_failed = bool(args.keep_failed or not args.discard_failed)
    try:
        while successes < args.num_successes and attempts < max_attempts:
            seed = args.start_seed + attempts
            obs, _ = env.reset(seed=seed, save=last_keep)
            base_policy.reset()

            success = False
            steps = 0
            episode_start = time.perf_counter()
            episode_windows = 0
            while steps < max_steps:
                requested_window_steps = min(
                    args.rollout_window_steps,
                    max_steps - steps,
                )
                window_env_start = time.perf_counter()
                inference_s = 0.0
                executed_in_window = 0
                done = False
                for _ in range(requested_window_steps):
                    # Match the fast multi-seed evaluator. RemoteOpenPIBasePolicy.act
                    # consumes its local action queue and only performs a websocket
                    # inference when that queue is empty; calling it per env step does
                    # not imply one remote inference per env step.
                    inference_start = time.perf_counter()
                    raw_action = np.asarray(base_policy.act(obs), dtype=np.float32).reshape(-1)
                    inference_s += time.perf_counter() - inference_start
                    if raw_action.shape != (action_dim,):
                        raise ValueError(
                            f"Base action shape {raw_action.shape} does not match "
                            f"expected {(action_dim,)} for {args.control_mode}"
                        )
                    if not np.all(np.isfinite(raw_action)):
                        raise ValueError("Base policy produced a non-finite action")
                    action = np.clip(raw_action, action_low, action_high).astype(
                        np.float32, copy=False
                    )
                    clipped_action_steps += int(
                        np.any(~np.isclose(action, raw_action, atol=1e-6))
                    )
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_env_steps += 1
                    executed_in_window += 1
                    success = _as_success(info)
                    if render_mode is not None:
                        env.render()
                    done = _as_done(terminated) or _as_done(truncated)
                    if done or steps >= max_steps:
                        break

                episode_windows += 1
                total_rollout_windows += 1
                if (
                    args.log_every_windows > 0
                    and episode_windows % args.log_every_windows == 0
                ):
                    episode_elapsed = time.perf_counter() - episode_start
                    print(
                        "window",
                        dict(
                            attempt=attempts + 1,
                            seed=seed,
                            idx=episode_windows,
                            steps=f"{steps}/{max_steps}",
                            executed=executed_in_window,
                            inference_s=round(inference_s, 3),
                            env_s=round(time.perf_counter() - window_env_start, 3),
                            episode_steps_per_s=round(steps / max(episode_elapsed, 1e-9), 2),
                        ),
                        flush=True,
                    )
                if done:
                    break

            last_keep = bool(success or keep_failed)
            successes += int(success)
            attempts += 1
            episode_elapsed = time.perf_counter() - episode_start
            print(
                "episode",
                dict(
                    attempt=attempts,
                    seed=seed,
                    steps=steps,
                    windows=episode_windows,
                    elapsed_s=round(episode_elapsed, 2),
                    steps_per_s=round(steps / max(episode_elapsed, 1e-9), 2),
                    success=success,
                    kept=last_keep,
                    successes=f"{successes}/{args.num_successes}",
                ),
                flush=True,
            )

        if not last_keep:
            env.reset(seed=args.start_seed + attempts, save=False)
    finally:
        env.close()

    print(
        "saved_dir:",
        args.output_dir,
        "successes:",
        successes,
        "/",
        attempts,
        "attempts",
        "clipped_action_steps:",
        clipped_action_steps,
        "rollout_windows:",
        total_rollout_windows,
        "env_steps:",
        total_env_steps,
        "elapsed_s:",
        round(time.perf_counter() - collection_start, 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
