#!/usr/bin/env python
"""
Collect successful frozen base-policy rollouts in ManiSkill RecordEpisode format.

This is the PLD Algorithm 1 initialization dataset:
  D_offline = successful trials collected by pi_b.

The output is standard ManiSkill `.h5 + .json`, with rgb observations, Panda
wrist camera, pd_ee_delta_pose actions, and the same keys expected by
docs/maniskill_dataset_standard.md and scripts/convert_traj_to_lerobot.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
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
    parser.add_argument("--keep-failed", action="store_true")

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi")
    parser.add_argument("--server", type=str, default=None, help="ws://host:port for remote_openpi")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/Pi0_rollout_OpenSafeDoor-v2",
        help="Directory for RecordEpisode .h5 + .json outputs.",
    )
    parser.add_argument("--trajectory-name", type=str, default="pi0_base_policy")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode

    import maniskill_myws
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
    if args.robot_uids.lower() not in {"none", "null", ""}:
        env_kwargs["robot_uids"] = args.robot_uids
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

    successes = 0
    attempts = 0
    last_keep = False
    try:
        while successes < args.num_successes and attempts < max_attempts:
            seed = args.start_seed + attempts
            obs, _ = env.reset(seed=seed, save=(last_keep or args.keep_failed))
            base_policy.reset()

            success = False
            steps = 0
            for step in range(max_steps):
                action = base_policy.act(obs)
                obs, _, terminated, truncated, info = env.step(action)
                steps = step + 1
                success = _as_success(info)
                if render_mode is not None:
                    env.render()
                if _as_done(terminated) or _as_done(truncated):
                    break

            last_keep = success
            successes += int(success)
            attempts += 1
            print(
                "episode",
                dict(
                    attempt=attempts,
                    seed=seed,
                    steps=steps,
                    success=success,
                    kept=success or args.keep_failed,
                    successes=f"{successes}/{args.num_successes}",
                ),
            )

        if not (last_keep or args.keep_failed):
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
    )


if __name__ == "__main__":
    main()
