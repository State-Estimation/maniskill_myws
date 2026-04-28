#!/usr/bin/env python
"""
Collect PLD hybrid takeover trajectories for SFT.

This is Stage 2 of PLD:
  - run the frozen base policy for a random probing horizon T_base
  - switch to a_base + residual_delta
  - record successful hybrid episodes as ManiSkill RecordEpisode H5/JSON
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_done(x) -> bool:
    return bool(np.asarray(x).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--probe-alpha", type=float, default=0.6)
    parser.add_argument("--probe-steps", type=int, default=None)
    parser.add_argument(
        "--stochastic-residual",
        action="store_true",
        help="Sample from the residual actor instead of using its tanh mean.",
    )
    parser.add_argument("--keep-failed", action="store_true")

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi")
    parser.add_argument("--server", type=str, default=None, help="ws://host:port for remote_openpi")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="data/pld/OpenSafeDoor-v2")
    parser.add_argument("--trajectory-name", type=str, default="pld_hybrid")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    import torch
    from mani_skill.utils.wrappers.record import RecordEpisode

    import maniskill_myws
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.sac import ResidualSAC
    from maniskill_myws.pld.state import StateAdapter
    from maniskill_myws.task_prompts import get_task_prompt

    maniskill_myws.register()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = ResidualSAC.load(args.checkpoint, device=device)

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.save_video else args.render_mode,
    )
    env = RecordEpisode(
        env,
        output_dir=str(Path(args.output_dir)),
        trajectory_name=args.trajectory_name,
        save_trajectory=True,
        save_video=bool(args.save_video),
        video_fps=args.video_fps,
        source_type="pld",
        source_desc=(
            "PLD hybrid rollout: frozen base-policy probing followed by residual SAC takeover."
        ),
    )

    prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    action_dim = agent.config.action_dim
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

    max_steps = args.max_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    max_steps = int(max_steps or 500)
    rng = np.random.default_rng(args.start_seed)

    last_keep = False
    successes = 0
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.start_seed + ep, save=(last_keep or args.keep_failed))
        base_policy.reset()
        if args.probe_steps is not None:
            probe_steps = int(args.probe_steps)
        else:
            max_probe = max(0, int(round(float(args.probe_alpha) * max_steps)))
            probe_steps = int(rng.integers(0, max_probe + 1))

        success = False
        for step in range(max_steps):
            if args.render_mode is not None:
                try:
                    env.render()
                except Exception:
                    pass
            base_action = base_policy.act(obs)
            if step < probe_steps:
                action = base_action
            else:
                state = state_adapter(obs)
                action = agent.select_action(
                    state,
                    base_action,
                    deterministic=not bool(args.stochastic_residual),
                )
            obs, reward, terminated, truncated, info = env.step(action)
            success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            if _as_done(terminated) or _as_done(truncated):
                break

        last_keep = success
        successes += int(success)
        print(
            "episode",
            dict(
                idx=ep,
                seed=args.start_seed + ep,
                probe_steps=probe_steps,
                success=success,
                kept=success or args.keep_failed,
            ),
        )

    if not (last_keep or args.keep_failed):
        env.reset(seed=args.start_seed + args.num_episodes, save=False)
    env.close()
    print("saved_dir:", args.output_dir, "successes:", successes, "/", args.num_episodes)


if __name__ == "__main__":
    main()
