#!/usr/bin/env python
"""
Record random rollouts to ManiSkill's trajectory format (.h5 + .json).

This is a non-interactive baseline collector useful for debugging the pipeline.

Example:
  python scripts/record_random.py --env-id TurnGlobeValve-v1 --out-dir data/demos/debug --num-episodes 5
"""

import argparse
from pathlib import Path
import sys

import numpy as np


def _zero_action(action_space):
    import gymnasium as gym

    if isinstance(action_space, gym.spaces.Dict):
        return {k: np.zeros(v.shape, dtype=np.float32) for k, v in action_space.spaces.items()}
    return np.zeros(action_space.shape, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="TurnGlobeValve-v1")
    parser.add_argument("--out-dir", type=str, default="data/demos/debug")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--obs-mode", type=str, default="state_dict")
    parser.add_argument("--reward-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    # Allow running without `pip install -e .` by adding repo/src to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode

    import maniskill_myws

    maniskill_myws.register()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.save_video else None,
    )
    env = RecordEpisode(
        env,
        output_dir=str(out_dir),
        save_trajectory=True,
        save_video=bool(args.save_video),
        source_type="random",
        source_desc="Random policy rollouts for debugging the data pipeline.",
    )

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = _zero_action(env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
        print(f"episode {ep} done, success={bool(info.get('success', False))}")

    env.close()


if __name__ == "__main__":
    main()


