#!/usr/bin/env python
"""
Quick environment sanity check.

Example:
  python scripts/check_env.py --env-id TurnGlobeValve-v1
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="TurnGlobeValve-v1")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--reward-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    args = parser.parse_args()

    # Allow running without `pip install -e .` by adding repo/src to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym

    import maniskill_myws

    maniskill_myws.register()

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=None,
    )
    obs, info = env.reset(seed=0)
    print("reset ok:", type(obs), "info keys:", list(info.keys()))
    env.close()


if __name__ == "__main__":
    main()


