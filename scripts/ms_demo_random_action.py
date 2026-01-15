#!/usr/bin/env python
"""
Wrapper around `mani_skill.examples.demo_random_action` that registers custom tasks first.

Usage (same flags as ManiSkill):
  python scripts/ms_demo_random_action.py -e TurnGlobeValve-v1 --render-mode human
"""

import sys
from pathlib import Path


def main():
    # Allow running without `pip install -e .` by adding repo/src to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import maniskill_myws

    maniskill_myws.register()

    # Delegate CLI+main to ManiSkill's script.
    import tyro
    from mani_skill.examples.demo_random_action import Args, main as ms_main

    args = tyro.cli(Args)
    ms_main(args)


if __name__ == "__main__":
    main()


