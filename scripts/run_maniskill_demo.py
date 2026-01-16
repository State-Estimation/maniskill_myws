#!/usr/bin/env python
"""
Run any ManiSkill demo module, but register maniskill_myws custom tasks first.

Example:
  python scripts/run_maniskill_demo.py mani_skill.examples.demo_random_action -e TurnGlobeValve-v1 --render-mode human

This avoids writing one wrapper per ManiSkill demo.
"""

import runpy
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/run_maniskill_demo.py <module> [demo args...]\n"
            "Example: python scripts/run_maniskill_demo.py mani_skill.examples.demo_random_action -e TurnGlobeValve-v1"
        )

    module = sys.argv[1]
    demo_argv = [sys.argv[0]] + sys.argv[2:]

    # Allow running without `pip install -e .` by adding repo/src to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import maniskill_myws

    maniskill_myws.register()

    # Run the module as if `python -m <module> ...` was invoked.
    sys.argv = demo_argv
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()


