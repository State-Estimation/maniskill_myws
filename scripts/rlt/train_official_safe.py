#!/usr/bin/env python
"""Run the official SAFE trainer on ManiSkill pre-velocity rollouts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SAFE_ROOT = ROOT / "third_party" / "SAFE"
EXPECTED_SAFE_COMMIT = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SAFE_ROOT))


def _require_official_safe_checkout() -> None:
    if not (SAFE_ROOT / "failure_prob" / "train.py").is_file():
        raise FileNotFoundError(
            f"Official SAFE checkout not found at {SAFE_ROOT}; clone vla-safe/SAFE there"
        )
    commit = subprocess.run(
        ["git", "-C", str(SAFE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_SAFE_COMMIT:
        raise RuntimeError(
            f"Official SAFE commit {commit} != supported {EXPECTED_SAFE_COMMIT}"
        )


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            "Official SAFE is supported here on Python 3.10 only; create the "
            "documented `vla-safe` environment before training."
        )
    _require_official_safe_checkout()

    try:
        from failure_prob.data.utils import Rollout
        from failure_prob import train as official_train
        from maniskill_myws.rlt.safe_rollouts import OfficialSafeAdapter
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing SAFE training dependency {error.name!r}. Install a compatible "
            "PyTorch build, then run `pip install -r requirements-safe.txt` and "
            "`pip install -e third_party/SAFE`."
        ) from None

    adapter = OfficialSafeAdapter(Rollout)
    original_process_cfg = official_train.process_cfg

    def process_cfg(cfg):
        return adapter.prepare_config(original_process_cfg(cfg))

    official_train.process_cfg = process_cfg
    official_train.load_rollouts = adapter.load_rollouts
    official_train.split_rollouts = adapter.split_rollouts
    official_train.main()


if __name__ == "__main__":
    main()
