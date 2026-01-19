#!/usr/bin/env python
"""
Inspect ManiSkill RecordEpisode trajectory (.h5) structure.

Usage:
  conda activate mani_skill
  python scripts/inspect_traj_h5.py --h5 data/demos/debug/xxx.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--max-items", type=int, default=400)
    args = parser.parse_args()

    try:
        import h5py
    except Exception as e:  # pragma: no cover
        raise SystemExit("Please `pip install h5py` in your active conda env.") from e

    p = Path(args.h5)
    if not p.exists():
        raise SystemExit(f"Not found: {p}")

    count = 0

    def visit(name: str, obj: Any) -> None:
        nonlocal count
        if count >= args.max_items:
            return
        if isinstance(obj, h5py.Dataset):
            print(f"{name}\tshape={obj.shape}\tdtype={obj.dtype}")
            count += 1

    with h5py.File(p, "r") as f:
        trajs = sorted([k for k in f.keys() if k.startswith("traj_")])
        print("Trajs:", trajs)
        if not trajs:
            return
        g = f[trajs[0]]
        print("Top keys:", list(g.keys()))
        g.visititems(visit)


if __name__ == "__main__":
    main()


