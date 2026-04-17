#!/usr/bin/env python
"""
Replay ManiSkill trajectory files into a separate output tree.

Why this exists:
- `mani_skill.trajectory.replay_trajectory` writes outputs next to its input file.
- Our replayed dataset files already use the standard
  `<name>.<obs_mode>.<control_mode>.<sim_backend>.h5` naming pattern, so replaying
  them in place would collide with the output naming scheme.

This helper mirrors the source directory structure under a new root, creates a
temporary unsuffixed copy of each trajectory (`trajectory.h5/.json`), replays it,
then removes the temporary source copy so only the newly rendered outputs remain.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _strip_standard_suffix(stem: str) -> str:
    parts = stem.split(".")
    if len(parts) > 1:
        return parts[0]
    return stem


def _iter_h5_files(src_root: Path) -> list[Path]:
    return sorted(p for p in src_root.rglob("*.h5") if p.is_file())


def _build_replay_cmd(
    repo_root: Path,
    tmp_h5: Path,
    *,
    sim_backend: str,
    obs_mode: str,
    control_mode: str,
    render_mode: str,
    shader: str | None,
    count: int | None,
) -> list[str]:
    replay_bootstrap = """
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))

import maniskill_myws
maniskill_myws.register()

from mani_skill.trajectory.replay_trajectory import main, parse_args

main(parse_args(sys.argv[2:]))
""".strip()
    cmd = [
        sys.executable,
        "-c",
        replay_bootstrap,
        str(repo_root),
        "--traj-path",
        str(tmp_h5),
        "--sim-backend",
        sim_backend,
        "--obs-mode",
        obs_mode,
        "--target-control-mode",
        control_mode,
        "--render-mode",
        render_mode,
        "--use-env-states",
        "--save-traj",
    ]
    if shader:
        cmd += ["--shader", shader]
    if count is not None:
        cmd += ["--count", str(count)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Replay an existing ManiSkill dataset into a new directory tree.")
    p.add_argument("--src-dir", type=str, required=True, help="Root directory containing source .h5/.json files.")
    p.add_argument("--dst-dir", type=str, required=True, help="Root directory to write replayed outputs into.")
    p.add_argument("--sim-backend", type=str, default="physx_cpu")
    p.add_argument("--obs-mode", type=str, default="rgb")
    p.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    p.add_argument("--render-mode", type=str, default="sensors")
    p.add_argument("--shader", type=str, default=None, help="Optional shader override, e.g. rt-fast.")
    p.add_argument("--count", type=int, default=None, help="Optional per-file episode limit for debugging.")
    p.add_argument(
        "--keep-temp-source",
        action="store_true",
        help="Keep the temporary copied source h5/json files in dst-dir after replay.",
    )
    args = p.parse_args()

    src_root = Path(args.src_dir).expanduser().resolve()
    dst_root = Path(args.dst_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    h5_files = _iter_h5_files(src_root)
    if not h5_files:
        raise SystemExit(f"No .h5 files found under {src_root}")

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Files: {len(h5_files)}")

    for idx, src_h5 in enumerate(h5_files, start=1):
        rel = src_h5.relative_to(src_root)
        dst_subdir = (dst_root / rel.parent).resolve()
        dst_subdir.mkdir(parents=True, exist_ok=True)

        src_json = src_h5.with_suffix(".json")
        base_name = _strip_standard_suffix(src_h5.stem)
        tmp_h5 = dst_subdir / f"{base_name}.h5"
        tmp_json = dst_subdir / f"{base_name}.json"

        print(f"[{idx}/{len(h5_files)}] {src_h5}")
        shutil.copy2(src_h5, tmp_h5)
        if src_json.exists():
            shutil.copy2(src_json, tmp_json)

        cmd = _build_replay_cmd(
            repo_root,
            tmp_h5,
            sim_backend=args.sim_backend,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            shader=args.shader,
            count=args.count,
        )
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-myws")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"Replay failed for {src_h5}\n"
                f"Temporary input: {tmp_h5}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}"
            ) from e

        if not args.keep_temp_source:
            if tmp_h5.exists():
                tmp_h5.unlink()
            if tmp_json.exists():
                tmp_json.unlink()

    print("Replay complete.")


if __name__ == "__main__":
    main()
