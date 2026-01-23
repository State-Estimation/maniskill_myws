#!/usr/bin/env python
"""
ManiSkill visualization runner for π0 served remotely via websocket.

This is the maniskill_myws analogue of:
  test_any_policy/scripts/graspvla/run_graspvla*.py

Workflow:
1) Start server (separate env/machine):
   python scripts/pi0/serve.py --checkpoint gs://openpi-assets/checkpoints/pi05_libero --config pi05_libero --port 8000
2) Run this client (mani_skill env):
   python scripts/pi0/run_pi0_remote.py --server ws://127.0.0.1:8000 --env-id TurnGlobeValve-v1 --obs-mode rgb ...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


def _maybe_save_png(path: Path, img: np.ndarray) -> None:
    try:
        from PIL import Image
    except Exception:
        # No extra dependency: just skip saving.
        return
    Image.fromarray(img).save(path)


def _maybe_open_video_writers(out_dir: Path, *, views: str, fps: int):
    """
    Returns dict(view_name -> imageio writer). If imageio isn't installed, returns {} and prints a hint.
    """
    if not views:
        return {}
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        print("video: imageio not installed. Install one of:")
        print("  pip install imageio[ffmpeg]")
        print("  pip install imageio-ffmpeg")
        return {}

    writers = {}
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    def _open(name: str):
        return imageio.get_writer((videos_dir / f"{name}.mp4").as_posix(), fps=fps)

    if views in ("base", "both"):
        writers["base"] = _open("base")
    if views in ("wrist", "both"):
        writers["wrist"] = _open("wrist")
    return writers


def _close_video_writers(writers: dict) -> None:
    for w in writers.values():
        try:
            w.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=str, required=True, help="e.g. ws://127.0.0.1:8000")

    p.add_argument("--env-id", type=str, default="TurnGlobeValve-v1")
    p.add_argument("--obs-mode", type=str, default="rgb")
    p.add_argument("--reward-mode", type=str, default="none")
    p.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    p.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Optional ManiSkill render_mode (e.g. 'human'). Requires a display if using 'human'.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)

    # Where to read images/state from the ManiSkill obs dict (keypaths from inspect_obs output).
    p.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    p.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    p.add_argument(
        "--state-keys",
        type=str,
        nargs="+",
        default=None,
        help="One or more obs keypaths to concat into a 1D state vector.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="If unset, uses env.DEFAULT_TASK_PROMPT.",
    )

    # Logging / outputs
    p.add_argument("--output-root", type=str, default="outputs/pi0")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--image-every", type=int, default=1)
    p.add_argument("--save-video", action="store_true", help="Write mp4 videos (requires imageio + ffmpeg)")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument(
        "--video-views",
        type=str,
        default="base",
        choices=["base", "wrist", "both"],
        help="Which camera views to write into mp4(s).",
    )
    p.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save trajectory.npz (actions/tcp/prompt/env_id). Off by default to keep rollout side-effect free.",
    )
    args = p.parse_args()

    # Allow running without `pip install -e .` by adding repo/src to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.openpi_bridge.obs_to_openpi import (
        ObsAdapter,
        _as_numpy,
        _squeeze_leading_batch,
        _to_uint8_hwc,
    )
    from maniskill_myws.openpi_bridge.remote_policy import RemoteWebsocketChunkPolicy
    from maniskill_myws.openpi_bridge.keypath import get_by_path

    maniskill_myws.register()

    should_save_any = bool(args.save_images or args.save_video or args.save_trajectory)
    out_dir: Path | None = None
    if should_save_any:
        out_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save_images:
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
        if args.save_video:
            (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    obs, info = env.reset(seed=args.seed)

    prompt = args.prompt
    if prompt is None:
        if hasattr(env.unwrapped, "DEFAULT_TASK_PROMPT"):
            prompt = env.unwrapped.DEFAULT_TASK_PROMPT
        else:
            raise SystemExit(
                f"Env '{args.env_id}' does not define DEFAULT_TASK_PROMPT; "
                "please pass --prompt explicitly."
            )

    if args.state_keys is not None and len(args.state_keys) > 0:
        state_keys = args.state_keys
    else:
        state_keys = ["extra/tcp_pose"]

    adapter = ObsAdapter(
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=state_keys,
        prompt=prompt,
    )
    policy = RemoteWebsocketChunkPolicy(server=args.server, obs_adapter=adapter, act_dim=7, resize=224)
    policy.reset()

    traj_actions: list[np.ndarray] = []
    traj_tcp: list[np.ndarray] = []

    video_writers = (
        _maybe_open_video_writers(out_dir, views=args.video_views, fps=args.video_fps)
        if (args.save_video and out_dir is not None)
        else {}
    )

    try:
        for step in range(args.max_steps):
            # Optional realtime rendering (some ManiSkill setups require calling render() explicitly).
            if args.render_mode is not None:
                try:
                    env.render()
                except Exception:
                    pass

            # Save images from obs (client-side render) for quick visualization.
            if (args.save_images or video_writers) and (step % args.image_every == 0):
                base = _to_uint8_hwc(get_by_path(obs, args.image_key))
                wrist = _to_uint8_hwc(get_by_path(obs, args.wrist_image_key))
                if args.save_images:
                    assert out_dir is not None
                    _maybe_save_png(out_dir / "images" / f"base_{step:06d}.png", base)
                    _maybe_save_png(out_dir / "images" / f"wrist_{step:06d}.png", wrist)
                if "base" in video_writers:
                    video_writers["base"].append_data(base)
                if "wrist" in video_writers:
                    video_writers["wrist"].append_data(wrist)

            act = policy.act(obs)
            traj_actions.append(np.asarray(act, dtype=np.float32))

            # Record tcp pose if present
            try:
                tcp_arr = _as_numpy(get_by_path(obs, state_keys[0]))
                tcp_arr = _squeeze_leading_batch(tcp_arr)
                tcp = np.asarray(tcp_arr).reshape(-1).astype(np.float32, copy=False)
                traj_tcp.append(tcp)
            except Exception:
                pass

            obs, rew, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break
    finally:
        _close_video_writers(video_writers)

    if args.save_trajectory and out_dir is not None:
        np.savez_compressed(
            out_dir / "trajectory.npz",
            actions=np.stack(traj_actions) if traj_actions else np.zeros((0, 7), dtype=np.float32),
            tcp=np.stack(traj_tcp) if traj_tcp else np.zeros((0, 0), dtype=np.float32),
            prompt=prompt,
            env_id=args.env_id,
        )
    print(
        "done:",
        dict(
            out_dir=None if out_dir is None else str(out_dir),
            steps=len(traj_actions),
            success=bool(info.get("success", False)),
        ),
    )
    env.close()


if __name__ == "__main__":
    main()


