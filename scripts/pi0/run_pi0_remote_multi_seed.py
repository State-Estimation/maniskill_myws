#!/usr/bin/env python
"""Fast multi-seed evaluation for a remote OpenPI policy.

Unlike the old wrapper, this keeps one ManiSkill environment, one viewer window,
and one websocket policy client alive while it loops over seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_save_png(path: Path, img: np.ndarray) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    Image.fromarray(img).save(path)


def _maybe_open_video_writers(out_dir: Path, *, views: str, fps: int) -> dict:
    if not views:
        return {}
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        print("video: imageio not installed. Install one of:")
        print("  pip install imageio[ffmpeg]")
        print("  pip install imageio-ffmpeg")
        return {}

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    if views in ("base", "both"):
        writers["base"] = imageio.get_writer(
            (videos_dir / "base.mp4").as_posix(),
            fps=fps,
        )
    if views in ("wrist", "both"):
        writers["wrist"] = imageio.get_writer(
            (videos_dir / "wrist.mp4").as_posix(),
            fps=fps,
        )
    return writers


def _close_video_writers(writers: dict) -> None:
    for writer in writers.values():
        try:
            writer.close()
        except Exception:
            pass


def _as_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _as_done(x) -> bool:
    return bool(_as_numpy(x).reshape(-1)[0])


def _as_scalar(x) -> float:
    return float(_as_numpy(x).reshape(-1)[0])


def _infer_control_dt(env, fallback_hz: float = 20.0) -> tuple[float, float]:
    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = fallback_hz
    control_freq = float(control_freq)
    if control_freq <= 0:
        control_freq = fallback_hz
    return 1.0 / control_freq, control_freq


def _default_max_steps(env, fallback: int = 200) -> int:
    max_steps = getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    return int(max_steps or fallback)


def _seed_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / f"seed_{seed:03d}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run remote OpenPI policy inference with multiple seeds."
    )

    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--server", type=str, required=True, help="e.g. ws://127.0.0.1:8000")
    p.add_argument("--env-id", type=str, default="StackCube-v2")
    p.add_argument("--obs-mode", type=str, default="rgb")
    p.add_argument("--reward-mode", type=str, default="none")
    p.add_argument("--control-mode", type=str, default="pd_joint_pos")
    p.add_argument("--render-mode", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--real-time", action="store_true")
    p.add_argument(
        "--execution-chunk-size",
        type=int,
        default=None,
        help=(
            "Execute only this many actions from each predicted chunk before "
            "replanning. By default the full policy action horizon is executed."
        ),
    )

    p.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    p.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    p.add_argument("--state-keys", type=str, nargs="+", default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--resize", type=int, default=224)

    p.add_argument("--save-videos", "--save-video", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--save-trajectories", "--save-trajectory", action="store_true")
    p.add_argument("--image-every", type=int, default=1)
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument(
        "--video-views",
        type=str,
        default="base",
        choices=["base", "wrist", "both"],
    )
    p.add_argument("--tcp-pose-key", type=str, default="extra/tcp_pose")
    p.add_argument("--output-root", type=str, default="outputs/pi0_multi_seed")
    p.add_argument("--output-name", type=str, default=None)

    # Accepted for backward compatibility with older commands. The old wrapper
    # only forwarded these flags to the single-seed script; the fast runner does
    # not draw path markers.
    p.add_argument("--visualize-tcp-path", action="store_true")
    p.add_argument("--path-every", type=int, default=2)
    p.add_argument("--path-max-points", type=int, default=500)
    p.add_argument("--path-radius", type=float, default=0.008)
    p.add_argument("--base-chunk-max-actions", type=int, default=16)
    p.add_argument("--base-chunk-position-scale", type=float, default=0.1)
    p.add_argument("--base-path-color", type=str, default="0.05,0.35,1.0,1.0")
    p.add_argument("--residual-path-color", type=str, default="1.0,0.28,0.02,1.0")

    args = p.parse_args()

    sys.path.insert(0, str(_repo_root() / "src"))

    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.openpi_bridge.keypath import get_by_path
    from maniskill_myws.openpi_bridge.obs_to_openpi import (
        ObsAdapter,
        _as_numpy as obs_as_numpy,
        _squeeze_leading_batch,
        _to_uint8_hwc,
    )
    from maniskill_myws.openpi_bridge.remote_policy import RemoteWebsocketChunkPolicy

    maniskill_myws.register()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"{args.env_id}_{timestamp}"
    output_dir = Path(args.output_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    if args.max_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_steps

    env = gym.make(args.env_id, **env_kwargs)
    max_steps = args.max_steps or _default_max_steps(env)

    real_time_dt = None
    if args.real_time:
        real_time_dt, control_freq = _infer_control_dt(env)
        print(
            "real-time pacing:",
            dict(control_freq_hz=control_freq, target_dt_s=round(real_time_dt, 4)),
            flush=True,
        )

    prompt = args.prompt
    if prompt is None:
        prompt = getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    if not prompt:
        raise SystemExit(
            f"Env '{args.env_id}' does not define DEFAULT_TASK_PROMPT; "
            "please pass --prompt explicitly."
        )

    state_keys = args.state_keys if args.state_keys else DEFAULT_STATE_KEYS
    adapter = ObsAdapter(
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=state_keys,
        prompt=prompt,
    )
    action_dim = int(np.prod(env.action_space.shape))
    policy = RemoteWebsocketChunkPolicy(
        server=args.server,
        obs_adapter=adapter,
        act_dim=action_dim,
        resize=args.resize,
        execution_chunk_size=args.execution_chunk_size,
    )

    if args.visualize_tcp_path:
        print(
            "warning: --visualize-tcp-path is accepted for compatibility, "
            "but path markers are not drawn by the fast multi-seed runner.",
            flush=True,
        )

    print("=== Multi-Seed Evaluation ===", flush=True)
    print(f"Environment: {args.env_id}", flush=True)
    print(f"Server: {args.server}", flush=True)
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}", flush=True)
    print(f"Max steps: {max_steps}", flush=True)
    print(
        "Execution chunk size: "
        f"{args.execution_chunk_size or 'full policy horizon'}",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    print("", flush=True)

    results: list[dict[str, object]] = []
    success_count = 0
    total_steps = 0

    try:
        for i in range(args.num_seeds):
            seed = args.start_seed + i
            obs, info = env.reset(seed=seed)
            policy.reset()

            should_save = bool(
                args.save_images or args.save_videos or args.save_trajectories
            )
            seed_dir = _seed_dir(output_dir, seed)
            if should_save:
                seed_dir.mkdir(parents=True, exist_ok=True)
            if args.save_images:
                (seed_dir / "images").mkdir(parents=True, exist_ok=True)

            video_writers = (
                _maybe_open_video_writers(
                    seed_dir,
                    views=args.video_views,
                    fps=args.video_fps,
                )
                if args.save_videos
                else {}
            )

            traj_actions: list[np.ndarray] = []
            traj_tcp: list[np.ndarray] = []
            episode_return = 0.0
            done = False

            print(
                f"[{i + 1}/{args.num_seeds}] Running seed {seed}...",
                end=" ",
                flush=True,
            )

            try:
                for step in range(max_steps):
                    step_wall_start = time.perf_counter()
                    action = policy.act(obs)
                    traj_actions.append(np.asarray(action, dtype=np.float32))

                    if args.render_mode is not None:
                        try:
                            env.render()
                        except Exception:
                            pass

                    if (
                        args.save_images or video_writers
                    ) and step % max(1, args.image_every) == 0:
                        base = _to_uint8_hwc(get_by_path(obs, args.image_key))
                        wrist = _to_uint8_hwc(get_by_path(obs, args.wrist_image_key))
                        if args.save_images:
                            _maybe_save_png(
                                seed_dir / "images" / f"base_{step:06d}.png",
                                base,
                            )
                            _maybe_save_png(
                                seed_dir / "images" / f"wrist_{step:06d}.png",
                                wrist,
                            )
                        if "base" in video_writers:
                            video_writers["base"].append_data(base)
                        if "wrist" in video_writers:
                            video_writers["wrist"].append_data(wrist)

                    try:
                        tcp_arr = obs_as_numpy(get_by_path(obs, args.tcp_pose_key))
                        tcp_arr = _squeeze_leading_batch(tcp_arr)
                        traj_tcp.append(
                            np.asarray(tcp_arr, dtype=np.float32).reshape(-1)
                        )
                    except Exception:
                        pass

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += _as_scalar(reward)
                    done = _as_done(terminated) or _as_done(truncated)
                    if done:
                        break

                    if real_time_dt is not None:
                        sleep_s = real_time_dt - (time.perf_counter() - step_wall_start)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
            finally:
                _close_video_writers(video_writers)

            if args.save_trajectories:
                np.savez_compressed(
                    seed_dir / "trajectory.npz",
                    actions=(
                        np.stack(traj_actions)
                        if traj_actions
                        else np.zeros((0, action_dim), dtype=np.float32)
                    ),
                    tcp=(
                        np.stack(traj_tcp)
                        if traj_tcp
                        else np.zeros((0, 0), dtype=np.float32)
                    ),
                    prompt=prompt,
                    env_id=args.env_id,
                    seed=seed,
                )

            steps = len(traj_actions)
            total_steps += steps
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if success:
                success_count += 1
            results.append(
                {
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "return": episode_return,
                }
            )

            mark = "✓" if success else "✗"
            label = "Success" if success else "Failed"
            print(f"{mark} {label} ({steps} steps, return={episode_return:.4f})")
    finally:
        env.close()

    success_rate = 100.0 * success_count / max(1, args.num_seeds)
    avg_steps = total_steps / max(1, args.num_seeds)
    avg_return = float(np.mean([float(r["return"]) for r in results])) if results else 0.0

    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"Total runs: {args.num_seeds}", flush=True)
    print(f"Success: {success_count}", flush=True)
    print(f"Failure: {args.num_seeds - success_count}", flush=True)
    print(f"Success rate: {success_rate:.1f}%", flush=True)
    print(f"Average steps: {avg_steps:.1f}", flush=True)
    print(f"Average return: {avg_return:.4f}", flush=True)

    results_file = output_dir / "results.txt"
    with results_file.open("w", encoding="utf-8") as f:
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Server: {args.server}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Max steps: {max_steps}\n")
        f.write(
            "Execution chunk size: "
            f"{args.execution_chunk_size or 'full policy horizon'}\n"
        )
        f.write(f"Prompt: {prompt}\n")
        f.write("\nResults:\n")
        for r in results:
            f.write(
                f"  Seed {r['seed']}: {'✓' if r['success'] else '✗'} "
                f"({r['steps']} steps, return={float(r['return']):.4f})\n"
            )
        f.write("\nSummary:\n")
        f.write(f"  Success rate: {success_rate:.1f}%\n")
        f.write(f"  Average steps: {avg_steps:.1f}\n")
        f.write(f"  Average return: {avg_return:.4f}\n")

    print(f"\nResults saved to: {results_file}", flush=True)


if __name__ == "__main__":
    main()
