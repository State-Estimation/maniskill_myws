#!/usr/bin/env python
"""Evaluate a frozen vision-only progress ensemble during live policy rollouts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _as_scalar(value: Any) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _as_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


class VisualClipBuffer:
    def __init__(self, *, image_adapter, context_frames: int, context_stride: int) -> None:
        self.image_adapter = image_adapter
        self.context_frames = int(context_frames)
        self.context_stride = int(context_stride)
        self.max_length = (self.context_frames - 1) * self.context_stride + 1
        self.frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.frames.clear()

    def append(self, obs) -> np.ndarray:
        frame = self.image_adapter(obs)
        self.frames.append(frame)
        if len(self.frames) > self.max_length:
            self.frames = self.frames[-self.max_length :]
        indices = [
            len(self.frames) - 1 - offset * self.context_stride
            for offset in range(self.context_frames - 1, -1, -1)
        ]
        clip = [self.frames[max(0, index)] for index in indices]
        return np.ascontiguousarray(np.stack(clip, axis=0), dtype=np.uint8)


def _plot_episode(
    rows: list[dict[str, float]],
    path: Path,
    *,
    audit_enabled: bool,
    live: bool,
    figure_size: tuple[float, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plot warning: matplotlib unavailable: {e}", flush=True)
        return
    if not rows:
        return
    step = np.asarray([row["step"] for row in rows])
    progress = np.asarray([row["progress"] for row in rows])
    smoothed = np.asarray([row["progress_ema"] for row in rows])
    success = np.asarray([row["success_probability"] for row in rows])
    uncertainty = np.asarray([row["progress_uncertainty"] for row in rows])
    success_uncertainty = np.asarray(
        [row["success_uncertainty"] for row in rows]
    )
    latent_change = np.asarray([row["latent_change"] for row in rows])
    embedding_uncertainty = np.asarray(
        [row["embedding_uncertainty"] for row in rows]
    )

    if live:
        plt.ion()
    fig = plt.figure("visual-task-progress", figsize=figure_size)
    fig.clear()
    axes = fig.subplots(3, 1, sharex=True)
    axes[0].plot(step, progress, color="tab:blue", alpha=0.4, label="visual progress")
    axes[0].plot(step, smoothed, color="tab:blue", linewidth=2, label="EMA progress")
    if audit_enabled:
        audit = np.asarray([row["audit_progress"] for row in rows])
        axes[0].plot(step, audit, "k--", linewidth=1.5, label="hidden audit target")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("progress")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(step, success, color="tab:orange", label="eventual success p")
    axes[1].fill_between(
        step,
        np.clip(success - success_uncertainty, 0.0, 1.0),
        np.clip(success + success_uncertainty, 0.0, 1.0),
        color="tab:orange",
        alpha=0.15,
        label="ensemble disagreement",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("success")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")

    axes[2].plot(
        step,
        latent_change,
        color="tab:green",
        label="implicit latent change",
    )
    axes[2].plot(
        step,
        uncertainty,
        color="tab:red",
        alpha=0.8,
        label="progress uncertainty",
    )
    axes[2].plot(
        step,
        embedding_uncertainty,
        color="tab:purple",
        alpha=0.7,
        label="embedding disagreement",
    )
    axes[2].set_ylabel("latent change / uncertainty")
    axes[2].set_xlabel("environment step")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    if live:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="SolarPanelStatic-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--robot-uids", type=str, default="panda_wristcam")
    parser.add_argument("--env-device", type=str, default=None)
    parser.add_argument("--progress-device", type=str, default="cuda:0")
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Optional ManiSkill render mode. Use 'human' for an interactive window.",
    )
    parser.add_argument(
        "--realtime-render",
        action="store_true",
        help="Open a realtime ManiSkill viewer (equivalent to --render-mode human).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Refresh the realtime environment viewer every N environment steps.",
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout-window-steps", type=int, default=50)

    parser.add_argument(
        "--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi"
    )
    parser.add_argument("--server", type=str, default="ws://127.0.0.1:8010")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-width", type=float, default=6.0)
    parser.add_argument("--plot-height", type=float, default=5.0)
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument(
        "--audit-progress-key",
        type=str,
        default="none",
        help=(
            "Optional hidden environment key used only for evaluation, e.g. "
            "extra/clean_coverage. It is never passed to the predictor."
        ),
    )
    parser.add_argument("--audit-progress-start", type=float, default=0.0)
    parser.add_argument("--audit-progress-goal", type=float, default=0.6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pld/SolarPanelStatic-v2/visual_progress_rollout",
    )
    args = parser.parse_args()

    if args.rollout_window_steps <= 0 or args.num_seeds <= 0:
        raise SystemExit("--rollout-window-steps and --num-seeds must be positive")
    if args.render_every <= 0:
        raise SystemExit("--render-every must be positive")
    if args.realtime_render and args.render_mode not in {None, "human"}:
        raise SystemExit(
            "--realtime-render cannot be combined with a non-human --render-mode"
        )
    if not 0.0 < args.ema_alpha <= 1.0:
        raise SystemExit("--ema-alpha must be in (0, 1]")
    if args.plot_width <= 0.0 or args.plot_height <= 0.0:
        raise SystemExit("--plot-width and --plot-height must be positive")
    if args.audit_progress_goal == args.audit_progress_start:
        raise SystemExit("audit progress goal and start must differ")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.pld.env_device import apply_env_device_kwargs
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.state import (
        ImageAdapter,
        as_numpy,
        get_by_path_flexible,
    )
    from maniskill_myws.pld.visual_progress import VisualProgressEnsemble
    from maniskill_myws.task_prompts import get_task_prompt

    maniskill_myws.register()
    ensemble, metadata = VisualProgressEnsemble.load(
        args.checkpoint,
        device=args.progress_device,
    )
    if not metadata.get("visual_only", False):
        raise ValueError("Checkpoint is not marked as a visual-only progress model")
    if metadata.get("env_id") != args.env_id:
        raise ValueError(
            f"Checkpoint env_id {metadata.get('env_id')!r} != {args.env_id!r}"
        )
    image_keys = list(metadata["image_keys"])
    image_adapter = ImageAdapter(
        image_keys,
        image_size=int(metadata["image_size"]),
        image_shape=(
            len(image_keys),
            int(metadata["image_size"]),
            int(metadata["image_size"]),
            3,
        ),
    )
    clip_buffer = VisualClipBuffer(
        image_adapter=image_adapter,
        context_frames=int(metadata["context_frames"]),
        context_stride=int(metadata["context_stride"]),
    )

    render_mode = "human" if args.realtime_render else args.render_mode
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
    )
    if args.robot_uids.lower() not in {"none", "null", ""}:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.max_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_steps
    apply_env_device_kwargs(env_kwargs, args.env_device)
    env = gym.make(args.env_id, **env_kwargs)
    max_steps = args.max_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    max_steps = int(max_steps or 500)

    prompt = args.prompt or get_task_prompt(args.env_id) or getattr(
        env.unwrapped, "DEFAULT_TASK_PROMPT", ""
    )
    action_dim = int(np.prod(env.action_space.shape))
    base_policy = make_base_policy(
        args.base_policy,
        action_space=env.action_space,
        action_dim=action_dim,
        server=args.server,
        prompt=prompt,
        image_key=image_keys[0],
        wrist_image_key=image_keys[1] if len(image_keys) > 1 else image_keys[0],
        state_keys=["agent/qpos", "agent/qvel", "extra/tcp_pose"],
        resize=args.resize,
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    audit_key = args.audit_progress_key
    if audit_key.lower() in {"none", "null", ""}:
        audit_key = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    print(
        "visual_progress_rollout",
        dict(
            checkpoint=args.checkpoint,
            ensemble_size=ensemble.ensemble_size,
            visual_only=True,
            image_keys=image_keys,
            context_frames=metadata["context_frames"],
            context_stride=metadata["context_stride"],
            audit_progress_key=audit_key,
            render_mode=render_mode,
            render_every=args.render_every,
            seeds=f"{args.start_seed}:{args.start_seed + args.num_seeds - 1}",
            output_dir=str(output_dir),
        ),
        flush=True,
    )

    try:
        for episode_index in range(args.num_seeds):
            seed = args.start_seed + episode_index
            obs, info = env.reset(seed=seed)
            if render_mode is not None:
                env.render()
            base_policy.reset()
            clip_buffer.reset()
            rows: list[dict[str, float]] = []
            ema_progress: float | None = None
            previous_embedding: np.ndarray | None = None
            step = 0
            success = False
            episode_start = time.perf_counter()

            while step < max_steps:
                requested = min(args.rollout_window_steps, max_steps - step)
                action_window = np.asarray(
                    base_policy.plan_window(obs, window_steps=requested),
                    dtype=np.float32,
                )
                expected_shape = (requested, action_dim)
                if action_window.shape != expected_shape:
                    raise ValueError(
                        f"Policy action window {action_window.shape} != {expected_shape}"
                    )
                action_window = np.clip(
                    action_window, action_low[None], action_high[None]
                ).astype(np.float32)

                done = False
                for action in action_window:
                    clip = clip_buffer.append(obs)
                    prediction = ensemble.predict(clip)
                    progress = float(prediction["progress"])
                    embedding = np.asarray(
                        prediction["embedding"], dtype=np.float32
                    )
                    latent_change = (
                        0.0
                        if previous_embedding is None
                        else float(np.linalg.norm(embedding - previous_embedding))
                    )
                    previous_embedding = embedding
                    ema_progress = (
                        progress
                        if ema_progress is None
                        else args.ema_alpha * progress
                        + (1.0 - args.ema_alpha) * ema_progress
                    )
                    audit_progress = float("nan")
                    if audit_key is not None:
                        raw_audit = _as_scalar(get_by_path_flexible(obs, audit_key))
                        audit_progress = float(
                            np.clip(
                                (raw_audit - args.audit_progress_start)
                                / (args.audit_progress_goal - args.audit_progress_start),
                                0.0,
                                1.0,
                            )
                        )
                    row: dict[str, float] = {
                        "step": float(step),
                        "progress": progress,
                        "progress_ema": float(ema_progress),
                        "success_probability": float(
                            prediction["success_probability"]
                        ),
                        "progress_uncertainty": float(
                            prediction["progress_uncertainty"]
                        ),
                        "success_uncertainty": float(
                            prediction["success_uncertainty"]
                        ),
                        "embedding_uncertainty": float(
                            prediction["embedding_uncertainty"]
                        ),
                        "latent_change": latent_change,
                        "audit_progress": audit_progress,
                    }
                    rows.append(row)

                    if step % max(1, args.log_every) == 0:
                        print(
                            "progress_step",
                            dict(
                                seed=seed,
                                step=step,
                                progress=round(progress, 4),
                                ema=round(float(ema_progress), 4),
                                success_p=round(
                                    float(prediction["success_probability"]), 4
                                ),
                                uncertainty=round(
                                    float(prediction["progress_uncertainty"]), 4
                                ),
                                latent_change=round(latent_change, 4),
                                audit=(
                                    None
                                    if not np.isfinite(audit_progress)
                                    else round(audit_progress, 4)
                                ),
                            ),
                            flush=True,
                        )
                    if args.live_plot and step % max(1, args.plot_every) == 0:
                        _plot_episode(
                            rows,
                            output_dir / f"seed_{seed:05d}" / "progress_curve.png",
                            audit_enabled=audit_key is not None,
                            live=True,
                            figure_size=(args.plot_width, args.plot_height),
                        )
                    obs, _, terminated, truncated, info = env.step(action)
                    step += 1
                    if render_mode is not None and step % args.render_every == 0:
                        env.render()
                    success = (
                        bool(_as_bool(info.get("success", False)))
                        if isinstance(info, dict)
                        else False
                    )
                    done = _as_bool(terminated) or _as_bool(truncated)
                    if done or step >= max_steps:
                        break
                if done:
                    break

            seed_dir = output_dir / f"seed_{seed:05d}"
            _write_csv(seed_dir / "progress.csv", rows)
            _plot_episode(
                rows,
                seed_dir / "progress_curve.png",
                audit_enabled=audit_key is not None,
                live=False,
                figure_size=(args.plot_width, args.plot_height),
            )
            predicted = np.asarray([row["progress"] for row in rows])
            summary: dict[str, Any] = {
                "seed": seed,
                "success": success,
                "steps": step,
                "elapsed_s": time.perf_counter() - episode_start,
                "start_progress": float(predicted[0]) if predicted.size else None,
                "terminal_progress": float(predicted[-1]) if predicted.size else None,
                "monotonic_violation_rate": (
                    float((np.diff(predicted) < -1e-3).mean())
                    if predicted.size > 1
                    else 0.0
                ),
            }
            if audit_key is not None and rows:
                audit = np.asarray([row["audit_progress"] for row in rows])
                summary["audit_mae"] = float(np.abs(predicted - audit).mean())
                summary["audit_terminal"] = float(audit[-1])
            summaries.append(summary)
            (seed_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print("progress_episode", summary, flush=True)
    finally:
        env.close()

    aggregate = {
        "episodes": len(summaries),
        "success_rate": float(
            np.mean([summary["success"] for summary in summaries])
            if summaries
            else 0.0
        ),
        "mean_terminal_progress": float(
            np.mean(
                [
                    summary["terminal_progress"]
                    for summary in summaries
                    if summary["terminal_progress"] is not None
                ]
            )
            if summaries
            else 0.0
        ),
        "mean_monotonic_violation_rate": float(
            np.mean([summary["monotonic_violation_rate"] for summary in summaries])
            if summaries
            else 0.0
        ),
        "runs": summaries,
    }
    if audit_key is not None and summaries:
        aggregate["mean_audit_mae"] = float(
            np.mean([summary["audit_mae"] for summary in summaries])
        )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print("visual_progress_summary", aggregate, flush=True)


if __name__ == "__main__":
    main()
