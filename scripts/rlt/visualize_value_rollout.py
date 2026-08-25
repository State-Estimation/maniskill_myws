#!/usr/bin/env python3
"""Render a base-policy rollout beside live distributional V_base curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maniskill_myws.openpi_bridge.remote_policy import (  # noqa: E402
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.value_model import (  # noqa: E402
    DistributionalBaseValueModel,
    infer_value_estimate,
    value_images_from_observation,
    value_progress_estimate,
)


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _scalar_bool(value: object) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar bool, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--env-id", default="SolarPanelStatic-v2")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--render-mode", choices=("human", "none"), default="human")
    parser.add_argument(
        "--enhanced-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--policy-resize", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=69_000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-backend", default="TkAgg")
    parser.add_argument("--stop-on-plot-close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.policy_resize, args.num_episodes) <= 0:
        parser.error("policy resize and episode count must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0.0 < args.failure_threshold < 1.0:
        parser.error("--failure-threshold must lie in (0,1)")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")
    if args.render_mode == "none" and args.real_time:
        parser.error("headless rendering requires --no-real-time")
    if args.plot and args.plot_backend.lower() in {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }:
        parser.error("--plot requires an interactive Matplotlib backend")
    return args


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"Value checkpoint/runtime mismatch for {name}: "
            f"checkpoint={actual!r}, runtime={expected!r}"
        )


def _validate_runtime_identity(
    *,
    model: DistributionalBaseValueModel,
    checkpoint_metadata: dict[str, Any],
    args: argparse.Namespace,
    prompt: str,
    policy_metadata: dict[str, Any],
    action_dim: int,
    state_dim: int,
    backend: dict[str, object],
) -> dict[str, Any]:
    from maniskill_myws.rlt.policies import openpi_policy_identity_sha256

    dataset = checkpoint_metadata.get("dataset_metadata")
    if not isinstance(dataset, dict):
        raise ValueError("Value checkpoint has no exact dataset_metadata mapping")
    expected = {
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "enhanced_determinism": bool(args.enhanced_determinism),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_keys": [args.image_key, args.wrist_image_key],
        "state_keys": list(args.state_keys),
        "policy_resize": args.policy_resize,
        "chunk_len": model.config.chunk_len,
        "max_episode_steps": model.config.max_episode_steps,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
        "safe_latent_dim": SAFE_LATENT_DIM,
        "openpi_policy_identity_sha256": openpi_policy_identity_sha256(policy_metadata),
        "base_policy_only": True,
    }
    for name, runtime_value in expected.items():
        _require_equal(name, dataset.get(name), runtime_value)
    _require_equal("backend", dataset.get("backend"), backend)
    _require_equal("value_image_size", dataset.get("value_image_size"), model.config.image_height)
    _require_equal("value image shape", model.config.image_width, model.config.image_height)
    _require_equal("model action_dim", model.config.action_dim, action_dim)
    _require_equal("model state_dim", model.config.state_dim, state_dim)
    _require_equal("model num_views", model.config.num_views, 2)
    _require_equal("model latent_dim", model.config.latent_dim, SAFE_LATENT_DIM)
    _require_equal("model latent_protocol", model.config.latent_protocol, SAFE_LATENT_PROTOCOL)
    return dataset


class _LiveValuePlot:
    def __init__(
        self,
        *,
        max_steps: int,
        failure_value: float,
        failure_threshold: float,
        backend: str,
    ) -> None:
        import matplotlib

        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(8.2, 7.4),
            num="V_base live rollout",
        )
        self.failure_axis, self.value_axis, self.progress_axis = axes
        (self.failure_line,) = self.failure_axis.plot(
            [], [], color="#c43c39", marker="o", markersize=3, linewidth=1.8
        )
        self.failure_axis.axhline(
            failure_threshold,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label=f"threshold {failure_threshold:.2f}",
        )
        self.failure_axis.set_ylabel("P(failure)")
        self.failure_axis.set_ylim(-0.03, 1.03)
        self.failure_axis.legend(loc="upper right", frameon=False)

        (self.value_line,) = self.value_axis.plot(
            [], [], color="#20805d", marker="o", markersize=3, linewidth=1.8
        )
        self.value_axis.set_ylabel("Expected value")
        self.value_axis.set_ylim(failure_value - 0.05, 0.05)

        (self.progress_line,) = self.progress_axis.plot(
            [],
            [],
            color="#2563a6",
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="V progress proxy",
        )
        (self.elapsed_line,) = self.progress_axis.plot(
            [],
            [],
            color="#777777",
            linestyle=":",
            linewidth=1.2,
            label="elapsed fraction",
        )
        self.progress_axis.set_ylabel("Completion proximity")
        self.progress_axis.set_xlabel("Environment step")
        self.progress_axis.set_ylim(-0.03, 1.03)
        self.progress_axis.legend(loc="upper right", frameon=False)
        for axis in axes:
            axis.set_xlim(0, max_steps)
            axis.grid(True, alpha=0.22, linewidth=0.8)
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(block=False)
        self._draw()

    @property
    def is_open(self) -> bool:
        return bool(self.plt.fignum_exists(self.figure.number))

    def _draw(self) -> None:
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def update(
        self,
        *,
        seed: int,
        max_steps: int,
        steps: list[int],
        failure: list[float],
        values: list[float],
        progress: list[float],
        status: str = "LIVE",
    ) -> None:
        elapsed = [step / float(max_steps) for step in steps]
        self.failure_line.set_data(steps, failure)
        self.value_line.set_data(steps, values)
        self.progress_line.set_data(steps, progress)
        self.elapsed_line.set_data(steps, elapsed)
        if steps:
            self.figure.suptitle(
                f"seed {seed} | {status} | step {steps[-1]} | "
                f"P(fail) {failure[-1]:.3f} | V {values[-1]:.3f} | "
                f"progress {progress[-1]:.3f}"
            )
        else:
            self.figure.suptitle(f"seed {seed} | {status}")
        self._draw()

    def hold(self, seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while self.is_open and time.perf_counter() < deadline:
            self.plt.pause(min(0.05, max(0.001, deadline - time.perf_counter())))

    def close(self) -> None:
        if self.is_open:
            self.plt.close(self.figure)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite live value traces in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "value_trace.jsonl"
    summary_path = output_dir / "summary.json"

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    value_model, checkpoint_metadata = DistributionalBaseValueModel.load(
        args.value_checkpoint, device=device
    )
    render_mode = None if args.render_mode == "none" else args.render_mode
    maniskill_myws.register()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=render_mode,
        enhanced_determinism=args.enhanced_determinism,
        max_episode_steps=value_model.config.max_episode_steps,
    )
    plot: _LiveValuePlot | None = None
    try:
        backend = require_resolved_backend(
            env,
            expected_sim_backend=args.sim_backend,
            expected_render_backend=args.render_backend,
        )
        prompt = args.prompt or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
        action_dim = int(np.prod(env.action_space.shape))
        policy = make_base_chunk_policy(
            "remote_openpi",
            action_space=env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.policy_resize,
            require_safe_latent=True,
        )
        state_adapter = StateAdapter(args.state_keys)
        probe_obs, _ = reset_env_fresh_scene(
            env, seed=args.seed, operation="live value state shape probe"
        )
        state_dim = int(np.asarray(state_adapter(probe_obs)).size)
        dataset_metadata = _validate_runtime_identity(
            model=value_model,
            checkpoint_metadata=checkpoint_metadata,
            args=args,
            prompt=prompt,
            policy_metadata=policy.server_metadata or {},
            action_dim=action_dim,
            state_dim=state_dim,
            backend=backend,
        )
        print(
            json.dumps(
                {
                    "stage": "value.live_ready",
                    "checkpoint": str(args.value_checkpoint),
                    "checkpoint_epoch": checkpoint_metadata.get("epoch"),
                    "device": str(device),
                    "env_id": args.env_id,
                    "chunk_len": value_model.config.chunk_len,
                    "max_episode_steps": value_model.config.max_episode_steps,
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.plot:
            plot = _LiveValuePlot(
                max_steps=value_model.config.max_episode_steps,
                failure_value=value_model.config.failure_value,
                failure_threshold=args.failure_threshold,
                backend=args.plot_backend,
            )

        control_frequency = float(getattr(env.unwrapped, "control_freq", 20.0))
        target_step_seconds = 1.0 / control_frequency if control_frequency > 0 else 0.05
        summaries: list[dict[str, object]] = []
        user_stopped = False
        with trace_path.open("x", encoding="ascii") as trace_file:
            for episode_index in range(args.num_episodes):
                episode_seed = args.seed + episode_index
                obs, _ = reset_env_fresh_scene(
                    env,
                    seed=episode_seed,
                    operation=f"live value episode {episode_index}",
                )
                policy.reset()
                episode_steps = 0
                episode_success = False
                done = False
                curve_steps: list[int] = []
                curve_failure: list[float] = []
                curve_values: list[float] = []
                curve_progress: list[float] = []
                if plot is not None:
                    plot.update(
                        seed=episode_seed,
                        max_steps=value_model.config.max_episode_steps,
                        steps=curve_steps,
                        failure=curve_failure,
                        values=curve_values,
                        progress=curve_progress,
                    )

                while not done and episode_steps < value_model.config.max_episode_steps:
                    if plot is not None and args.stop_on_plot_close and not plot.is_open:
                        user_stopped = True
                        break
                    ref_chunk, latent = policy.plan_with_latent(
                        obs,
                        chunk_len=value_model.config.chunk_len,
                        action_dim=action_dim,
                        inference_seed=inference_seed_for_step(episode_seed, episode_steps),
                    )
                    latent = np.asarray(latent, dtype=np.float32)
                    if latent.shape != (SAFE_LATENT_DIM,) or not np.all(np.isfinite(latent)):
                        raise ValueError("OpenPI returned an invalid SAFE endpoint latent")
                    raw_state = np.asarray(state_adapter(obs), dtype=np.float32)
                    estimate = infer_value_estimate(
                        value_model,
                        images=value_images_from_observation(
                            obs,
                            image_keys=(args.image_key, args.wrist_image_key),
                            height=value_model.config.image_height,
                            width=value_model.config.image_width,
                        ),
                        state=raw_state,
                        latent=latent,
                        ref_chunk=ref_chunk,
                        step_id=episode_steps,
                    )
                    progress = value_progress_estimate(
                        estimate,
                        max_remaining_chunks=value_model.config.max_remaining_chunks,
                    )
                    curve_steps.append(episode_steps)
                    curve_failure.append(estimate.failure_probability)
                    curve_values.append(estimate.potential)
                    curve_progress.append(progress.completion_proximity)
                    trace = {
                        "episode": episode_index,
                        "seed": episode_seed,
                        "boundary": len(curve_steps) - 1,
                        "step": episode_steps,
                        "failure_probability": estimate.failure_probability,
                        "success_probability": progress.success_probability,
                        "expected_value": estimate.potential,
                        "entropy": estimate.entropy,
                        "expected_remaining_chunks_unconditional": (
                            estimate.expected_remaining_chunks
                        ),
                        "conditional_remaining_chunks": progress.conditional_remaining_chunks,
                        "completion_proximity": progress.completion_proximity,
                        "success_so_far": episode_success,
                    }
                    trace_file.write(json.dumps(trace, sort_keys=True, allow_nan=False) + "\n")
                    trace_file.flush()
                    print(
                        json.dumps({"stage": "value.live_boundary", **trace}, sort_keys=True),
                        flush=True,
                    )
                    if plot is not None:
                        plot.update(
                            seed=episode_seed,
                            max_steps=value_model.config.max_episode_steps,
                            steps=curve_steps,
                            failure=curve_failure,
                            values=curve_values,
                            progress=curve_progress,
                        )

                    for action in ref_chunk:
                        wall_start = time.perf_counter()
                        obs, _, terminated, truncated, info = env.step(action)
                        episode_steps += 1
                        if render_mode is not None:
                            env.render()
                        if not isinstance(info, dict) or "success" not in info:
                            raise ValueError("Environment info does not expose success")
                        episode_success |= _scalar_bool(info["success"])
                        done = bool(
                            _scalar_bool(terminated)
                            or _scalar_bool(truncated)
                            or episode_steps >= value_model.config.max_episode_steps
                        )
                        if args.real_time:
                            remaining = target_step_seconds - (time.perf_counter() - wall_start)
                            if remaining > 0:
                                time.sleep(remaining)
                        if done:
                            break

                status = (
                    "STOPPED" if user_stopped else ("SUCCESS" if episode_success else "FAILURE")
                )
                episode_summary: dict[str, object] = {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "success": episode_success,
                    "env_steps": episode_steps,
                    "boundaries": len(curve_steps),
                    "status": status.lower(),
                }
                summaries.append(episode_summary)
                print(
                    json.dumps(
                        {"stage": "value.live_episode_complete", **episode_summary},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if plot is not None and plot.is_open:
                    plot.update(
                        seed=episode_seed,
                        max_steps=value_model.config.max_episode_steps,
                        steps=curve_steps,
                        failure=curve_failure,
                        values=curve_values,
                        progress=curve_progress,
                        status=status,
                    )
                    plot.hold(args.hold_seconds)
                if user_stopped:
                    break

        summary = {
            "schema": "live_distributional_base_value_rollout_v1",
            "checkpoint": str(args.value_checkpoint),
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "dataset_metadata": dataset_metadata,
            "episodes": summaries,
            "completed_episodes": len(summaries),
            "successes": sum(bool(item["success"]) for item in summaries),
            "user_stopped": user_stopped,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
    finally:
        if plot is not None:
            plot.close()
        env.close()


if __name__ == "__main__":
    main()
