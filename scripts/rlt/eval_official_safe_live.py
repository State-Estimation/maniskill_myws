#!/usr/bin/env python
"""Render base-policy rollouts with a live official-SAFE failure curve."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]
SAFE_LATENT_DIM = 4096


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--threshold-summary",
        default=None,
        help="gate_eval_summary_v2.json; defaults to the checkpoint directory",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--env-id", default="TakeSafetyHook-v1")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=52000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--enhanced-determinism", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="run for visualization even if the task source differs from training",
    )
    args = parser.parse_args()
    if min(
        args.resize,
        args.chunk_len,
        args.max_episode_steps,
        args.num_episodes,
    ) <= 0:
        parser.error("resize, chunk length, horizon, and episode count must be positive")
    if args.start_seed < 0:
        parser.error("start seed must be non-negative")
    if args.hold_seconds < 0:
        parser.error("hold seconds must be non-negative")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    return args


def _scalar_bool(value: object) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar value, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def _load_threshold(args: argparse.Namespace, checkpoint: Path) -> tuple[float, Path | None]:
    if args.threshold is not None:
        return float(args.threshold), None
    summary = (
        Path(args.threshold_summary)
        if args.threshold_summary is not None
        else checkpoint.with_name("gate_eval_summary_v2.json")
    )
    if not summary.is_file():
        raise FileNotFoundError(
            f"Missing calibrated threshold summary {summary}; pass --threshold explicitly"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    threshold = float(payload["calibrated_operating_point"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid calibrated threshold {threshold} in {summary}")
    return threshold, summary


def _load_training_metadata(checkpoint: Path) -> dict[str, Any] | None:
    config_path = checkpoint.with_name("config.yaml")
    if not config_path.is_file():
        return None
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = config.get("dataset", {}).get("data_path")
    if not dataset_path:
        return None
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        return None
    with np.load(dataset_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            return None
        return json.loads(str(archive["metadata_json"].item()))


def _environment_source_identity(env: Any) -> str:
    source_path = inspect.getsourcefile(type(env.unwrapped))
    if source_path is None:
        raise RuntimeError("Cannot locate environment source")
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def _check_identity(
    *,
    args: argparse.Namespace,
    env: Any,
    policy_metadata: dict[str, Any],
    training_metadata: dict[str, Any] | None,
) -> None:
    if training_metadata is None:
        warnings.warn("SAFE training metadata unavailable; runtime identity is unchecked")
        return
    from maniskill_myws.rlt.policies import openpi_policy_identity_sha256

    mismatches = []
    expected_source = training_metadata.get("environment_module_source_sha256")
    actual_source = _environment_source_identity(env)
    if expected_source != actual_source:
        mismatches.append(f"environment source {actual_source} != training {expected_source}")
    expected_policy = training_metadata.get("openpi_policy_identity_sha256")
    actual_policy = openpi_policy_identity_sha256(policy_metadata)
    if expected_policy != actual_policy:
        mismatches.append(f"OpenPI policy {actual_policy} != training {expected_policy}")
    for key in (
        "env_id",
        "obs_mode",
        "reward_mode",
        "control_mode",
        "chunk_len",
        "max_episode_steps",
    ):
        actual = getattr(args, key)
        expected = training_metadata.get(key)
        if expected != actual:
            mismatches.append(f"{key}={actual!r} != training {expected!r}")
    if not mismatches:
        return
    message = "SAFE runtime identity mismatch: " + "; ".join(mismatches)
    if not args.allow_environment_mismatch:
        raise RuntimeError(message + "; pass --allow-environment-mismatch for visualization only")
    warnings.warn(message + "; scores are diagnostic and not a valid accuracy evaluation")


class _OfficialSafeLstm:
    """Dependency-light loader for the official SAFE LSTM checkpoint."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        input_dim = int(state["lstm.weight_ih_l0"].shape[1])
        hidden_dim = int(state["lstm.weight_hh_l0"].shape[1])
        if input_dim != SAFE_LATENT_DIM:
            raise ValueError(f"SAFE checkpoint input dim {input_dim} != {SAFE_LATENT_DIM}")
        self._torch = torch
        self._device = torch.device(device)
        self._lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self._fc = torch.nn.Linear(hidden_dim, 1)
        self._lstm.load_state_dict(
            {
                key.removeprefix("lstm."): value
                for key, value in state.items()
                if key.startswith("lstm.")
            }
        )
        self._fc.load_state_dict(
            {
                key.removeprefix("fc."): value
                for key, value in state.items()
                if key.startswith("fc.")
            }
        )
        self._lstm.to(self._device).eval()
        self._fc.to(self._device).eval()

    def score(self, history: list[np.ndarray]) -> float:
        tensor = self._torch.from_numpy(np.stack(history)[None]).to(self._device)
        with self._torch.inference_mode():
            output, _ = self._lstm(tensor)
            score = self._torch.sigmoid(self._fc(output[:, -1])).item()
        if not np.isfinite(score):
            raise ValueError("SAFE detector produced a non-finite score")
        return float(score)


class _LiveCurve:
    def __init__(self, *, threshold: float, max_steps: int) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(8.5, 4.2))
        try:
            self.figure.canvas.manager.set_window_title("SAFE failure probability")
        except AttributeError:
            pass
        self.axes.set_xlim(0, max_steps)
        self.axes.set_ylim(-0.03, 1.03)
        self.axes.set_xlabel("Environment step (chunk boundary)")
        self.axes.set_ylabel("Failure probability")
        self.axes.grid(True, alpha=0.25)
        self.axes.axhline(threshold, color="#d62728", linestyle="--", label=f"gate {threshold:.3f}")
        (self.line,) = self.axes.plot(
            [], [], color="#1f77b4", marker="o", linewidth=2, label="SAFE"
        )
        self.alerts = self.axes.scatter(
            [], [], color="#d62728", marker="x", s=90, label="alert", zorder=3
        )
        self.axes.legend(loc="upper left")
        self.figure.tight_layout()
        plt.show(block=False)
        self.reset(seed=0)

    def reset(self, *, seed: int) -> None:
        self.steps: list[int] = []
        self.scores: list[float] = []
        self.alert_steps: list[int] = []
        self.alert_scores: list[float] = []
        self.line.set_data([], [])
        self.alerts.set_offsets(np.empty((0, 2)))
        self.axes.set_title(f"seed {seed} | waiting for first chunk")
        self._draw()

    def update(self, *, seed: int, step: int, score: float, alert: bool) -> None:
        self.steps.append(step)
        self.scores.append(score)
        if alert:
            self.alert_steps.append(step)
            self.alert_scores.append(score)
        self.line.set_data(self.steps, self.scores)
        offsets = (
            np.column_stack([self.alert_steps, self.alert_scores])
            if self.alert_steps
            else np.empty((0, 2))
        )
        self.alerts.set_offsets(offsets)
        state = "ALERT" if alert else "normal"
        self.axes.set_title(f"seed {seed} | step {step} | p(fail)={score:.3f} | {state}")
        self._draw()

    def finish(self, *, seed: int, success: bool, steps: int) -> None:
        result = "SUCCESS" if success else "FAILURE"
        self.axes.set_title(f"seed {seed} | {result} at step {steps}")
        self._draw()

    def _draw(self) -> None:
        if not self._plt.fignum_exists(self.figure.number):
            raise RuntimeError("SAFE curve window was closed")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self._plt.pause(0.001)

    def pause(self, seconds: float) -> None:
        if seconds > 0:
            self._plt.pause(seconds)


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAFE checkpoint not found: {checkpoint}")
    threshold, threshold_summary = _load_threshold(args, checkpoint)
    training_metadata = _load_training_metadata(checkpoint)

    import gymnasium as gym
    import maniskill_myws
    from maniskill_myws.openpi_bridge.remote_policy import SAFE_LATENT_DIM as BRIDGE_SAFE_DIM
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.policies import inference_seed_for_step, make_base_chunk_policy
    from maniskill_myws.rlt.reset import reset_env_fresh_scene

    if BRIDGE_SAFE_DIM != SAFE_LATENT_DIM:
        raise RuntimeError("Live evaluator and OpenPI bridge disagree on SAFE latent dim")
    maniskill_myws.register()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode="human",
        enhanced_determinism=args.enhanced_determinism,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        backend = require_resolved_backend(
            env,
            expected_sim_backend=args.sim_backend,
            expected_render_backend=args.render_backend,
        )
        action_dim = int(np.prod(env.action_space.shape))
        prompt = args.prompt or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
        if not prompt:
            raise ValueError("The environment has no default prompt; pass --prompt")
        policy = make_base_chunk_policy(
            "remote_openpi",
            action_space=env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.resize,
            require_safe_latent=True,
        )
        _check_identity(
            args=args,
            env=env,
            policy_metadata=policy.server_metadata or {},
            training_metadata=training_metadata,
        )
        detector = _OfficialSafeLstm(checkpoint, args.detector_device)
        curve = _LiveCurve(threshold=threshold, max_steps=args.max_episode_steps)
        control_frequency = float(getattr(env.unwrapped, "control_freq", 20.0))
        target_step_seconds = 1.0 / control_frequency if control_frequency > 0 else 0.05
        print(
            json.dumps(
                {
                    "stage": "safe.live_ready",
                    "checkpoint": str(checkpoint),
                    "threshold": threshold,
                    "threshold_summary": (
                        None if threshold_summary is None else str(threshold_summary)
                    ),
                    "backend": backend,
                    "mode": "shadow_only",
                },
                sort_keys=True,
            ),
            flush=True,
        )

        for episode_index in range(args.num_episodes):
            seed = args.start_seed + episode_index
            obs, _ = reset_env_fresh_scene(
                env, seed=seed, operation=f"SAFE live episode {episode_index}"
            )
            policy.reset()
            curve.reset(seed=seed)
            history: list[np.ndarray] = []
            episode_steps = 0
            episode_success = False
            done = False
            while not done and episode_steps < args.max_episode_steps:
                action_chunk, latent = policy.plan_with_latent(
                    obs,
                    chunk_len=args.chunk_len,
                    action_dim=action_dim,
                    inference_seed=inference_seed_for_step(seed, episode_steps),
                )
                latent = np.asarray(latent, dtype=np.float32)
                if latent.shape != (SAFE_LATENT_DIM,) or not np.all(np.isfinite(latent)):
                    raise ValueError(f"Invalid SAFE latent shape/value: {latent.shape}")
                history.append(latent.copy())
                score = detector.score(history)
                alert = score >= threshold
                curve.update(seed=seed, step=episode_steps, score=score, alert=alert)
                print(
                    json.dumps(
                        {
                            "stage": "safe.live_chunk",
                            "seed": seed,
                            "step": episode_steps,
                            "chunk": len(history) - 1,
                            "failure_probability": score,
                            "threshold": threshold,
                            "alert": alert,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                for action in np.asarray(action_chunk, dtype=np.float32):
                    wall_start = time.perf_counter()
                    obs, _, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    env.render()
                    if not isinstance(info, dict) or "success" not in info:
                        raise ValueError("Environment info does not contain success")
                    episode_success |= _scalar_bool(info["success"])
                    done = bool(
                        _scalar_bool(terminated)
                        or _scalar_bool(truncated)
                        or episode_steps >= args.max_episode_steps
                    )
                    if args.real_time:
                        remaining = target_step_seconds - (time.perf_counter() - wall_start)
                        if remaining > 0:
                            time.sleep(remaining)
                    if done:
                        break
            curve.finish(seed=seed, success=episode_success, steps=episode_steps)
            print(
                json.dumps(
                    {
                        "stage": "safe.live_episode",
                        "seed": seed,
                        "success": episode_success,
                        "env_steps": episode_steps,
                        "alerted": any(score >= threshold for score in curve.scores),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            curve.pause(args.hold_seconds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
