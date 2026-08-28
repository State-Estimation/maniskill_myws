#!/usr/bin/env python
"""Train V-gated candidate residuals with a one-step value-improvement bandit."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]
TRAINER_STATE_SCHEMA = "value_guided_bandit_trainer_state_v2"


def _scalar(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _done(value) -> bool:
    return bool(_scalar(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_bounds(action_space, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError("Environment action bounds do not match action_dim")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Environment action bounds must be finite")
    if np.any(low >= high):
        raise ValueError("Environment action bounds require low < high")
    return low, high


def _validate(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}, got {array.shape}")
    return array


def _validate_environment_reward(
    *,
    reward_mode: str,
    reward_value: float,
    info: dict,
    process_reward: float | None,
    success_reward: float,
) -> tuple[bool, bool]:
    if not np.isfinite(reward_value):
        raise ValueError("Environment reward is NaN or Inf")
    if not isinstance(info, dict) or "success" not in info:
        raise ValueError("Environment info must contain success")
    success = _done(info["success"])
    process_event = _done(info.get("grasp_reward_event", False))
    expected = success_reward * float(success)
    if process_reward is not None:
        if "grasp_reward_event" not in info:
            raise ValueError("Milestone reward info is missing grasp_reward_event")
        expected += process_reward * float(process_event)
    else:
        process_event = False
    if reward_mode not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported reward mode {reward_mode!r}")
    if not np.isclose(reward_value, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Environment reward {reward_value} does not match declared components {expected}"
        )
    return success, process_event


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _encode_torch_rng_state(state) -> str:
    values = state.detach().cpu().numpy().astype(np.uint8, copy=False)
    return base64.b64encode(values.tobytes()).decode("ascii")


def _decode_torch_rng_state(encoded: str, torch):
    try:
        values = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Trainer state has an invalid PyTorch RNG payload") from exc
    return torch.from_numpy(values)


def _capture_torch_rng_state(torch) -> dict:
    payload = {"cpu": _encode_torch_rng_state(torch.get_rng_state())}
    if torch.cuda.is_available():
        payload["cuda"] = [
            _encode_torch_rng_state(state) for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _restore_torch_rng_state(payload: dict, torch) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("cpu"), str):
        raise ValueError("Trainer state is missing the CPU PyTorch RNG state")
    torch.set_rng_state(_decode_torch_rng_state(payload["cpu"], torch))
    saved_cuda = payload.get("cuda")
    if saved_cuda is None:
        return
    if not isinstance(saved_cuda, list) or not torch.cuda.is_available():
        raise ValueError("Trainer CUDA RNG state cannot be restored on this runtime")
    if len(saved_cuda) != torch.cuda.device_count():
        raise ValueError("Trainer CUDA device count differs from the saved run")
    torch.cuda.set_rng_state_all(
        [_decode_torch_rng_state(str(state), torch) for state in saved_cuda]
    )


def _load_trainer_state(path: Path, *, rng: np.random.Generator, torch) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Resume trainer state not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TRAINER_STATE_SCHEMA:
        raise ValueError("Unsupported value-guided bandit trainer state")
    required = (
        "snapshot_id",
        "env_steps",
        "episode",
        "last_save",
        "recent_successes",
        "trainer_rng_state",
        "torch_rng_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Trainer state is missing fields: {missing}")
    snapshot_id = payload["snapshot_id"]
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Trainer state has no snapshot generation id")
    rng.bit_generator.state = payload["trainer_rng_state"]
    _restore_torch_rng_state(payload["torch_rng_state"], torch)
    return {
        "snapshot_id": snapshot_id,
        "env_steps": int(payload["env_steps"]),
        "episode": int(payload["episode"]),
        "last_save": int(payload["last_save"]),
        "recent_successes": [int(bool(value)) for value in payload["recent_successes"]][-50:],
        "post_rollout_actor_updates_completed": int(
            payload.get("post_rollout_actor_updates_completed", 0)
        ),
        "wandb_run_id": payload.get("wandb_run_id"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SolarPanelStatic-v2")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument(
        "--enhanced-determinism", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument(
        "--scorer-target",
        choices=("value_improvement",),
        default="value_improvement",
        help="Frozen V-base improvement target learned by the residual scorer",
    )
    parser.add_argument(
        "--scorer-return-mode",
        choices=("one_step", "episode_mc", "bounded_reward_trace"),
        default="one_step",
    )
    parser.add_argument("--scorer-return-gamma", type=float, default=0.99)
    parser.add_argument("--scorer-trace-chunks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-chunk", type=int, default=5)
    parser.add_argument("--warmup-transitions", type=int, default=256)
    parser.add_argument("--actor-warmup-transitions", type=int, default=1_000)
    parser.add_argument(
        "--post-rollout-actor-updates",
        type=int,
        default=0,
        help=(
            "Additional actor gradient updates after rollout collection; use with an "
            "actor warmup larger than the replay to separate exploration from distillation"
        ),
    )
    parser.add_argument("--replay-nonzero-fraction", type=float, default=0.5)
    parser.add_argument("--replay-nonzero-success-fraction", type=float, default=0.5)
    parser.add_argument(
        "--behavior-policy",
        choices=("candidate_lcb", "persistent_actor"),
        default="candidate_lcb",
    )
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--candidate-noise-start", type=float, default=0.30)
    parser.add_argument("--candidate-noise-end", type=float, default=0.08)
    parser.add_argument("--candidate-noise-anneal-start-step", type=int, default=0)
    parser.add_argument("--candidate-noise-anneal-steps", type=int, default=200_000)
    parser.add_argument("--candidate-explore-probability-start", type=float, default=1.0)
    parser.add_argument("--candidate-explore-probability-end", type=float, default=0.10)
    parser.add_argument("--candidate-explore-anneal-steps", type=int, default=100_000)
    parser.add_argument("--persistent-noise-correlation", type=float, default=0.95)
    parser.add_argument("--persistent-gripper-noise-scale", type=float, default=0.5)
    parser.add_argument(
        "--exploration-noise-mode",
        choices=("ou", "macro_knots", "vla_samples"),
        default="ou",
        help="Exploration source used across each bounded exploration burst",
    )
    parser.add_argument(
        "--vla-exploration-candidates",
        type=int,
        default=3,
        help="Alternative frozen-VLA chunks ranked by V during each vla_samples boundary",
    )
    parser.add_argument(
        "--vla-exploration-scales",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0, 8.0),
        help=(
            "Positive multipliers searched along each frozen-VLA residual direction; "
            "the unmodified base chunk is always an additional candidate"
        ),
    )
    parser.add_argument("--exploration-burst-chunks", type=int, default=5)
    parser.add_argument("--max-exploration-bursts-per-episode", type=int, default=2)
    parser.add_argument("--exploration-burst-cooldown-chunks", type=int, default=5)
    parser.add_argument(
        "--actor-requires-immediate-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Authorize persistent actor only after an immediate high-confidence VGate entry",
    )
    parser.add_argument(
        "--actor-safety-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily return to base after an intervention causes a negative V change",
    )
    parser.add_argument("--actor-safety-min-value-improvement", type=float, default=-0.02)
    parser.add_argument("--actor-safety-cooldown-chunks", type=int, default=3)
    parser.add_argument(
        "--actor-max-consecutive-chunks",
        type=int,
        default=0,
        help="Maximum deterministic actor chunks before a base-policy cooldown; 0 is unlimited",
    )
    parser.add_argument("--actor-throttle-cooldown-chunks", type=int, default=3)
    parser.add_argument("--background-probe-episode-probability", type=float, default=0.0)
    parser.add_argument(
        "--background-probe-episode-probability-end",
        type=float,
        default=None,
        help="Optional final background-probe probability after curriculum annealing",
    )
    parser.add_argument("--background-probe-anneal-start-step", type=int, default=0)
    parser.add_argument("--background-probe-anneal-steps", type=int, default=200_000)
    parser.add_argument(
        "--background-probe-max-chunks-per-episode",
        type=int,
        default=1,
        help=(
            "Maximum consecutive 10-step exploration chunks in one training-only background burst"
        ),
    )
    parser.add_argument("--background-probe-max-boundary", type=int, default=20)
    parser.add_argument(
        "--background-probe-min-value-improvement",
        type=float,
        default=-0.20,
        help="Cancel a training-only background burst after a larger frozen-V drop",
    )
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-scorers", type=int, default=5)
    parser.add_argument("--exploration-knots", type=int, default=6)
    parser.add_argument("--arm-residual-fraction", type=float, default=0.06)
    parser.add_argument("--gripper-residual-fraction", type=float, default=0.10)
    parser.add_argument("--actor-residual-limit", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--actor-update-period", type=int, default=2)
    parser.add_argument(
        "--actor-context-trainable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Give the actor an independent context encoder trained by its own "
            "objectives instead of the value-delta scorer representation"
        ),
    )
    parser.add_argument(
        "--actor-hypotheses",
        type=int,
        default=1,
        help="Number of context-selected residual modes (1 preserves the legacy actor)",
    )
    parser.add_argument("--actor-hypothesis-loss-weight", type=float, default=1.0)
    parser.add_argument("--actor-value-objective-weight", type=float, default=1.0)
    parser.add_argument("--actor-l2-weight", type=float, default=0.5)
    parser.add_argument("--actor-smoothness-weight", type=float, default=0.2)
    parser.add_argument("--actor-awr-weight", type=float, default=2.0)
    parser.add_argument("--actor-awr-temperature", type=float, default=0.10)
    parser.add_argument("--actor-awr-min-advantage", type=float, default=0.01)
    parser.add_argument("--actor-success-bc-weight", type=float, default=2.0)
    parser.add_argument("--actor-success-bc-min-value-improvement", type=float, default=-0.02)
    parser.add_argument("--actor-success-bc-min-residual-rms", type=float, default=1e-4)
    parser.add_argument(
        "--actor-success-bc-requires-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Distill only successful exploration chunks executed while VGate was active",
    )
    parser.add_argument(
        "--actor-success-credit-mode",
        choices=("chunk_threshold", "best_positive_burst"),
        default="chunk_threshold",
        help=(
            "Credit each positive successful chunk independently, or imitate the one "
            "successful exploration burst with the largest net frozen-V improvement"
        ),
    )
    parser.add_argument(
        "--actor-deployment-policy",
        choices=("actor_mean", "success_retrieval"),
        default="actor_mean",
        help="Deterministic correction used after VGate entry",
    )
    parser.add_argument("--retrieval-max-step-distance", type=int, default=50)
    parser.add_argument("--retrieval-min-similarity", type=float, default=0.0)
    parser.add_argument("--bootstrap-probability", type=float, default=0.8)
    parser.add_argument("--selection-uncertainty-penalty", type=float, default=1.0)
    parser.add_argument("--selection-residual-penalty", type=float, default=0.01)
    parser.add_argument("--selection-min-advantage", type=float, default=0.01)
    parser.add_argument("--gate-enter-failure-probability", type=float, default=0.65)
    parser.add_argument("--gate-immediate-failure-probability", type=float, default=0.85)
    parser.add_argument("--gate-exit-failure-probability", type=float, default=0.25)
    parser.add_argument("--gate-enter-confirm-chunks", type=int, default=2)
    parser.add_argument("--gate-exit-confirm-chunks", type=int, default=2)
    parser.add_argument("--gate-min-active-chunks", type=int, default=1)
    parser.add_argument("--gate-ema-alpha", type=float, default=0.5)
    parser.add_argument("--gate-immediate-max-entropy", type=float, default=1.5)
    parser.add_argument("--gate-latest-entry-step", type=int, default=400)
    parser.add_argument("--max-intervention-env-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/rlt/SolarPanelStatic-v2_safe_vgate_bandit")
    parser.add_argument("--save-every-env-steps", type=int, default=5_000)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-replay", default=None)
    parser.add_argument("--resume-trainer-state", default=None)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="maniskill-vgate-bandit")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    args = parser.parse_args()

    if args.control_mode != "pd_joint_pos":
        parser.error("This implementation requires --control-mode pd_joint_pos")
    if args.chunk_len != 10:
        parser.error("Value-guided bandit requires --chunk-len 10")
    if not 0.0 <= args.scorer_return_gamma <= 1.0:
        parser.error("--scorer-return-gamma must lie in [0, 1]")
    if args.scorer_trace_chunks <= 0:
        parser.error("--scorer-trace-chunks must be positive")
    if args.max_episode_steps % args.chunk_len:
        parser.error("--max-episode-steps must be divisible by --chunk-len")
    if (
        min(
            args.total_env_steps,
            args.buffer_capacity,
            args.batch_size,
            args.updates_per_chunk,
            args.warmup_transitions,
            args.actor_warmup_transitions,
            args.save_every_env_steps,
            args.exploration_burst_chunks,
            args.max_exploration_bursts_per_episode,
            args.actor_hypotheses,
        )
        <= 0
    ):
        parser.error("Training sizes, updates, warmups, and save interval must be positive")
    if args.buffer_capacity < args.batch_size:
        parser.error("--buffer-capacity must be at least --batch-size")
    if args.exploration_burst_cooldown_chunks < 0:
        parser.error("--exploration-burst-cooldown-chunks must be non-negative")
    if args.actor_safety_cooldown_chunks < 0:
        parser.error("--actor-safety-cooldown-chunks must be non-negative")
    if args.actor_max_consecutive_chunks < 0:
        parser.error("--actor-max-consecutive-chunks must be non-negative")
    if args.actor_throttle_cooldown_chunks < 0:
        parser.error("--actor-throttle-cooldown-chunks must be non-negative")
    if args.retrieval_max_step_distance < 0:
        parser.error("--retrieval-max-step-distance must be non-negative")
    if not -1.0 <= args.retrieval_min_similarity <= 1.0:
        parser.error("--retrieval-min-similarity must lie in [-1, 1]")
    if args.post_rollout_actor_updates < 0:
        parser.error("--post-rollout-actor-updates must be non-negative")
    if not 1 <= args.vla_exploration_candidates <= 8:
        parser.error("--vla-exploration-candidates must lie in [1, 8]")
    if (
        not args.vla_exploration_scales
        or any(not np.isfinite(scale) or scale <= 0.0 for scale in args.vla_exploration_scales)
        or list(args.vla_exploration_scales) != sorted(set(args.vla_exploration_scales))
    ):
        parser.error(
            "--vla-exploration-scales must be finite, positive, unique, and increasing"
        )
    if not 0.0 <= args.background_probe_episode_probability <= 1.0:
        parser.error("--background-probe-episode-probability must lie in [0, 1]")
    if (
        args.background_probe_episode_probability_end is not None
        and not 0.0 <= args.background_probe_episode_probability_end <= 1.0
    ):
        parser.error("--background-probe-episode-probability-end must lie in [0, 1]")
    if min(args.candidate_noise_anneal_start_step, args.background_probe_anneal_start_step) < 0:
        parser.error("Curriculum start steps must be non-negative")
    if min(
        args.candidate_noise_anneal_steps,
        args.candidate_explore_anneal_steps,
        args.background_probe_anneal_steps,
    ) <= 0:
        parser.error("Curriculum anneal durations must be positive")
    if args.background_probe_max_chunks_per_episode < 0:
        parser.error("--background-probe-max-chunks-per-episode must be non-negative")
    if args.background_probe_max_boundary < 0:
        parser.error("--background-probe-max-boundary must be non-negative")
    if not np.isfinite(args.background_probe_min_value_improvement):
        parser.error("--background-probe-min-value-improvement must be finite")
    for name in (
        "replay_nonzero_fraction",
        "replay_nonzero_success_fraction",
        "candidate_noise_start",
        "candidate_noise_end",
        "candidate_explore_probability_start",
        "candidate_explore_probability_end",
        "bootstrap_probability",
        "gate_enter_failure_probability",
        "gate_immediate_failure_probability",
        "gate_exit_failure_probability",
        "gate_ema_alpha",
        "persistent_noise_correlation",
        "persistent_gripper_noise_scale",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.persistent_noise_correlation >= 1.0:
        parser.error("--persistent-noise-correlation must be below 1")
    if not args.resume_checkpoint and any((args.resume_replay, args.resume_trainer_state)):
        parser.error("resume sidecars require --resume-checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    managed = (
        "run_config.json",
        "history.jsonl",
        "value_guided_bandit.pt",
        "online_bandit_replay.npz",
        "trainer_state.json",
    )
    collisions = [name for name in managed if (output_dir / name).exists()]
    if collisions:
        raise FileExistsError(f"Refusing to mix a new run with {output_dir}: {collisions}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.openpi_bridge.remote_policy import (
        SAFE_LATENT_DIM,
        SAFE_LATENT_PROTOCOL,
    )
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.latent_actor import make_runtime_identity
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter
    from maniskill_myws.rlt.value_guided_bandit import (
        ActorChunkThrottle,
        ExplorationBurstSchedule,
        PersistentResidualExplorer,
        SmoothKnotResidualExplorer,
        VGate,
        VGateConfig,
        ValueBanditReplayBuffer,
        ValueGuidedBanditAgent,
        ValueGuidedBanditConfig,
        linear_curriculum_value,
        update_actor_gate_authorization,
        value_improvement_target,
    )
    from maniskill_myws.rlt.value_model import (
        VALUE_FEATURE_SCHEMA,
        DistributionalBaseValueModel,
        infer_value_estimate,
        value_images_from_observation,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 19_337)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    maniskill_myws.register()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=args.render_mode,
        enhanced_determinism=args.enhanced_determinism,
        max_episode_steps=args.max_episode_steps,
    )
    backend = require_resolved_backend(
        env,
        expected_sim_backend=args.sim_backend,
        expected_render_backend=args.render_backend,
    )
    base_env = env.unwrapped
    raw_reward_schema = getattr(base_env, "grasp_reward_schema", None)
    if raw_reward_schema is None:
        if args.reward_mode != "sparse":
            raise ValueError("Tasks without a reward schema require sparse mode")
        reward_schema = {"schema": "terminal_success_sparse_reward_v1"}
        process_reward = None
        success_reward = 1.0
    else:
        reward_schema = dict(raw_reward_schema)
        process_reward = float(getattr(base_env, "grasp_process_reward"))
        success_reward = float(getattr(base_env, "task_success_reward"))

    action_dim = int(np.prod(env.action_space.shape))
    low, high = _action_bounds(env.action_space, action_dim)
    prompt = args.prompt or getattr(base_env, "DEFAULT_TASK_PROMPT", "")
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
    metadata = policy.server_metadata or {}
    if metadata.get("safe_latent_protocol") != SAFE_LATENT_PROTOCOL:
        raise RuntimeError("OpenPI server does not expose the required SAFE latent")
    if metadata.get("safe_latent_shape") != [SAFE_LATENT_DIM]:
        raise RuntimeError("OpenPI server SAFE latent shape is invalid")
    policy_identity = openpi_policy_identity_sha256(metadata)
    state_adapter = StateAdapter(args.state_keys)
    probe_obs, _ = reset_env_fresh_scene(
        env, seed=args.seed, operation="value-guided bandit shape probe"
    )
    raw_state = np.asarray(state_adapter(probe_obs), dtype=np.float32)

    value_path = Path(args.value_checkpoint)
    if not value_path.is_file():
        raise FileNotFoundError(value_path)
    value_model, value_metadata = DistributionalBaseValueModel.load(value_path, device=device)
    value_model.requires_grad_(False)
    value_config = value_model.config
    expected_value_config = {
        "state_dim": int(raw_state.size),
        "action_dim": action_dim,
        "chunk_len": args.chunk_len,
        "max_episode_steps": args.max_episode_steps,
        "latent_dim": SAFE_LATENT_DIM,
        "num_views": 2,
    }
    config_mismatches = {
        key: (expected, getattr(value_config, key))
        for key, expected in expected_value_config.items()
        if getattr(value_config, key) != expected
    }
    if config_mismatches:
        raise ValueError(f"Value checkpoint configuration mismatch: {config_mismatches}")
    dataset_metadata = value_metadata.get("dataset_metadata")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("Value checkpoint has no rollout dataset identity")
    expected_dataset = {
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "control_mode": args.control_mode,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_keys": [args.image_key, args.wrist_image_key],
        "state_keys": list(args.state_keys),
        "chunk_len": args.chunk_len,
        "max_episode_steps": args.max_episode_steps,
        "action_dim": action_dim,
        "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
        "safe_latent_dim": SAFE_LATENT_DIM,
        "openpi_policy_identity_sha256": policy_identity,
        "base_policy_only": True,
    }
    dataset_mismatches = {
        key: (expected, dataset_metadata.get(key))
        for key, expected in expected_dataset.items()
        if dataset_metadata.get(key) != expected
    }
    if dataset_mismatches:
        raise ValueError(f"Value rollout identity mismatch: {dataset_mismatches}")

    runtime_identity = make_runtime_identity(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enhanced_determinism=args.enhanced_determinism,
        prompt=prompt,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=args.state_keys,
        resize=args.resize,
        chunk_len=args.chunk_len,
        max_episode_steps=args.max_episode_steps,
        openpi_policy_identity_sha256=policy_identity,
        latent_protocol=SAFE_LATENT_PROTOCOL,
        latent_dim=SAFE_LATENT_DIM,
    )
    runtime_identity["environment_reward_schema"] = reward_schema
    runtime_identity["value_model"] = {
        "checkpoint_sha256": _sha256_file(value_path),
        "feature_schema": VALUE_FEATURE_SCHEMA,
        "training_target": "episode_success_plus_remaining_chunks_no_environment_reward",
    }
    runtime_identity["scorer_target"] = {
        "name": args.scorer_target,
        "return_mode": args.scorer_return_mode,
        "return_gamma": args.scorer_return_gamma,
        "trace_chunks": args.scorer_trace_chunks,
        # Retained for exact compatibility with the validated pilot snapshot.
        "flow_signal_checkpoint_sha256": None,
    }
    gate_config = VGateConfig(
        enter_failure_probability=args.gate_enter_failure_probability,
        immediate_failure_probability=args.gate_immediate_failure_probability,
        exit_failure_probability=args.gate_exit_failure_probability,
        enter_confirm_chunks=args.gate_enter_confirm_chunks,
        exit_confirm_chunks=args.gate_exit_confirm_chunks,
        min_active_chunks=args.gate_min_active_chunks,
        ema_alpha=args.gate_ema_alpha,
        immediate_max_entropy=args.gate_immediate_max_entropy,
        latest_entry_step=args.gate_latest_entry_step,
        max_intervention_env_steps=args.max_intervention_env_steps,
    )
    runtime_identity["gate"] = asdict(gate_config)
    runtime_identity["behavior_policy"] = {
        "name": args.behavior_policy,
        "deterministic": (
            "vgate_actor_mean"
            if args.behavior_policy == "persistent_actor"
            else "vgate_candidate_lcb"
        ),
        "exploration": (
            (
                "frozen_vla_tangent_line_search"
                if args.exploration_noise_mode == "vla_samples"
                else (
                    "macro_knot_residual"
                    if args.exploration_noise_mode == "macro_knots"
                    else "cross_chunk_ou_residual"
                )
            )
            if args.behavior_policy == "persistent_actor"
            else "candidate_ensemble_ucb"
        ),
        "vla_exploration_candidates": args.vla_exploration_candidates,
        "vla_exploration_scales": list(args.vla_exploration_scales),
        "persistent_noise_correlation": args.persistent_noise_correlation,
        "persistent_gripper_noise_scale": args.persistent_gripper_noise_scale,
        "burst_chunks": args.exploration_burst_chunks,
        "max_bursts_per_episode": args.max_exploration_bursts_per_episode,
        "burst_cooldown_chunks": args.exploration_burst_cooldown_chunks,
        "actor_requires_immediate_gate": args.actor_requires_immediate_gate,
        "actor_safety_enabled": args.actor_safety_enabled,
        "actor_safety_min_value_improvement": args.actor_safety_min_value_improvement,
        "actor_safety_cooldown_chunks": args.actor_safety_cooldown_chunks,
        "actor_max_consecutive_chunks": args.actor_max_consecutive_chunks,
        "actor_throttle_cooldown_chunks": args.actor_throttle_cooldown_chunks,
        "background_probe_episode_probability": args.background_probe_episode_probability,
        "background_probe_max_chunks_per_episode": args.background_probe_max_chunks_per_episode,
        "background_probe_max_boundary": args.background_probe_max_boundary,
    }
    runtime_identity["exploration_curriculum"] = {
        "noise_start": args.candidate_noise_start,
        "noise_end": args.candidate_noise_end,
        "noise_anneal_start_step": args.candidate_noise_anneal_start_step,
        "noise_anneal_steps": args.candidate_noise_anneal_steps,
        "background_probe_probability_start": args.background_probe_episode_probability,
        "background_probe_probability_end": (
            args.background_probe_episode_probability
            if args.background_probe_episode_probability_end is None
            else args.background_probe_episode_probability_end
        ),
        "background_probe_anneal_start_step": args.background_probe_anneal_start_step,
        "background_probe_anneal_steps": args.background_probe_anneal_steps,
    }
    config = ValueGuidedBanditConfig(
        state_dim=int(raw_state.size) + value_config.critic_feature_dim,
        action_dim=action_dim,
        chunk_len=args.chunk_len,
        max_episode_steps=args.max_episode_steps,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_scorers=args.num_scorers,
        candidate_count=args.candidate_count,
        exploration_knots=args.exploration_knots,
        arm_residual_fraction=args.arm_residual_fraction,
        gripper_residual_fraction=args.gripper_residual_fraction,
        actor_residual_limit=args.actor_residual_limit,
        scorer_lr=args.lr,
        actor_lr=args.lr,
        actor_update_period=args.actor_update_period,
        actor_context_trainable=args.actor_context_trainable,
        actor_hypotheses=args.actor_hypotheses,
        actor_hypothesis_loss_weight=args.actor_hypothesis_loss_weight,
        actor_value_objective_weight=args.actor_value_objective_weight,
        actor_l2_weight=args.actor_l2_weight,
        actor_smoothness_weight=args.actor_smoothness_weight,
        actor_awr_weight=args.actor_awr_weight,
        actor_awr_temperature=args.actor_awr_temperature,
        actor_awr_min_advantage=args.actor_awr_min_advantage,
        actor_success_bc_weight=args.actor_success_bc_weight,
        actor_success_bc_min_value_improvement=(args.actor_success_bc_min_value_improvement),
        actor_success_bc_min_residual_rms=args.actor_success_bc_min_residual_rms,
        actor_success_bc_requires_gate=args.actor_success_bc_requires_gate,
        actor_success_credit_mode=args.actor_success_credit_mode,
        actor_deployment_policy=args.actor_deployment_policy,
        retrieval_max_step_distance=args.retrieval_max_step_distance,
        retrieval_min_similarity=args.retrieval_min_similarity,
        bootstrap_probability=args.bootstrap_probability,
        selection_uncertainty_penalty=args.selection_uncertainty_penalty,
        selection_residual_penalty=args.selection_residual_penalty,
        selection_min_advantage=args.selection_min_advantage,
        deterministic_actor_on_gate=args.behavior_policy == "persistent_actor",
        actor_requires_immediate_gate=args.actor_requires_immediate_gate,
        actor_safety_enabled=args.actor_safety_enabled,
        actor_safety_min_value_improvement=args.actor_safety_min_value_improvement,
        actor_safety_cooldown_chunks=args.actor_safety_cooldown_chunks,
        actor_max_consecutive_chunks=args.actor_max_consecutive_chunks,
        actor_throttle_cooldown_chunks=args.actor_throttle_cooldown_chunks,
        action_low=tuple(float(value) for value in low),
        action_high=tuple(float(value) for value in high),
    )
    replay = ValueBanditReplayBuffer(
        args.buffer_capacity,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=args.seed,
    )
    checkpoint_path = output_dir / "value_guided_bandit.pt"
    replay_path = output_dir / "online_bandit_replay.npz"
    state_path = output_dir / "trainer_state.json"
    history_path = output_dir / "history.jsonl"
    resume_checkpoint_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_replay_path = (
        Path(args.resume_replay)
        if args.resume_replay
        else (
            resume_checkpoint_path.with_name("online_bandit_replay.npz")
            if resume_checkpoint_path is not None
            else None
        )
    )
    resume_state_path = (
        Path(args.resume_trainer_state)
        if args.resume_trainer_state
        else (
            resume_checkpoint_path.with_name("trainer_state.json")
            if resume_checkpoint_path is not None
            else None
        )
    )
    resume_progress = None
    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        if resume_replay_path is None or not resume_replay_path.is_file():
            raise FileNotFoundError(f"Resume replay not found: {resume_replay_path}")
        if resume_state_path is None or not resume_state_path.is_file():
            raise FileNotFoundError(f"Resume trainer state not found: {resume_state_path}")
        agent = ValueGuidedBanditAgent.load(resume_checkpoint_path, device=device)
        stored_identity = agent.runtime_identity or {}
        compatible_identity = json.loads(json.dumps(runtime_identity))
        stored_behavior = stored_identity.get("behavior_policy", {})
        compatible_behavior = compatible_identity.get("behavior_policy", {})
        for key in (
            "vla_exploration_candidates",
            "vla_exploration_scales",
            "actor_max_consecutive_chunks",
            "actor_throttle_cooldown_chunks",
        ):
            if key not in stored_behavior:
                compatible_behavior.pop(key, None)
        if "scorer_target" not in stored_identity:
            legacy_scorer_target = compatible_identity.pop("scorer_target", None)
            expected_legacy_target = {
                "name": "value_improvement",
                "return_mode": "one_step",
                "return_gamma": 0.99,
                "trace_chunks": 5,
                "flow_signal_checkpoint_sha256": None,
            }
            if legacy_scorer_target != expected_legacy_target:
                raise ValueError(
                    "Legacy checkpoint can only resume with its original one-step value target"
                )
        if "exploration_curriculum" not in stored_identity:
            compatible_identity.pop("exploration_curriculum", None)
        agent.assert_runtime_identity(compatible_identity)
        if agent.config != config:
            raise ValueError("Resume checkpoint configuration does not match")
        # Persist the explicit curriculum identity in all continuation snapshots.
        agent.runtime_identity = runtime_identity
        replay.load(resume_replay_path)
        resume_progress = _load_trainer_state(resume_state_path, rng=rng, torch=torch)
        generations = {
            "checkpoint": agent.snapshot_id,
            "replay": replay.last_loaded_snapshot_id,
            "trainer_state": resume_progress["snapshot_id"],
        }
        if any(value != resume_progress["snapshot_id"] for value in generations.values()):
            raise ValueError(f"Resume snapshot files are from different generations: {generations}")
    else:
        agent = ValueGuidedBanditAgent(config, device=device, runtime_identity=runtime_identity)
    run_config = {
        "schema": "safe_value_guided_chunk_bandit_run_v1",
        "args": vars(args),
        "agent_config": asdict(config),
        "gate_config": asdict(gate_config),
        "runtime_identity": runtime_identity,
        "backend": backend,
        "initialization": {
            "rl_checkpoint": (
                str(resume_checkpoint_path.resolve())
                if resume_checkpoint_path is not None
                else None
            ),
            "replay": (
                str(resume_replay_path.resolve()) if resume_replay_path is not None else None
            ),
            "actor": (
                "resumed_exact_snapshot"
                if resume_progress is not None
                else "random_zero_output_head"
            ),
            "scorer": (
                "resumed_exact_snapshot" if resume_progress is not None else "random_ensemble"
            ),
            "value_model": "frozen_pretrained",
        },
        "resume": (
            {
                "snapshot_id": resume_progress["snapshot_id"],
                "trainer_state": str(resume_state_path.resolve()),
            }
            if resume_progress is not None
            else None
        ),
        "reward_contract": {
            "environment_reward_recorded_for_audit": True,
            "environment_reward_used_as_bandit_target": False,
            "value_model_uses_process_reward": False,
            "bandit_target": args.scorer_target,
            "bandit_return_mode": args.scorer_return_mode,
            "bandit_return_gamma": args.scorer_return_gamma,
            "bandit_trace_chunks": args.scorer_trace_chunks,
            "terminal_success_utility": 0.0,
            "terminal_failure_utility": value_config.failure_value,
            "actor_success_bc_target": "terminal_episode_success_only",
        },
    }
    _write_json_atomic(output_dir / "run_config.json", run_config)

    def value_inputs(observation: dict) -> tuple[np.ndarray, np.ndarray]:
        raw = _validate("raw state", state_adapter(observation), (raw_state.size,))
        images = value_images_from_observation(
            observation,
            image_keys=(args.image_key, args.wrist_image_key),
            height=value_config.image_height,
            width=value_config.image_width,
        )
        return raw, images

    def estimate_plan(
        *,
        raw: np.ndarray,
        images: np.ndarray,
        ref: np.ndarray,
        latent: np.ndarray,
        step_id: int,
    ):
        ref = _validate("reference chunk", ref, (args.chunk_len, action_dim))
        latent = _validate("SAFE latent", latent, (SAFE_LATENT_DIM,))
        return infer_value_estimate(
            value_model,
            images=images,
            state=raw,
            latent=latent,
            ref_chunk=ref,
            step_id=step_id,
        )

    def plan_context(observation: dict, step_id: int, episode_seed: int):
        ref, latent = policy.plan_with_latent(
            observation,
            chunk_len=args.chunk_len,
            action_dim=action_dim,
            inference_seed=inference_seed_for_step(episode_seed, step_id),
        )
        ref = _validate("reference chunk", ref, (args.chunk_len, action_dim))
        latent = _validate("SAFE latent", latent, (SAFE_LATENT_DIM,))
        raw, images = value_inputs(observation)
        estimate = estimate_plan(
            raw=raw, images=images, ref=ref, latent=latent, step_id=step_id
        )
        augmented = np.concatenate([raw, estimate.critic_features]).astype(np.float32, copy=False)
        return ref, latent, augmented, estimate

    def sample_vla_residual(
        observation: dict,
        *,
        primary_ref: np.ndarray,
        primary_estimate,
        step_id: int,
        episode_seed: int,
    ) -> tuple[np.ndarray, dict]:
        raw, images = value_inputs(observation)
        candidates: list[tuple[float, float, np.ndarray, int | None, float]] = [
            (
                float(primary_estimate.potential),
                0.0,
                np.zeros_like(primary_ref, dtype=np.float32),
                None,
                0.0,
            )
        ]
        for index in range(args.vla_exploration_candidates):
            inference_seed = inference_seed_for_step(
                episode_seed, step_id, stream=211 + index
            )
            alternative, alternative_latent = policy.plan_with_latent(
                observation,
                chunk_len=args.chunk_len,
                action_dim=action_dim,
                inference_seed=inference_seed,
            )
            alternative = _validate(
                "VLA exploration chunk", alternative, (args.chunk_len, action_dim)
            )
            alternative_latent = _validate(
                "VLA exploration latent", alternative_latent, (SAFE_LATENT_DIM,)
            )
            residuals = agent.vla_tangent_residuals(
                primary_ref,
                alternative,
                scales=args.vla_exploration_scales,
            )
            for scale, residual in zip(args.vla_exploration_scales, residuals, strict=True):
                effective = agent.apply_residual(primary_ref, residual)
                estimate = estimate_plan(
                    raw=raw,
                    images=images,
                    ref=effective,
                    latent=alternative_latent,
                    step_id=step_id,
                )
                residual_rms = float(np.sqrt(np.mean(np.square(residual))))
                candidates.append(
                    (
                        float(estimate.potential),
                        residual_rms,
                        residual,
                        inference_seed,
                        float(scale),
                    )
                )
        best_potential, best_rms, best_residual, best_seed, best_scale = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        return best_residual, {
            "step": step_id,
            "candidate_count": len(candidates),
            "vla_sample_count": args.vla_exploration_candidates,
            "selected_inference_seed": best_seed,
            "selected_scale": best_scale,
            "base_selected": best_seed is None,
            "base_potential": float(primary_estimate.potential),
            "selected_potential": float(best_potential),
            "potential_gain": float(best_potential - primary_estimate.potential),
            "residual_rms": best_rms,
        }

    last_save = int(resume_progress["last_save"]) if resume_progress else 0
    env_steps = int(resume_progress["env_steps"]) if resume_progress else 0
    episode_index = int(resume_progress["episode"]) if resume_progress else 0
    recent_successes = list(resume_progress["recent_successes"]) if resume_progress else []
    post_rollout_actor_updates_completed = (
        int(resume_progress["post_rollout_actor_updates_completed"])
        if resume_progress
        else 0
    )
    started = time.time()
    last_metrics: dict[str, float] = {}
    wandb_run = None
    if args.wandb_enabled:
        import wandb

        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            tags=args.wandb_tags,
            config=run_config,
        )
        if resume_progress and resume_progress.get("wandb_run_id"):
            wandb_kwargs.update(id=str(resume_progress["wandb_run_id"]), resume="must")
        wandb_run = wandb.init(**wandb_kwargs)

    def save_snapshot() -> None:
        nonlocal last_save
        snapshot_id = f"step-{env_steps:012d}-episode-{episode_index:08d}-{uuid.uuid4().hex}"
        agent.save(checkpoint_path, snapshot_id=snapshot_id)
        replay.save(replay_path, snapshot_id=snapshot_id)
        payload = {
            "schema": TRAINER_STATE_SCHEMA,
            "snapshot_id": snapshot_id,
            "env_steps": env_steps,
            "episode": episode_index,
            "last_save": env_steps,
            "recent_successes": recent_successes[-50:],
            "scorer_updates": agent.scorer_updates,
            "actor_updates": agent.actor_updates,
            "post_rollout_actor_updates_completed": post_rollout_actor_updates_completed,
            "trainer_rng_state": rng.bit_generator.state,
            "torch_rng_state": _capture_torch_rng_state(torch),
            "wandb_run_id": (str(wandb_run.id) if wandb_run is not None else None),
        }
        _write_json_atomic(state_path, payload)
        last_save = env_steps

    try:
        while env_steps < args.total_env_steps:
            episode_seed = args.seed + episode_index
            obs, _ = reset_env_fresh_scene(
                env,
                seed=episode_seed,
                operation=f"value-guided bandit episode {episode_index}",
            )
            policy.reset()
            agent.reset_deployment_state()
            gate = VGate(gate_config)
            explorer = (
                None
                if args.exploration_noise_mode == "vla_samples"
                else SmoothKnotResidualExplorer(
                    action_dim,
                    knot_count=args.exploration_knots,
                    seed=inference_seed_for_step(episode_seed, 0, stream=117),
                    gripper_scale=args.persistent_gripper_noise_scale,
                )
                if args.exploration_noise_mode == "macro_knots"
                else PersistentResidualExplorer(
                    action_dim,
                    correlation=args.persistent_noise_correlation,
                    seed=inference_seed_for_step(episode_seed, 0, stream=117),
                    gripper_scale=args.persistent_gripper_noise_scale,
                )
            )
            burst_schedule = ExplorationBurstSchedule(
                burst_chunks=args.exploration_burst_chunks,
                max_bursts=args.max_exploration_bursts_per_episode,
                cooldown_chunks=args.exploration_burst_cooldown_chunks,
            )
            episode_step = 0
            episode_return = 0.0
            episode_success = False
            process_events = 0
            episode_interventions = 0
            episode_explorations = 0
            gate_entries: list[int] = []
            gate_exits: list[int] = []
            selected_indices: list[int] = []
            improvements: list[float] = []
            selected_lcbs: list[float] = []
            exploration_burst_events: list[dict] = []
            actor_gate_authorized = False
            actor_safety_cooldown = 0
            actor_throttle = ActorChunkThrottle(
                max_consecutive_chunks=args.actor_max_consecutive_chunks,
                cooldown_chunks=args.actor_throttle_cooldown_chunks,
            )
            actor_throttled_chunks = 0
            background_probe_probability = linear_curriculum_value(
                env_steps,
                start_step=args.background_probe_anneal_start_step,
                anneal_steps=args.background_probe_anneal_steps,
                start_value=args.background_probe_episode_probability,
                end_value=(
                    args.background_probe_episode_probability
                    if args.background_probe_episode_probability_end is None
                    else args.background_probe_episode_probability_end
                ),
            )
            background_probe_start_boundary: int | None = None
            if (
                args.behavior_policy == "persistent_actor"
                and args.background_probe_max_chunks_per_episode > 0
                and rng.random() < background_probe_probability
            ):
                max_boundary = min(
                    args.max_episode_steps // args.chunk_len - 1,
                    args.background_probe_max_boundary,
                )
                if max_boundary >= 0:
                    background_probe_start_boundary = int(rng.integers(max_boundary + 1))
            background_probe_chunks = 0
            background_probe_chunks_remaining = 0
            background_probe_events: list[dict] = []
            vla_sample_events: list[dict] = []
            pending = None

            while episode_step < args.max_episode_steps:
                if pending is None:
                    ref, latent, state, estimate = plan_context(obs, episode_step, episode_seed)
                else:
                    ref, latent, state, estimate = pending
                    pending = None
                gate_decision = gate.decide(
                    failure_probability=estimate.failure_probability,
                    entropy=estimate.entropy,
                    step_id=episode_step,
                )
                actor_gate_authorized = update_actor_gate_authorization(
                    actor_gate_authorized,
                    gate_decision,
                    require_immediate_entry=args.actor_requires_immediate_gate,
                )
                if gate_decision.event.startswith("ENTER"):
                    gate_entries.append(episode_step)
                if gate_decision.event.startswith("EXIT"):
                    gate_exits.append(episode_step)

                noise_std = linear_curriculum_value(
                    env_steps,
                    start_step=args.candidate_noise_anneal_start_step,
                    anneal_steps=args.candidate_noise_anneal_steps,
                    start_value=args.candidate_noise_start,
                    end_value=args.candidate_noise_end,
                )
                explore_progress = min(1.0, env_steps / float(args.candidate_explore_anneal_steps))
                explore_probability = args.candidate_explore_probability_start + (
                    explore_progress
                    * (
                        args.candidate_explore_probability_end
                        - args.candidate_explore_probability_start
                    )
                )
                allow_exploration = bool(
                    gate_decision.active
                    and (
                        len(replay) < args.actor_warmup_transitions
                        or rng.random() < explore_probability
                    )
                )
                if args.behavior_policy == "persistent_actor":
                    persistent_gate_active = bool(
                        gate_decision.active
                        and actor_gate_authorized
                        and actor_safety_cooldown <= 0
                    )
                    current_boundary = episode_step // args.chunk_len
                    if gate_decision.active and background_probe_chunks_remaining > 0:
                        background_probe_chunks_remaining = 0
                        background_probe_events.append(
                            {"step": episode_step, "event": "CANCEL_GATE_ENTRY"}
                        )
                    background_probe_start = bool(
                        not gate_decision.active
                        and actor_safety_cooldown <= 0
                        and background_probe_chunks == 0
                        and background_probe_start_boundary is not None
                        and current_boundary == background_probe_start_boundary
                    )
                    if background_probe_start:
                        background_probe_chunks_remaining = (
                            args.background_probe_max_chunks_per_episode
                        )
                        if explorer is not None:
                            explorer.start_burst(
                                total_steps=(
                                    args.background_probe_max_chunks_per_episode
                                    * args.chunk_len
                                ),
                                std=noise_std,
                            )
                        background_probe_events.append(
                            {"step": episode_step, "event": "BURST_START"}
                        )
                    background_probe = bool(
                        not gate_decision.active
                        and actor_safety_cooldown <= 0
                        and background_probe_chunks_remaining > 0
                    )
                    burst = burst_schedule.decide(
                        gate_active=persistent_gate_active,
                        start_requested=allow_exploration,
                    )
                    actor_requested = bool(
                        persistent_gate_active and not background_probe and not burst.explore
                    )
                    actor_throttle_decision = actor_throttle.decide(requested=actor_requested)
                    actor_allowed = bool(actor_requested and actor_throttle_decision.allowed)
                    actor_throttled_chunks += int(
                        actor_throttle_decision.event == "THROTTLED"
                    )
                    if (
                        explorer is not None
                        and burst.event in {"BURST_START", "BURST_SINGLE"}
                    ):
                        explorer.start_burst(
                            total_steps=args.exploration_burst_chunks * args.chunk_len,
                            std=noise_std,
                        )
                    if burst.event in {"BURST_START", "BURST_END", "CANCEL_GATE_EXIT"}:
                        exploration_burst_events.append(
                            {"step": episode_step, "event": burst.event}
                        )
                    if background_probe:
                        if args.exploration_noise_mode == "vla_samples":
                            residual, sample_event = sample_vla_residual(
                                obs,
                                primary_ref=ref,
                                primary_estimate=estimate,
                                step_id=episode_step,
                                episode_seed=episode_seed,
                            )
                            sample_event["source"] = "background_probe"
                            vla_sample_events.append(sample_event)
                            selection_reason = "VLA_BACKGROUND_EXPLORE"
                        else:
                            assert explorer is not None
                            residual = agent.propose_actor_residual(
                                state,
                                latent,
                                ref,
                                step_id=episode_step,
                                exploration_noise=explorer.sample(
                                    args.chunk_len, std=noise_std
                                ),
                            )
                            selection_reason = "BACKGROUND_EXPLORE"
                        selected = 2
                        background_probe_chunks += 1
                        background_probe_chunks_remaining -= 1
                        if background_probe_chunks_remaining == 0:
                            background_probe_events.append(
                                {"step": episode_step, "event": "BURST_END"}
                            )
                    elif burst.explore:
                        if args.exploration_noise_mode == "vla_samples":
                            residual, sample_event = sample_vla_residual(
                                obs,
                                primary_ref=ref,
                                primary_estimate=estimate,
                                step_id=episode_step,
                                episode_seed=episode_seed,
                            )
                            sample_event["source"] = "gate_burst"
                            vla_sample_events.append(sample_event)
                            selection_reason = "VLA_GATE_EXPLORE"
                        else:
                            assert explorer is not None
                            residual = agent.propose_actor_residual(
                                state,
                                latent,
                                ref,
                                step_id=episode_step,
                                exploration_noise=explorer.sample(
                                    args.chunk_len, std=noise_std
                                ),
                            )
                            selection_reason = "PERSISTENT_EXPLORE"
                        selected = 2
                    elif not persistent_gate_active or not actor_allowed:
                        if explorer is not None:
                            explorer.reset()
                        residual = np.zeros_like(ref)
                        selected = 0
                        selection_reason = (
                            "ACTOR_THROTTLED"
                            if actor_requested and not actor_allowed
                            else "VGATE_BASE"
                        )
                    else:
                        residual, deployment = agent.propose_deployment_residual(
                            state, latent, ref, step_id=episode_step
                        )
                        selected = 1
                        selection_reason = str(deployment["source"])
                    diagnostic_candidates = np.stack([np.zeros_like(ref), residual])
                    scores = agent.score_candidates(
                        state,
                        latent,
                        ref,
                        diagnostic_candidates,
                        step_id=episode_step,
                    )
                    selected_lcb = float(scores["lcb"][1])
                else:
                    candidates = agent.propose_candidates(
                        state,
                        latent,
                        ref,
                        step_id=episode_step,
                        noise_std=noise_std,
                        seed=inference_seed_for_step(episode_seed, episode_step, stream=91),
                    )
                    scores = agent.score_candidates(
                        state, latent, ref, candidates, step_id=episode_step
                    )
                    if gate_decision.active:
                        selected, selection_reason = agent.choose_candidate(
                            scores, allow_exploration=allow_exploration
                        )
                    else:
                        selected, selection_reason = 0, "VGATE_BASE"
                    residual = candidates[selected].copy()
                    selected_lcb = float(scores["lcb"][selected])
                intervened = bool(np.any(np.abs(residual) > 1e-6))
                if intervened:
                    episode_interventions += 1
                    episode_explorations += int(
                        selection_reason
                        in {
                            "UCB_EXPLORE",
                            "PERSISTENT_EXPLORE",
                            "BACKGROUND_EXPLORE",
                            "VLA_GATE_EXPLORE",
                            "VLA_BACKGROUND_EXPLORE",
                        }
                    )
                selected_indices.append(selected)
                selected_lcbs.append(selected_lcb)
                action_chunk = agent.apply_residual(ref, residual)
                start_step = episode_step
                rewards: list[float] = []
                done = False
                next_obs = obs
                for action in action_chunk:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    reward_value = _scalar(reward)
                    step_success, process_event = _validate_environment_reward(
                        reward_mode=args.reward_mode,
                        reward_value=reward_value,
                        info=info,
                        process_reward=process_reward,
                        success_reward=success_reward,
                    )
                    rewards.append(reward_value)
                    episode_return += reward_value
                    episode_success |= step_success
                    process_events += int(process_event)
                    episode_step += 1
                    env_steps += 1
                    if args.render_mode is not None:
                        env.render()
                    done = bool(
                        _done(terminated)
                        or _done(truncated)
                        or episode_step >= args.max_episode_steps
                    )
                    if done:
                        break
                duration = len(rewards)
                gate.observe_execution(duration=duration, intervened=intervened)
                if done:
                    next_estimate = None
                else:
                    pending = plan_context(next_obs, episode_step, episode_seed)
                    next_estimate = pending[3]
                improvement = value_improvement_target(
                    current_potential=estimate.potential,
                    next_potential=(None if next_estimate is None else next_estimate.potential),
                    terminal=done,
                    success=done and episode_success,
                    failure_value=value_config.failure_value,
                )
                improvements.append(improvement)
                scorer_target = improvement
                replay.add(
                    state=state,
                    latent=latent,
                    ref_chunk=ref,
                    residual=residual,
                    value_improvement=scorer_target,
                    environment_return=float(np.sum(rewards, dtype=np.float64)),
                    duration=duration,
                    step_id=start_step,
                    gate_active=gate_decision.active,
                    exploration_active=selection_reason
                    in {
                        "UCB_EXPLORE",
                        "PERSISTENT_EXPLORE",
                        "BACKGROUND_EXPLORE",
                        "VLA_GATE_EXPLORE",
                        "VLA_BACKGROUND_EXPLORE",
                    },
                    terminal_success=done and episode_success,
                    terminal_failure=done and not episode_success,
                    episode_id=episode_index,
                )
                if args.actor_safety_enabled and args.behavior_policy == "persistent_actor":
                    if (
                        selection_reason
                        in {"BACKGROUND_EXPLORE", "VLA_BACKGROUND_EXPLORE"}
                        and improvement < args.background_probe_min_value_improvement
                    ):
                        actor_safety_cooldown = args.actor_safety_cooldown_chunks
                        if background_probe_chunks_remaining > 0:
                            background_probe_chunks_remaining = 0
                            background_probe_events.append(
                                {"step": episode_step, "event": "CANCEL_VALUE_DROP"}
                            )
                    elif (
                        selection_reason
                        not in {"BACKGROUND_EXPLORE", "VLA_BACKGROUND_EXPLORE"}
                        and intervened
                        and improvement < args.actor_safety_min_value_improvement
                    ):
                        actor_safety_cooldown = args.actor_safety_cooldown_chunks
                    elif actor_safety_cooldown > 0:
                        actor_safety_cooldown -= 1
                obs = next_obs

                if len(replay) >= args.batch_size:
                    for _ in range(args.updates_per_chunk):
                        batch = replay.sample(
                            args.batch_size,
                            nonzero_fraction=args.replay_nonzero_fraction,
                            nonzero_success_fraction=(args.replay_nonzero_success_fraction),
                        )
                        last_metrics = agent.update(
                            batch,
                            update_actor=len(replay) >= args.actor_warmup_transitions,
                        )
                if done:
                    break

            replay.finalize_episode(
                episode_index,
                success=episode_success,
                success_credit_mode=args.actor_success_credit_mode,
                min_burst_improvement=args.actor_success_bc_min_value_improvement,
                scorer_return_mode=args.scorer_return_mode,
                scorer_return_gamma=args.scorer_return_gamma,
                scorer_trace_chunks=args.scorer_trace_chunks,
            )
            success_memory_size = agent.refresh_success_memory(replay)
            recent_successes.append(int(episode_success))
            recent_successes = recent_successes[-50:]
            pools = replay.pool_counts()
            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "env_step": env_steps,
                "steps": episode_step,
                "return": episode_return,
                "success": episode_success,
                "process_reward_events": process_events,
                "recent_success_rate": float(np.mean(recent_successes)),
                "replay": len(replay),
                "replay_zero": pools["zero"],
                "replay_nonzero": pools["nonzero"],
                "replay_positive_nonzero": pools["positive_nonzero"],
                "replay_successful_nonzero": pools["successful_nonzero"],
                "replay_failed_nonzero": pools["failed_nonzero"],
                "success_memory_size": success_memory_size,
                "intervention_chunks": episode_interventions,
                "exploration_chunks": episode_explorations,
                "exploration_bursts": burst_schedule.bursts_started,
                "background_probe_chunks": background_probe_chunks,
                "background_probe_probability": background_probe_probability,
                "exploration_noise_std": noise_std,
                "gate_explore_probability": explore_probability,
                "actor_throttled_chunks": actor_throttled_chunks,
                "background_probe_start_boundary": background_probe_start_boundary,
                "background_probe_events": background_probe_events,
                "vla_sample_events": vla_sample_events,
                "mean_vla_sample_potential_gain": (
                    float(np.mean([event["potential_gain"] for event in vla_sample_events]))
                    if vla_sample_events
                    else 0.0
                ),
                "exploration_burst_events": exploration_burst_events,
                "gate_entries": gate_entries,
                "gate_exits": gate_exits,
                "selected_candidate_indices": selected_indices,
                "mean_value_improvement": float(np.mean(improvements)),
                "positive_value_improvements": int(np.count_nonzero(np.asarray(improvements) > 0)),
                "mean_selected_lcb": float(np.mean(selected_lcbs)),
                "scorer_updates": agent.scorer_updates,
                "actor_updates": agent.actor_updates,
                "elapsed_s": time.time() - started,
                **last_metrics,
            }
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            print("episode", record, flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"train/{key}": value
                        for key, value in record.items()
                        if isinstance(value, (bool, int, float))
                    },
                    step=env_steps,
                )
            episode_index += 1
            if env_steps - last_save >= args.save_every_env_steps:
                save_snapshot()
        if args.post_rollout_actor_updates:
            pools = replay.pool_counts()
            if len(replay) < args.batch_size:
                raise RuntimeError("Replay is too small for post-rollout actor training")
            if pools["successful_nonzero"] == 0:
                raise RuntimeError(
                    "Post-rollout actor training requires successful exploratory residuals"
                )
            while (
                post_rollout_actor_updates_completed
                < args.post_rollout_actor_updates
            ):
                batch = replay.sample(
                    args.batch_size,
                    nonzero_fraction=args.replay_nonzero_fraction,
                    nonzero_success_fraction=args.replay_nonzero_success_fraction,
                )
                actor_updates_before = agent.actor_updates
                last_metrics = agent.update(batch, update_actor=True)
                post_rollout_actor_updates_completed += (
                    agent.actor_updates - actor_updates_before
                )
                if (
                    post_rollout_actor_updates_completed > 0
                    and post_rollout_actor_updates_completed % 250 == 0
                    and agent.actor_updates != actor_updates_before
                ):
                    print(
                        "post-rollout actor",
                        {
                            "completed": post_rollout_actor_updates_completed,
                            "target": args.post_rollout_actor_updates,
                            "successful_nonzero": pools["successful_nonzero"],
                            **last_metrics,
                        },
                        flush=True,
                    )
                if (
                    post_rollout_actor_updates_completed > 0
                    and post_rollout_actor_updates_completed % 500 == 0
                    and agent.actor_updates != actor_updates_before
                ):
                    save_snapshot()
        save_snapshot()
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "final/env_steps": env_steps,
                    "final/episodes": episode_index,
                    "final/scorer_updates": agent.scorer_updates,
                    "final/actor_updates": agent.actor_updates,
                }
            )
            wandb_run.finish()
    print(
        "training complete",
        {
            "checkpoint": str(checkpoint_path),
            "replay": str(replay_path),
            "trainer_state": str(state_path),
            "episodes": episode_index,
            "env_steps": env_steps,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
