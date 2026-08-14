#!/usr/bin/env python
"""Train mean frozen-Pi0-latent Advantage-BC residual TD3."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]
TRAINER_STATE_SCHEMA = "frozen_latent_residual_trainer_state_v2"


def _scalar(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _done(value) -> bool:
    return bool(_scalar(value))


def _validate_environment_reward(
    *,
    reward_mode: str,
    reward_value: float,
    info: dict,
    grasp_process_reward: float | None,
    task_success_reward: float | None,
) -> tuple[bool, bool]:
    """Fail closed if the environment reward no longer matches its schema."""

    if not np.isfinite(reward_value):
        raise ValueError("Environment reward is NaN or Inf")
    if not isinstance(info, dict) or "success" not in info:
        raise ValueError("Environment info must contain scalar success")
    success = _done(info["success"])
    grasp_event = _done(info.get("grasp_reward_event", False))
    if reward_mode not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported reward mode {reward_mode!r}")
    if grasp_process_reward is None or task_success_reward is None:
        raise ValueError("Milestone reward scales were not bound from the environment")
    if "grasp_reward_event" not in info:
        raise ValueError("Milestone reward info is missing grasp_reward_event")
    expected = (
        grasp_process_reward * float(grasp_event)
        + task_success_reward * float(success)
    )
    if not np.isclose(reward_value, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Environment {reward_mode} reward {reward_value} does not match "
            f"declared components {expected}"
        )
    return success, grasp_event


def _action_bounds(action_space, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError("Environment action bounds do not match action_dim")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low >= high):
        raise ValueError("Environment action bounds must be finite with low < high")
    return low, high


def _validate(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _plan_frozen_policy(
    policy,
    obs: dict,
    *,
    chunk_len: int,
    action_dim: int,
    latent_dim: int,
    inference_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref, mean_latent = policy.plan_with_latent(
        obs,
        chunk_len=chunk_len,
        action_dim=action_dim,
        inference_seed=inference_seed,
    )
    return ref, _validate("frozen Pi0 latent", mean_latent, (latent_dim,))


def _discounted_mc(
    transitions: list[dict], gamma: float, chunk_len: int
) -> list[float]:
    returns = [0.0] * len(transitions)
    future = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        duration = len(transitions[index]["executed_rewards"])
        macro_reward = float(np.sum(transitions[index]["executed_rewards"]))
        future = macro_reward + float(gamma) ** (duration / chunk_len) * future
        returns[index] = future
    return returns


def _should_explore_online(
    *,
    eligible: bool,
    accepted: bool,
    independent: bool,
    probability: float,
    rng: np.random.Generator,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("online exploration probability must lie in [0, 1]")
    if not eligible or (accepted and not independent):
        return False
    return bool(rng.random() < probability)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
        values = np.frombuffer(
            base64.b64decode(encoded, validate=True), dtype=np.uint8
        ).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Trainer state has an invalid PyTorch RNG payload") from exc
    return torch.from_numpy(values)


def _capture_torch_rng_state(torch) -> dict[str, Any]:
    payload: dict[str, Any] = {"cpu": _encode_torch_rng_state(torch.get_rng_state())}
    if torch.cuda.is_available():
        payload["cuda"] = [
            _encode_torch_rng_state(state) for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _restore_torch_rng_state(payload: dict[str, Any], torch) -> None:
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


def _load_trainer_state(path: Path, *, rng: np.random.Generator, torch) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume trainer state not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TRAINER_STATE_SCHEMA:
        raise ValueError("Unsupported frozen-latent trainer state")
    required = (
        "env_steps",
        "episode",
        "last_save",
        "recent_successes",
        "warmup_successes",
        "warmup_failures",
        "warmup_nonzero_transitions",
        "trainer_rng_state",
        "torch_rng_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Trainer state is missing fields: {missing}")
    rng.bit_generator.state = payload["trainer_rng_state"]
    _restore_torch_rng_state(payload["torch_rng_state"], torch)
    snapshot_id = payload.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Trainer state is missing its snapshot generation id")
    return {
        "env_steps": int(payload["env_steps"]),
        "episode": int(payload["episode"]),
        "last_save": int(payload["last_save"]),
        "recent_successes": [int(bool(value)) for value in payload["recent_successes"]][-50:],
        "warmup_successes": int(payload["warmup_successes"]),
        "warmup_failures": int(payload["warmup_failures"]),
        "warmup_nonzero_transitions": int(payload["warmup_nonzero_transitions"]),
        "resume_mode": "generation_bound_state_restore",
        "snapshot_id": snapshot_id,
        "wandb_run_id": payload.get("wandb_run_id"),
    }


def _trainer_state_payload(
    *,
    env_steps: int,
    episode: int,
    last_save: int,
    recent_successes: list[int],
    warmup_successes: int,
    warmup_failures: int,
    warmup_nonzero_transitions: int,
    rng: np.random.Generator,
    torch,
    wandb_run_id: str | None,
    snapshot_id: str,
) -> dict[str, Any]:
    return {
        "schema": TRAINER_STATE_SCHEMA,
        "env_steps": int(env_steps),
        "episode": int(episode),
        "last_save": int(last_save),
        "recent_successes": [int(bool(value)) for value in recent_successes[-50:]],
        "warmup_successes": int(warmup_successes),
        "warmup_failures": int(warmup_failures),
        "warmup_nonzero_transitions": int(warmup_nonzero_transitions),
        "trainer_rng_state": rng.bit_generator.state,
        "torch_rng_state": _capture_torch_rng_state(torch),
        "wandb_run_id": wandb_run_id,
        "snapshot_id": snapshot_id,
    }


def _validate_snapshot_generation(
    *,
    resume_progress: dict[str, Any],
    checkpoint_snapshot_id: str | None,
    replay_snapshot_id: str | None,
) -> None:
    if resume_progress.get("resume_mode") != "generation_bound_state_restore":
        return
    expected_snapshot_id = resume_progress.get("snapshot_id")
    snapshot_ids = {
        "checkpoint": checkpoint_snapshot_id,
        "replay": replay_snapshot_id,
        "trainer_state": expected_snapshot_id,
    }
    if not isinstance(expected_snapshot_id, str) or not expected_snapshot_id:
        raise ValueError("Generation-bound trainer state has no snapshot id")
    if any(value != expected_snapshot_id for value in snapshot_ids.values()):
        raise ValueError(
            "Resume snapshot files are from different generations: "
            f"{snapshot_ids}"
        )


def _actor_ready(
    *,
    replay_size: int,
    critic_updates: int,
    warmup_successes: int,
    warmup_failures: int,
    warmup_nonzero_transitions: int,
    args,
) -> bool:
    return bool(
        replay_size >= args.warmup_transitions
        and critic_updates >= args.critic_warmup_updates
        and warmup_successes >= args.min_warmup_successes
        and warmup_failures >= args.min_warmup_failures
        and warmup_nonzero_transitions >= args.min_warmup_nonzero_transitions
    )


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-replay", default=None)
    parser.add_argument("--resume-history", default=None)
    parser.add_argument("--resume-trainer-state", default=None)
    parser.add_argument("--warmup-transitions", type=int, default=128)
    parser.add_argument("--critic-warmup-updates", type=int, default=2_000)
    parser.add_argument("--updates-per-chunk", type=int, default=5)
    parser.add_argument("--max-warmup-explore-chunks-per-episode", type=int, default=1)
    parser.add_argument("--min-warmup-successes", type=int, default=8)
    parser.add_argument("--min-warmup-failures", type=int, default=8)
    parser.add_argument("--min-warmup-nonzero-transitions", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--arm-residual-fraction", type=float, default=0.06)
    parser.add_argument("--gripper-residual-fraction", type=float, default=0.10)
    parser.add_argument("--fixed-std", type=float, default=0.10)
    parser.add_argument("--warmup-exploration-std", type=float, default=0.45)
    parser.add_argument("--exploration-std-start", type=float, default=0.25)
    parser.add_argument("--exploration-std-end", type=float, default=0.05)
    parser.add_argument("--exploration-anneal-steps", type=int, default=50_000)
    parser.add_argument("--online-explore-probability", type=float, default=0.05)
    parser.add_argument(
        "--independent-online-exploration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--base-control-episode-probability", type=float, default=0.0)
    parser.add_argument("--replay-nonzero-fraction", type=float, default=0.0)
    parser.add_argument("--replay-nonzero-success-fraction", type=float, default=0.5)
    parser.add_argument("--min-q-advantage", type=float, default=0.10)
    parser.add_argument("--max-online-intervention-chunks-per-episode", type=int, default=1)
    parser.add_argument("--intervention-cooldown-chunks", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-residual-limit", type=float, default=0.35)
    parser.add_argument("--actor-l2-weight", type=float, default=0.5)
    parser.add_argument("--actor-smoothness-weight", type=float, default=0.2)
    parser.add_argument("--mc-loss-weight", type=float, default=0.5)
    parser.add_argument("--conservative-q-weight", type=float, default=1.0)
    parser.add_argument("--conservative-random-std", type=float, default=0.35)
    parser.add_argument("--success-bc-min-return", type=float, default=0.5)
    parser.add_argument("--actor-success-bc-weight", type=float, default=2.0)
    parser.add_argument("--actor-success-bc-min-residual-rms", type=float, default=1e-4)
    parser.add_argument("--actor-success-bc-min-q-advantage", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", default="outputs/rlt/SolarPanelStatic-v2_frozen_latent_td3"
    )
    parser.add_argument("--save-every-env-steps", type=int, default=5_000)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="maniskill-frozen-latent-td3")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-new-run", action="store_true")
    args = parser.parse_args()

    if args.control_mode != "pd_joint_pos":
        parser.error("This implementation requires --control-mode pd_joint_pos")
    if args.reward_mode not in {"dense", "sparse"}:
        parser.error("This implementation requires environment dense or sparse reward")
    if min(args.chunk_len, args.max_episode_steps, args.total_env_steps) <= 0:
        parser.error("chunk and training horizons must be positive")
    if args.max_episode_steps % args.chunk_len != 0:
        parser.error("--max-episode-steps must be divisible by --chunk-len")
    if args.batch_size <= 0 or args.buffer_capacity < args.batch_size:
        parser.error("--buffer-capacity must be at least --batch-size > 0")
    if not 0 <= args.warmup_transitions <= args.buffer_capacity:
        parser.error("--warmup-transitions must lie in [0, buffer-capacity]")
    if min(
        args.critic_warmup_updates,
        args.updates_per_chunk,
        args.max_warmup_explore_chunks_per_episode,
        args.min_warmup_successes,
        args.min_warmup_failures,
        args.min_warmup_nonzero_transitions,
        args.max_online_intervention_chunks_per_episode,
        args.intervention_cooldown_chunks,
    ) < 0:
        parser.error("update, coverage, and intervention counts must be non-negative")
    if args.exploration_anneal_steps <= 0:
        parser.error("--exploration-anneal-steps must be positive")
    if args.resize <= 0 or args.save_every_env_steps <= 0:
        parser.error("resize and save interval must be positive")
    if not np.isfinite(args.min_q_advantage):
        parser.error("--min-q-advantage must be finite")
    if not 0.0 < args.actor_residual_limit <= 1.0:
        parser.error("--actor-residual-limit must lie in (0,1]")
    for name in (
        "fixed_std",
        "warmup_exploration_std",
        "exploration_std_start",
        "exploration_std_end",
        "online_explore_probability",
        "base_control_episode_probability",
        "replay_nonzero_fraction",
        "replay_nonzero_success_fraction",
        "conservative_random_std",
        "success_bc_min_return",
        "actor_success_bc_min_residual_rms",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0,1]")
    if min(
        args.actor_l2_weight,
        args.actor_smoothness_weight,
        args.mc_loss_weight,
        args.conservative_q_weight,
        args.actor_success_bc_weight,
    ) < 0.0:
        parser.error("loss weights must be non-negative")
    if args.wandb_enabled and not args.wandb_project.strip():
        parser.error("--wandb-project must be non-empty when W&B is enabled")

    if not args.resume_checkpoint and any(
        (args.resume_replay, args.resume_history, args.resume_trainer_state)
    ):
        parser.error("resume sidecars require --resume-checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    collisions = [
        name
        for name in (
            "run_config.json",
            "history.jsonl",
            "frozen_latent_residual.pt",
            "online_replay.npz",
            "trainer_state.json",
        )
        if (output_dir / name).exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to mix a new run with {output_dir}: {collisions}")

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.frozen_latent_rl import (
        FROZEN_LATENT_DIM,
        FROZEN_LATENT_PROTOCOL,
        FrozenLatentReplayBuffer,
        FrozenLatentResidualAgent,
        FrozenLatentRLConfig,
        make_runtime_identity,
    )
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maniskill_myws.register()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

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
    reward_schema = dict(getattr(base_env, "grasp_reward_schema"))
    grasp_process_reward = float(getattr(base_env, "grasp_process_reward"))
    task_success_reward = float(getattr(base_env, "task_success_reward"))
    if not bool(reward_schema.get("grasp_reward_once_per_episode", False)):
        raise ValueError("MC labels require a once-per-episode grasp reward")
    minimum_discounted_success = task_success_reward * float(args.gamma) ** (
        args.max_episode_steps / args.chunk_len
    )
    if not (
        grasp_process_reward
        < args.success_bc_min_return
        < minimum_discounted_success
    ):
        raise ValueError(
            "Success threshold does not separate grasp-only failures from task "
            "successes under the configured discount"
        )

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
        require_frozen_latent=True,
    )
    state_adapter = StateAdapter(args.state_keys)
    obs, _ = reset_env_fresh_scene(
        env, seed=args.seed, operation="frozen latent shape probe"
    )
    state = np.asarray(state_adapter(obs), dtype=np.float32)
    metadata = policy.server_metadata or {}
    if metadata.get("frozen_latent_protocol") != FROZEN_LATENT_PROTOCOL:
        raise RuntimeError("OpenPI server latent protocol changed after validation")
    if metadata.get("frozen_latent_shape") != [FROZEN_LATENT_DIM]:
        raise RuntimeError("OpenPI server frozen latent has an invalid shape")

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
        openpi_policy_identity_sha256=openpi_policy_identity_sha256(metadata),
    )
    runtime_identity["environment_reward_schema"] = reward_schema
    config = FrozenLatentRLConfig(
        state_dim=int(state.size),
        latent_dim=FROZEN_LATENT_DIM,
        action_dim=action_dim,
        chunk_len=args.chunk_len,
        max_episode_steps=args.max_episode_steps,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_critics=args.num_critics,
        arm_residual_fraction=args.arm_residual_fraction,
        gripper_residual_fraction=args.gripper_residual_fraction,
        fixed_std=args.fixed_std,
        actor_residual_limit=args.actor_residual_limit,
        gamma=args.gamma,
        actor_lr=args.lr,
        critic_lr=args.lr,
        target_tau=args.target_tau,
        actor_l2_weight=args.actor_l2_weight,
        actor_smoothness_weight=args.actor_smoothness_weight,
        mc_loss_weight=args.mc_loss_weight,
        conservative_q_weight=args.conservative_q_weight,
        conservative_random_std=args.conservative_random_std,
        success_bc_min_return=args.success_bc_min_return,
        actor_success_bc_weight=args.actor_success_bc_weight,
        actor_success_bc_min_residual_rms=args.actor_success_bc_min_residual_rms,
        actor_success_bc_min_q_advantage=args.actor_success_bc_min_q_advantage,
        action_low=tuple(float(value) for value in low),
        action_high=tuple(float(value) for value in high),
    )
    replay = FrozenLatentReplayBuffer(
        args.buffer_capacity,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed + 17_041)
    resume_checkpoint_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_replay_path = (
        Path(args.resume_replay)
        if args.resume_replay
        else (
            resume_checkpoint_path.with_name("online_replay.npz")
            if resume_checkpoint_path is not None
            else None
        )
    )
    resume_history_path = (
        Path(args.resume_history)
        if args.resume_history
        else (
            resume_checkpoint_path.with_name("history.jsonl")
            if resume_checkpoint_path is not None
            else None
        )
    )
    resume_trainer_state_path = (
        Path(args.resume_trainer_state)
        if args.resume_trainer_state
        else (
            resume_checkpoint_path.with_name("trainer_state.json")
            if resume_checkpoint_path is not None
            else None
        )
    )
    resume_progress: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        if resume_replay_path is None or not resume_replay_path.is_file():
            raise FileNotFoundError(f"Resume replay not found: {resume_replay_path}")
        if resume_history_path is None or not resume_history_path.is_file():
            raise FileNotFoundError(f"Resume history not found: {resume_history_path}")
        agent = FrozenLatentResidualAgent.load(
            resume_checkpoint_path,
            device=device,
        )
        agent.assert_runtime_identity(runtime_identity)
        if agent.config != config:
            raise ValueError("Resume checkpoint configuration does not match")
        replay.load(resume_replay_path)
        if resume_trainer_state_path is not None and resume_trainer_state_path.is_file():
            resume_progress = _load_trainer_state(
                resume_trainer_state_path, rng=rng, torch=torch
            )
            if not replay.last_load_was_exact:
                raise ValueError("Trainer state requires an exact replay snapshot")
            _validate_snapshot_generation(
                resume_progress=resume_progress,
                checkpoint_snapshot_id=agent.snapshot_id,
                replay_snapshot_id=replay.last_loaded_snapshot_id,
            )
        else:
            raise FileNotFoundError(
                f"Resume trainer state not found: {resume_trainer_state_path}"
            )
    else:
        agent = FrozenLatentResidualAgent(
            config, device=device, runtime_identity=runtime_identity
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    checkpoint_path = output_dir / "frozen_latent_residual.pt"
    replay_path = output_dir / "online_replay.npz"
    trainer_state_path = output_dir / "trainer_state.json"
    run_config = {
        "schema": "mean_frozen_latent_advantage_bc_run_v1",
        "args": vars(args),
        "agent_config": asdict(config),
        "backend": backend,
        "runtime_identity": runtime_identity,
        "frozen_latent_protocol": FROZEN_LATENT_PROTOCOL,
        "representation": "mean_pooled_action_suffix",
        "action_parameterization": "six_knot_continuous_fifty_step_residual",
        "replay_stores_exact_executed_actions": True,
        "reward_source": "environment_grasp_event_plus_task_success",
        "environment_reward_schema": reward_schema,
        "discount_semantics": "gamma_per_full_chunk_with_duration_fraction_v1",
        "training_budget_semantics": "finish_started_episode",
        "selector": "conservative_twin_q",
        "uses_advantage_filtered_self_imitation": bool(
            config.actor_success_bc_weight
        ),
        "exploration": {
            "independent_of_q_acceptance": args.independent_online_exploration,
            "probability_per_eligible_boundary": args.online_explore_probability,
            "base_control_episode_probability": args.base_control_episode_probability,
        },
        "replay_sampling": {
            "nonzero_fraction": args.replay_nonzero_fraction,
            "nonzero_success_fraction": args.replay_nonzero_success_fraction,
            "success_threshold": config.success_bc_min_return,
        },
        "intervention_scheduling": {
            "scope": "rollout_local_not_critic_input",
            "max_chunks_per_episode": args.max_online_intervention_chunks_per_episode,
            "cooldown_chunks": args.intervention_cooldown_chunks,
            "min_q_advantage": args.min_q_advantage,
        },
        "resume": (
            {
                "checkpoint": str(resume_checkpoint_path.resolve()),
                "replay": str(resume_replay_path.resolve()),
                "history": str(resume_history_path.resolve()),
                "mode": resume_progress["resume_mode"],
            }
            if resume_progress is not None
            else None
        ),
    }
    _write_json_atomic(output_dir / "run_config.json", run_config)

    env_steps = int(resume_progress["env_steps"]) if resume_progress else 0
    episode = int(resume_progress["episode"]) if resume_progress else 0
    last_save = int(resume_progress["last_save"]) if resume_progress else 0
    recent_successes = list(resume_progress["recent_successes"]) if resume_progress else []
    warmup_successes = (
        int(resume_progress["warmup_successes"])
        if resume_progress
        else 0
    )
    warmup_failures = (
        int(resume_progress["warmup_failures"])
        if resume_progress
        else 0
    )
    warmup_nonzero_transitions = (
        int(resume_progress["warmup_nonzero_transitions"])
        if resume_progress
        else 0
    )
    if env_steps >= args.total_env_steps:
        raise ValueError(
            f"--total-env-steps must exceed resumed env step {env_steps}"
        )

    wandb_run = None
    if args.wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B logging requested but wandb is unavailable") from exc
        wandb_kwargs: dict[str, Any] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name or output_dir.name,
            "tags": list(args.wandb_tags),
            "config": run_config,
            "mode": "online",
            "dir": str(output_dir),
        }
        resumed_wandb_id = (
            resume_progress.get("wandb_run_id")
            if resume_progress and not args.wandb_new_run
            else None
        )
        if resumed_wandb_id:
            wandb_kwargs.update(id=str(resumed_wandb_id), resume="must")
        wandb_run = wandb.init(**wandb_kwargs)
        if wandb_run is None:
            raise RuntimeError("W&B did not return a run handle")
        wandb_run.define_metric("train/env_step")
        wandb_run.define_metric("train/*", step_metric="train/env_step")
        print({"wandb_run": wandb_run.id, "wandb_url": wandb_run.url}, flush=True)

    def save_snapshot() -> None:
        snapshot_id = (
            f"step-{env_steps:012d}-episode-{episode:08d}-{uuid.uuid4().hex}"
        )
        agent.save(checkpoint_path, snapshot_id=snapshot_id)
        replay.save(replay_path, snapshot_id=snapshot_id)
        _write_json_atomic(
            trainer_state_path,
            _trainer_state_payload(
                env_steps=env_steps,
                episode=episode,
                last_save=last_save,
                recent_successes=recent_successes,
                warmup_successes=warmup_successes,
                warmup_failures=warmup_failures,
                warmup_nonzero_transitions=warmup_nonzero_transitions,
                rng=rng,
                torch=torch,
                wandb_run_id=(str(wandb_run.id) if wandb_run is not None else None),
                snapshot_id=snapshot_id,
            ),
        )

    def sample_training_batch():
        if args.replay_nonzero_fraction > 0.0 and replay.has_stratified_support(
            args.batch_size,
            nonzero_fraction=args.replay_nonzero_fraction,
            nonzero_success_fraction=args.replay_nonzero_success_fraction,
            success_threshold=config.success_bc_min_return,
        ):
            return replay.sample_stratified(
                args.batch_size,
                nonzero_fraction=args.replay_nonzero_fraction,
                nonzero_success_fraction=args.replay_nonzero_success_fraction,
                success_threshold=config.success_bc_min_return,
            ), True
        return replay.sample(args.batch_size), False

    metrics: dict[str, float] = {}
    started = time.time()
    episode_in_progress = False
    try:
        while env_steps < args.total_env_steps:
            episode_in_progress = True
            episode_seed = args.seed + episode
            obs, _ = reset_env_fresh_scene(
                env,
                seed=episode_seed,
                operation=f"frozen latent train episode {episode}",
            )
            policy.reset()
            state = np.asarray(state_adapter(obs), dtype=np.float32)
            episode_step = 0
            episode_return = 0.0
            episode_success = False
            episode_grasp_events = 0
            warmup_explores = 0
            online_interventions = 0
            last_intervention_boundary: int | None = None
            cooldown_rejections = 0
            base_control_episode = bool(
                args.base_control_episode_probability > 0.0
                and rng.random() < args.base_control_episode_probability
            )
            boundary_count = args.max_episode_steps // args.chunk_len
            explore_count = min(
                args.max_warmup_explore_chunks_per_episode, boundary_count
            )
            warmup_explore_boundaries = set(
                int(value)
                for value in rng.choice(
                    boundary_count, size=explore_count, replace=False
                )
            )
            pending: tuple[np.ndarray, np.ndarray] | None = None
            episode_transitions: list[dict[str, Any]] = []

            while episode_step < args.max_episode_steps:
                if pending is None:
                    ref, latent = _plan_frozen_policy(
                        policy,
                        obs,
                        chunk_len=args.chunk_len,
                        action_dim=action_dim,
                        latent_dim=config.latent_dim,
                        inference_seed=inference_seed_for_step(
                            episode_seed, episode_step
                        ),
                    )
                else:
                    ref, latent = pending
                    pending = None
                ref = _validate("reference chunk", ref, (args.chunk_len, action_dim))
                latent = _validate("frozen Pi0 latent", latent, replay.latent_shape)
                boundary_index = episode_step // args.chunk_len
                has_budget = (
                    online_interventions
                    < args.max_online_intervention_chunks_per_episode
                )
                cooldown_elapsed = (
                    last_intervention_boundary is None
                    or boundary_index - last_intervention_boundary
                    > args.intervention_cooldown_chunks
                )
                cooldown_rejections += int(has_budget and not cooldown_elapsed)
                eligible = bool(
                    has_budget
                    and cooldown_elapsed
                    and not base_control_episode
                )
                ready = _actor_ready(
                    replay_size=len(replay),
                    critic_updates=agent.critic_updates,
                    warmup_successes=warmup_successes,
                    warmup_failures=warmup_failures,
                    warmup_nonzero_transitions=warmup_nonzero_transitions,
                    args=args,
                )
                accepted = False
                explore = False
                q_advantage = float("nan")
                if ready:
                    actor_residual = agent.select_residual(
                        state,
                        latent,
                        ref,
                        step_id=episode_step,
                        deterministic=True,
                    )
                    q_advantage = agent.conservative_advantage(
                        state,
                        latent,
                        ref,
                        actor_residual,
                        step_id=episode_step,
                    )
                    accepted = bool(eligible and q_advantage >= args.min_q_advantage)
                    explore = _should_explore_online(
                        eligible=eligible,
                        accepted=accepted,
                        independent=args.independent_online_exploration,
                        probability=args.online_explore_probability,
                        rng=rng,
                    )
                    if accepted or explore:
                        if explore:
                            progress = min(
                                1.0,
                                env_steps / float(args.exploration_anneal_steps),
                            )
                            exploration_std = (
                                args.exploration_std_start
                                + progress
                                * (
                                    args.exploration_std_end
                                    - args.exploration_std_start
                                )
                            )
                            requested_residual = agent.select_residual(
                                state,
                                latent,
                                ref,
                                step_id=episode_step,
                                deterministic=False,
                                noise_std=exploration_std,
                            )
                        else:
                            requested_residual = actor_residual
                        action_chunk = agent.apply_residual(ref, requested_residual)
                        phase = (
                            "online_exploration_override"
                            if explore and accepted
                            else (
                                "online_q_gate" if accepted else "online_epsilon"
                            )
                        )
                        intervened = True
                    else:
                        action_chunk = ref.copy()
                        requested_residual = np.zeros_like(ref)
                        phase = "online_base_fallback"
                        intervened = False
                elif eligible and boundary_index in warmup_explore_boundaries:
                    action_chunk, requested_residual = agent.select_chunk(
                        state,
                        latent,
                        ref,
                        step_id=episode_step,
                        deterministic=False,
                        noise_std=args.warmup_exploration_std,
                    )
                    warmup_explores += 1
                    phase = "critic_warmup_explore"
                    intervened = True
                else:
                    action_chunk = ref.copy()
                    requested_residual = np.zeros_like(ref)
                    phase = "critic_warmup_base"
                    intervened = False

                if intervened:
                    online_interventions += 1
                    last_intervention_boundary = boundary_index
                actions: list[np.ndarray] = []
                rewards: list[float] = []
                done = False
                next_obs = obs
                start_step = episode_step
                for offset in range(args.chunk_len):
                    action = _validate(
                        "executed action", action_chunk[offset], (action_dim,)
                    )
                    if np.any(action < low) or np.any(action > high):
                        raise ValueError("Residual action violates environment bounds")
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    if args.render_mode is not None:
                        env.render()
                    reward_value = _scalar(reward)
                    step_success, grasp_event = _validate_environment_reward(
                        reward_mode=args.reward_mode,
                        reward_value=reward_value,
                        info=info,
                        grasp_process_reward=grasp_process_reward,
                        task_success_reward=task_success_reward,
                    )
                    actions.append(action.copy())
                    rewards.append(reward_value)
                    episode_return += reward_value
                    episode_step += 1
                    env_steps += 1
                    episode_success = episode_success or step_success
                    episode_grasp_events += int(grasp_event)
                    done = bool(
                        _done(terminated)
                        or _done(truncated)
                        or episode_step >= args.max_episode_steps
                    )
                    if done:
                        break

                duration = len(actions)
                executed = ref.copy()
                executed[:duration] = np.stack(actions)
                executed_residual = np.clip(
                    (executed - ref) / agent.residual_scale[None, :], -1.0, 1.0
                ).astype(np.float32)
                executed_q_advantage = (
                    agent.conservative_advantage(
                        state,
                        latent,
                        ref,
                        executed_residual,
                        step_id=start_step,
                    )
                    if ready
                    else float("nan")
                )
                reward_array = np.zeros((args.chunk_len,), dtype=np.float32)
                reward_array[:duration] = rewards
                next_state = np.asarray(state_adapter(next_obs), dtype=np.float32)
                if done:
                    next_ref = ref.copy()
                    next_latent = np.zeros(replay.latent_shape, dtype=np.float32)
                else:
                    next_ref, next_latent = _plan_frozen_policy(
                        policy,
                        next_obs,
                        chunk_len=args.chunk_len,
                        action_dim=action_dim,
                        latent_dim=config.latent_dim,
                        inference_seed=inference_seed_for_step(
                            episode_seed, episode_step
                        ),
                    )
                    next_ref = _validate(
                        "next reference chunk", next_ref, (args.chunk_len, action_dim)
                    )
                    next_latent = _validate(
                        "next frozen Pi0 latent", next_latent, replay.latent_shape
                    )
                    pending = (next_ref, next_latent)
                episode_transitions.append(
                    {
                        "state": state.copy(),
                        "latent": latent.copy(),
                        "ref_chunk": ref.copy(),
                        "action_chunk": executed,
                        "rewards": reward_array,
                        "executed_rewards": list(rewards),
                        "done": done,
                        "next_state": next_state.copy(),
                        "next_latent": next_latent.copy(),
                        "next_ref_chunk": next_ref.copy(),
                        "duration": duration,
                        "step_id": start_step,
                        "phase": phase,
                        "intervened": intervened,
                        "q_advantage": q_advantage,
                        "executed_q_advantage": executed_q_advantage,
                        "requested_residual_rms": float(
                            np.sqrt(np.mean(np.square(requested_residual)))
                        ),
                    }
                )
                obs = next_obs
                state = next_state
                if done:
                    break

            mc_returns = _discounted_mc(
                episode_transitions, args.gamma, args.chunk_len
            )
            for transition, mc_return in zip(
                episode_transitions, mc_returns, strict=True
            ):
                replay.add(
                    state=transition["state"],
                    latent=transition["latent"],
                    ref_chunk=transition["ref_chunk"],
                    action_chunk=transition["action_chunk"],
                    rewards=transition["rewards"],
                    done=transition["done"],
                    next_state=transition["next_state"],
                    next_latent=transition["next_latent"],
                    next_ref_chunk=transition["next_ref_chunk"],
                    duration=transition["duration"],
                    step_id=transition["step_id"],
                    mc_return=mc_return,
                )
                warmup_nonzero_transitions += int(
                    np.any(
                        np.abs(
                            transition["action_chunk"] - transition["ref_chunk"]
                        )
                        > 1e-6
                    )
                )
            warmup_successes += int(episode_success)
            warmup_failures += int(not episode_success)

            update_count = args.updates_per_chunk * len(episode_transitions)
            stratified_used = False
            for _ in range(update_count):
                if len(replay) < args.batch_size:
                    break
                update_actor = _actor_ready(
                    replay_size=len(replay),
                    critic_updates=agent.critic_updates,
                    warmup_successes=warmup_successes,
                    warmup_failures=warmup_failures,
                    warmup_nonzero_transitions=warmup_nonzero_transitions,
                    args=args,
                )
                training_batch, stratified_used = sample_training_batch()
                metrics = agent.update(
                    training_batch,
                    update_actor=update_actor,
                    freeze_policy_context=False,
                )

            recent_successes.append(int(episode_success))
            recent_successes = recent_successes[-50:]
            interventions = [item["intervened"] for item in episode_transitions]
            intervention_steps = [
                int(item["step_id"])
                for item in episode_transitions
                if item["intervened"]
            ]
            q_advantages = [
                float(item["q_advantage"])
                for item in episode_transitions
                if np.isfinite(item["q_advantage"])
            ]
            executed_advantages = [
                float(item["executed_q_advantage"])
                for item in episode_transitions
                if np.isfinite(item["executed_q_advantage"])
            ]
            pool_counts = replay.stratified_pool_counts(
                success_threshold=config.success_bc_min_return
            )
            finite_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if np.isfinite(value)
            }
            record = {
                "episode": episode,
                "seed": episode_seed,
                "env_step": env_steps,
                "steps": episode_step,
                "return": episode_return,
                "success": episode_success,
                "grasp_reward_events": episode_grasp_events,
                "grasped": bool(episode_grasp_events),
                "recent_success_rate": float(np.mean(recent_successes)),
                "replay": len(replay),
                "critic_updates": agent.critic_updates,
                "actor_updates": agent.actor_updates,
                "warmup_successes": warmup_successes,
                "warmup_failures": warmup_failures,
                "warmup_nonzero_transitions": warmup_nonzero_transitions,
                "warmup_explore_chunks": warmup_explores,
                "base_control_episode": base_control_episode,
                "intervention_chunks": int(sum(interventions)),
                "intervention_steps": intervention_steps,
                "intervention_cooldown_rejections": cooldown_rejections,
                "q_gate_intervention_chunks": sum(
                    item["phase"] == "online_q_gate" for item in episode_transitions
                ),
                "epsilon_intervention_chunks": sum(
                    item["phase"] == "online_epsilon" for item in episode_transitions
                ),
                "exploration_override_chunks": sum(
                    item["phase"] == "online_exploration_override"
                    for item in episode_transitions
                ),
                "mean_q_advantage": (
                    float(np.mean(q_advantages)) if q_advantages else None
                ),
                "mean_executed_q_advantage": (
                    float(np.mean(executed_advantages))
                    if executed_advantages
                    else None
                ),
                "residual_rms": float(
                    np.mean(
                        [item["requested_residual_rms"] for item in episode_transitions]
                    )
                ),
                "replay_stratified_used": stratified_used,
                "replay_pool_zero": pool_counts["zero"],
                "replay_pool_successful_nonzero": pool_counts[
                    "successful_nonzero"
                ],
                "replay_pool_failed_nonzero": pool_counts["failed_nonzero"],
                "elapsed_s": time.time() - started,
                **finite_metrics,
            }
            print("episode", record, flush=True)
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, allow_nan=False) + "\n")
            episode += 1
            episode_in_progress = False
            if wandb_run is not None:
                wandb_metrics = {"train/env_step": int(env_steps)}
                for key, value in record.items():
                    if isinstance(value, (bool, int, float, np.integer, np.floating)):
                        numeric = float(value)
                        if np.isfinite(numeric):
                            wandb_metrics[f"train/{key}"] = numeric
                wandb_run.log(wandb_metrics)
            if env_steps - last_save >= args.save_every_env_steps:
                last_save = env_steps
                save_snapshot()
    finally:
        try:
            if episode_in_progress:
                print(
                    {
                        "snapshot_preserved": (
                            "training stopped inside an uncommitted episode; "
                            "last completed snapshot was not overwritten"
                        ),
                        "attempted_env_steps": int(env_steps),
                    },
                    flush=True,
                )
            else:
                last_save = env_steps
                save_snapshot()
        finally:
            env.close()
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "final/env_steps": int(env_steps),
                        "final/episodes": int(episode),
                        "final/actor_updates": int(agent.actor_updates),
                        "final/critic_updates": int(agent.critic_updates),
                    }
                )
                wandb_run.finish()

    print(
        "training complete",
        {
            "checkpoint": str(checkpoint_path),
            "replay": str(replay_path),
            "trainer_state": str(trainer_state_path),
            "episodes": episode,
            "env_steps": env_steps,
            "actor_updates": agent.actor_updates,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
