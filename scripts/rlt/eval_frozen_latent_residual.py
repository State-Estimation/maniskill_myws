#!/usr/bin/env python
"""Paired evaluation and live TCP trajectories for frozen-Pi0 residual TD3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]
BASE_TRAJECTORY_COLOR = np.asarray([0.08, 0.42, 1.0, 1.0], dtype=np.float32)
RLT_TRAJECTORY_COLOR = np.asarray([1.0, 0.30, 0.04, 1.0], dtype=np.float32)
TERMINAL_SUCCESS_SPARSE_REWARD_SCHEMA = {
    "schema": "terminal_success_sparse_reward_v1",
    "applies_to_reward_modes": ["sparse"],
    "grasp_process_reward": 0.0,
    "task_success_reward": 1.0,
    "grasp_reward_once_per_episode": False,
    "milestone_event_key": None,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_checkpoint_environment(
    requested_env_id: str,
    runtime_identity: dict,
) -> None:
    checkpoint_env_id = runtime_identity.get("env_id")
    if checkpoint_env_id is not None and requested_env_id != checkpoint_env_id:
        raise ValueError(
            f"Checkpoint was trained for {checkpoint_env_id!r}, but evaluation "
            f"requested {requested_env_id!r}"
        )


def _environment_reward_schema(
    env,
    *,
    reward_mode: str,
    expected_schema: dict | None,
) -> dict:
    base_env = env.unwrapped
    explicit_schema = getattr(base_env, "grasp_reward_schema", None)
    if explicit_schema is not None:
        return dict(explicit_schema)

    terminal_schema = dict(TERMINAL_SUCCESS_SPARSE_REWARD_SCHEMA)
    if dict(expected_schema or {}) != terminal_schema:
        raise RuntimeError(
            "Environment does not expose grasp_reward_schema and the checkpoint "
            "does not declare the exact terminal-only sparse reward protocol"
        )
    supported_modes = set(getattr(base_env, "SUPPORTED_REWARD_MODES", ()))
    if reward_mode != "sparse" or "sparse" not in supported_modes:
        raise RuntimeError(
            "Terminal-only reward fallback requires an environment that explicitly "
            "supports sparse reward mode"
        )
    return terminal_schema


def _scalar(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _done(value) -> bool:
    return bool(_scalar(value))


def _plan_context(policy, obs: dict, *, config, inference_seed: int):
    ref, latent = policy.plan_with_latent(
        obs,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        inference_seed=inference_seed,
    )
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape != (config.latent_dim,):
        raise ValueError("OpenPI latent does not match checkpoint schema")
    return np.asarray(ref, dtype=np.float32), latent


def _bootstrap_interval(deltas: np.ndarray, seed: int) -> list[float]:
    if deltas.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(10_000, deltas.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _aggregate_rows(rows: list[dict], *, bootstrap_seed: int | None) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    base_success = np.asarray(
        [row["base"]["success"] for row in rows], dtype=np.int8
    )
    rlt_success = np.asarray(
        [row["rlt"]["success"] for row in rows], dtype=np.int8
    )
    base_grasp = np.asarray(
        [row["base"]["grasped"] for row in rows], dtype=np.int8
    )
    rlt_grasp = np.asarray(
        [row["rlt"]["grasped"] for row in rows], dtype=np.int8
    )
    interventions = np.asarray(
        [row["rlt"]["intervention_chunks"] for row in rows], dtype=np.int64
    )
    deltas = rlt_success - base_success
    buckets = {
        "0": interventions == 0,
        "1": interventions == 1,
        "2_plus": interventions >= 2,
    }
    intervention_distribution = {
        name: {
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
            "successes": int(rlt_success[mask].sum()),
            "success_rate": (
                float(rlt_success[mask].mean()) if bool(mask.any()) else None
            ),
        }
        for name, mask in buckets.items()
    }
    return {
        "base_successes": int(base_success.sum()),
        "base_success_rate": float(base_success.mean()),
        "rlt_successes": int(rlt_success.sum()),
        "rlt_success_rate": float(rlt_success.mean()),
        "base_grasped": int(base_grasp.sum()),
        "base_grasp_rate": float(base_grasp.mean()),
        "rlt_grasped": int(rlt_grasp.sum()),
        "rlt_grasp_rate": float(rlt_grasp.mean()),
        "paired_delta": float(deltas.mean()),
        "paired_delta_bootstrap_95ci": (
            _bootstrap_interval(deltas, bootstrap_seed)
            if bootstrap_seed is not None
            else None
        ),
        "base_fail_rlt_success": int(
            np.count_nonzero((base_success == 0) & (rlt_success == 1))
        ),
        "base_success_rlt_fail": int(
            np.count_nonzero((base_success == 1) & (rlt_success == 0))
        ),
        "intervention_distribution": intervention_distribution,
    }


def _tcp_position(observation, key: str) -> np.ndarray:
    from maniskill_myws.rlt.state import (
        as_numpy,
        get_by_path_flexible,
        squeeze_leading_batch,
    )

    pose = squeeze_leading_batch(as_numpy(get_by_path_flexible(observation, key)))
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(
            f"TCP pose '{key}' must have finite single-environment shape (7,), "
            f"got {pose.shape}"
        )
    return np.ascontiguousarray(pose[:3])


def _trajectory_line_geometry(
    points: np.ndarray,
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"trajectory points must have shape [N, 3], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory points contain NaN or Inf")
    if color.shape != (4,) or not np.all(np.isfinite(color)):
        raise ValueError("trajectory color must be a finite RGBA vector")
    if points.shape[0] < 2:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    vertices = np.stack((points[:-1], points[1:]), axis=1).reshape(-1, 3)
    colors = np.repeat(color[None], vertices.shape[0], axis=0)
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(colors, dtype=np.float32),
    )


def _gate_decision(
    *,
    q_advantage: float,
    min_q_advantage: float,
    interventions: int,
    max_interventions: int,
    boundary_index: int | None = None,
    last_intervention_boundary: int | None = None,
    cooldown_chunks: int = 0,
) -> tuple[bool, str]:
    if not np.isfinite(q_advantage) or not np.isfinite(min_q_advantage):
        raise ValueError("Q gate inputs must be finite")
    if max_interventions < 0 or cooldown_chunks < 0:
        raise ValueError("Intervention budget and cooldown must be non-negative")
    if interventions >= max_interventions:
        return False, "BUDGET_EXHAUSTED"
    if (
        last_intervention_boundary is not None
        and boundary_index is not None
        and boundary_index - last_intervention_boundary <= cooldown_chunks
    ):
        return False, "COOLDOWN"
    if q_advantage < min_q_advantage:
        return False, "Q_REJECTED"
    return True, "EXECUTED"


def _validate_chunk(
    chunk: np.ndarray,
    *,
    chunk_len: int,
    action_dim: int,
    low: np.ndarray,
    high: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float32)
    if array.shape != (chunk_len, action_dim) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have finite shape {(chunk_len, action_dim)}, "
            f"got {array.shape}"
        )
    if np.any(array < low[None]) or np.any(array > high[None]):
        raise ValueError(f"{name} violates the environment action bounds")
    return np.ascontiguousarray(array)


class _LivePairedTrajectoryOverlay:
    """Draw Base and residual-RL TCP paths in the RL viewer scene."""

    def __init__(self, viewer, *, line_width: float):
        self.viewer = viewer
        self.render_scene = viewer.render_scene
        self.line_width = float(line_width)
        self.base_node = None
        self.rlt_node = None

    def _scene_is_current(self) -> bool:
        return (
            not self.viewer.closed
            and self.viewer.render_scene is self.render_scene
            and self.viewer.renderer_context is not None
        )

    def _replace(self, node, points: np.ndarray, color: np.ndarray, width: float):
        if node is not None and self._scene_is_current():
            self.render_scene.remove_node(node)
        if not self._scene_is_current():
            return None
        vertices, colors = _trajectory_line_geometry(points, color)
        if vertices.shape[0] == 0:
            return None
        primitive = self.viewer.renderer_context.create_line_set(vertices, colors)
        replacement = self.render_scene.add_line_set(primitive)
        replacement.line_width = float(width)
        return replacement

    def update(self, base_points: np.ndarray, rlt_points: np.ndarray) -> None:
        self.base_node = self._replace(
            self.base_node,
            base_points,
            BASE_TRAJECTORY_COLOR,
            self.line_width + 2.0,
        )
        self.rlt_node = self._replace(
            self.rlt_node,
            rlt_points,
            RLT_TRAJECTORY_COLOR,
            self.line_width,
        )

    def clear(self) -> None:
        if self._scene_is_current():
            for node in (self.base_node, self.rlt_node):
                if node is not None:
                    self.render_scene.remove_node(node)
        self.base_node = None
        self.rlt_node = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="SolarPanelStatic-v2")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pace human rendering to the environment control frequency.",
    )
    parser.add_argument(
        "--live-paired-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw Base and refined TCP paths in one lockstep SAPIEN viewer.",
    )
    parser.add_argument("--trajectory-tcp-key", default="extra/tcp_pose")
    parser.add_argument("--trajectory-update-every", type=int, default=5)
    parser.add_argument("--trajectory-line-width", type=float, default=3.0)
    parser.add_argument("--overlay-hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--enhanced-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-seed", type=int, default=30_000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seeds; overrides --start-seed and --num-seeds.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--min-q-advantage", type=float, default=0.10)
    parser.add_argument("--max-intervention-chunks-per-episode", type=int, default=1)
    parser.add_argument("--intervention-cooldown-chunks", type=int, default=0)
    parser.add_argument("--value-checkpoint", default=None)
    parser.add_argument("--intervention-keep-q-advantage", type=float, default=0.02)
    parser.add_argument("--max-intervention-env-steps", type=int, default=500)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="maniskill-myws-rlt")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--output-dir",
        default="outputs/rlt/SolarPanelStatic-v2_frozen_latent_td3/eval",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.num_seeds <= 0 or args.start_seed < 0:
        parser.error("seed range must be non-negative and non-empty")
    evaluation_seeds = (
        list(args.seeds)
        if args.seeds is not None
        else list(range(args.start_seed, args.start_seed + args.num_seeds))
    )
    if not evaluation_seeds or min(evaluation_seeds) < 0:
        parser.error("evaluation seeds must be non-empty and non-negative")
    if len(evaluation_seeds) != len(set(evaluation_seeds)):
        parser.error("evaluation seeds must be unique")
    if args.resize <= 0:
        parser.error("--resize must be positive")
    if args.render_every <= 0 or args.trajectory_update_every <= 0:
        parser.error("render intervals must be positive")
    if args.trajectory_line_width <= 0 or args.overlay_hold_seconds < 0:
        parser.error("trajectory width must be positive and hold time non-negative")
    if args.live_paired_trajectories and args.render_mode != "human":
        parser.error("--live-paired-trajectories requires --render-mode human")
    if not np.isfinite(args.min_q_advantage):
        parser.error("--min-q-advantage must be finite")
    if not np.isfinite(args.intervention_keep_q_advantage):
        parser.error("--intervention-keep-q-advantage must be finite")
    if (
        args.max_intervention_chunks_per_episode < 0
        or args.intervention_cooldown_chunks < 0
        or args.max_intervention_env_steps < 0
    ):
        parser.error("intervention budget and cooldown must be non-negative")
    if args.control_mode != "pd_joint_pos":
        parser.error("This implementation requires --control-mode pd_joint_pos")
    if args.reward_mode not in {"dense", "sparse"}:
        parser.error("This implementation requires dense or sparse environment reward")
    if args.wandb_enabled and not args.wandb_project.strip():
        parser.error("--wandb-project must be non-empty")

    output_dir = Path(args.output_dir)
    if any(
        (output_dir / name).exists()
        for name in ("paired_results.jsonl", "summary.json")
    ):
        raise FileExistsError(
            f"Refusing to append to an existing evaluation: {output_dir}"
        )

    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.frozen_latent_rl import (
        FROZEN_LATENT_PROTOCOL,
        FrozenLatentResidualAgent,
        make_runtime_identity,
    )
    from maniskill_myws.openpi_bridge.remote_policy import (
        SAFE_LATENT_DIM,
        SAFE_LATENT_PROTOCOL,
    )
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter
    from maniskill_myws.rlt.value_model import (
        VALUE_FEATURE_SCHEMA,
        DistributionalBaseValueModel,
        infer_value_estimate,
        value_images_from_observation,
    )

    maniskill_myws.register()
    agent = FrozenLatentResidualAgent.load(args.checkpoint, device=args.device)
    config = agent.config
    value_enabled = config.latent_protocol == SAFE_LATENT_PROTOCOL
    if value_enabled != bool(args.value_checkpoint):
        parser.error(
            "SAFE value-guided checkpoints require --value-checkpoint; mean-latent "
            "checkpoints must not receive one"
        )
    _assert_checkpoint_environment(args.env_id, agent.runtime_identity)
    expected_reward_schema = agent.runtime_identity.get(
        "environment_reward_schema"
    )
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "enhanced_determinism": args.enhanced_determinism,
        "max_episode_steps": config.max_episode_steps,
    }
    env = gym.make(args.env_id, render_mode=args.render_mode, **env_kwargs)
    base_env = (
        gym.make(args.env_id, render_mode=None, **env_kwargs)
        if args.live_paired_trajectories
        else None
    )
    backend = require_resolved_backend(
        env,
        expected_sim_backend=args.sim_backend,
        expected_render_backend=args.render_backend,
    )
    reward_schema = _environment_reward_schema(
        env,
        reward_mode=args.reward_mode,
        expected_schema=expected_reward_schema,
    )
    if base_env is not None:
        base_backend = require_resolved_backend(
            base_env,
            expected_sim_backend=args.sim_backend,
            expected_render_backend=args.render_backend,
        )
        if base_backend != backend:
            raise RuntimeError("Base and RL environments resolved different backends")
        if _environment_reward_schema(
            base_env,
            reward_mode=args.reward_mode,
            expected_schema=expected_reward_schema,
        ) != reward_schema:
            raise RuntimeError("Base and RL environments have different reward schemas")

    action_dim = int(np.prod(env.action_space.shape))
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if action_dim != config.action_dim:
        raise ValueError("Checkpoint and environment action dimensions differ")
    np.testing.assert_allclose(low, np.asarray(config.action_low), rtol=0, atol=0)
    np.testing.assert_allclose(high, np.asarray(config.action_high), rtol=0, atol=0)
    if base_env is not None:
        np.testing.assert_array_equal(
            np.asarray(base_env.action_space.low, dtype=np.float32).reshape(-1), low
        )
        np.testing.assert_array_equal(
            np.asarray(base_env.action_space.high, dtype=np.float32).reshape(-1), high
        )

    prompt = args.prompt or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    value_model = None
    value_checkpoint_sha256 = None
    if value_enabled:
        value_model, _ = DistributionalBaseValueModel.load(
            args.value_checkpoint, device=args.device
        )
        value_model.requires_grad_(False)
        value_checkpoint_sha256 = _sha256_file(Path(args.value_checkpoint))
    rlt_policy = make_base_chunk_policy(
        "remote_openpi",
        action_space=env.action_space,
        action_dim=action_dim,
        server=args.server,
        prompt=prompt,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=args.state_keys,
        resize=args.resize,
        require_frozen_latent=not value_enabled,
        require_safe_latent=value_enabled,
    )
    metadata = rlt_policy.server_metadata or {}
    expected_latent_protocol = SAFE_LATENT_PROTOCOL if value_enabled else FROZEN_LATENT_PROTOCOL
    expected_latent_dim = SAFE_LATENT_DIM if value_enabled else config.latent_dim
    protocol_key = "safe_latent_protocol" if value_enabled else "frozen_latent_protocol"
    shape_key = "safe_latent_shape" if value_enabled else "frozen_latent_shape"
    if metadata.get(protocol_key) != expected_latent_protocol:
        raise RuntimeError("OpenPI server does not expose the checkpoint latent protocol")
    if metadata.get(shape_key) != [expected_latent_dim]:
        raise RuntimeError("OpenPI server latent shape does not match checkpoint")

    base_policy = None
    if base_env is not None:
        base_policy = make_base_chunk_policy(
            "remote_openpi",
            action_space=base_env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.resize,
        )
        if openpi_policy_identity_sha256(base_policy.server_metadata or {}) != (
            openpi_policy_identity_sha256(metadata)
        ):
            raise RuntimeError("Base and RL OpenPI clients report different identities")

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
        chunk_len=config.chunk_len,
        max_episode_steps=config.max_episode_steps,
        openpi_policy_identity_sha256=openpi_policy_identity_sha256(metadata),
        latent_protocol=config.latent_protocol,
        latent_dim=config.latent_dim,
    )
    runtime_identity["environment_reward_schema"] = reward_schema
    if value_enabled:
        declared_value_identity = agent.runtime_identity.get("value_model")
        if not isinstance(declared_value_identity, dict):
            raise ValueError("Value-guided checkpoint has no value model identity")
        actual_value_fields = {
            "checkpoint_sha256": value_checkpoint_sha256,
            "feature_schema": VALUE_FEATURE_SCHEMA,
            "feature_dim": value_model.config.critic_feature_dim,
        }
        mismatches = {
            key: (declared_value_identity.get(key), actual)
            for key, actual in actual_value_fields.items()
            if declared_value_identity.get(key) != actual
        }
        if mismatches:
            raise ValueError(f"Value checkpoint identity mismatch: {mismatches}")
        runtime_identity["value_model"] = dict(declared_value_identity)
    agent.assert_runtime_identity(runtime_identity)

    state_adapter = StateAdapter(args.state_keys)
    probe_obs, _ = reset_env_fresh_scene(
        env, seed=evaluation_seeds[0], operation="value evaluation state shape probe"
    )
    raw_state_dim = int(np.asarray(state_adapter(probe_obs)).size)
    if value_enabled:
        expected_value_config = {
            "state_dim": raw_state_dim,
            "action_dim": config.action_dim,
            "chunk_len": config.chunk_len,
            "max_episode_steps": config.max_episode_steps,
            "latent_dim": SAFE_LATENT_DIM,
            "num_views": 2,
        }
        value_config_mismatches = {
            key: (expected, getattr(value_model.config, key))
            for key, expected in expected_value_config.items()
            if getattr(value_model.config, key) != expected
        }
        if value_config_mismatches:
            raise ValueError(
                f"Value model runtime configuration mismatch: {value_config_mismatches}"
            )
        expected_state_dim = raw_state_dim + value_model.config.critic_feature_dim
    else:
        expected_state_dim = raw_state_dim
    if config.state_dim != expected_state_dim:
        raise ValueError(
            f"Checkpoint state_dim {config.state_dim} != runtime {expected_state_dim}"
        )
    control_frequency = float(getattr(env.unwrapped, "control_freq", 20.0))
    target_step_seconds = 1.0 / control_frequency if control_frequency > 0 else 0.05

    def raw_state_vector(observation) -> np.ndarray:
        state = np.asarray(state_adapter(observation), dtype=np.float32)
        if state.shape != (raw_state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"Raw state must have finite shape {(raw_state_dim,)}, "
                f"got {state.shape}"
            )
        return state

    def augmented_state(
        observation: dict,
        *,
        latent: np.ndarray,
        ref: np.ndarray,
        step_id: int,
    ) -> tuple[np.ndarray, object | None]:
        raw = raw_state_vector(observation)
        if not value_enabled:
            return raw, None
        images = value_images_from_observation(
            observation,
            image_keys=(args.image_key, args.wrist_image_key),
            height=value_model.config.image_height,
            width=value_model.config.image_width,
        )
        estimate = infer_value_estimate(
            value_model,
            images=images,
            state=raw,
            latent=latent,
            ref_chunk=ref,
            step_id=step_id,
        )
        return np.concatenate([raw, estimate.critic_features]).astype(np.float32), estimate

    def validate_reference(chunk: np.ndarray) -> np.ndarray:
        return _validate_chunk(
            chunk,
            chunk_len=config.chunk_len,
            action_dim=config.action_dim,
            low=low,
            high=high,
            name="OpenPI reference chunk",
        )

    def choose_chunk(
        observation,
        *,
        ref: np.ndarray,
        latent: np.ndarray,
        step_id: int,
        interventions: int,
        intervention_env_steps: int,
        recovery_active: bool,
        last_intervention_boundary: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, str, dict]:
        state, value_estimate = augmented_state(
            observation, latent=latent, ref=ref, step_id=step_id
        )
        residual = np.asarray(
            agent.select_residual(
                state,
                latent,
                ref,
                step_id=step_id,
                deterministic=True,
            ),
            dtype=np.float32,
        )
        if residual.shape != ref.shape or not np.all(np.isfinite(residual)):
            raise ValueError("Residual actor returned an invalid chunk")
        q_advantage = agent.conservative_advantage(
            state,
            latent,
            ref,
            residual,
            step_id=step_id,
        )
        boundary_index = step_id // config.chunk_len
        if value_enabled:
            threshold = (
                args.intervention_keep_q_advantage
                if recovery_active
                else args.min_q_advantage
            )
            if intervention_env_steps >= args.max_intervention_env_steps:
                intervene, status = False, "STEP_BUDGET_EXHAUSTED"
            elif q_advantage < threshold:
                intervene, status = False, "Q_REJECTED"
            else:
                intervene, status = True, "EXECUTED"
        else:
            threshold = args.min_q_advantage
            intervene, status = _gate_decision(
                q_advantage=q_advantage,
                min_q_advantage=threshold,
                interventions=interventions,
                max_interventions=args.max_intervention_chunks_per_episode,
                boundary_index=boundary_index,
                last_intervention_boundary=last_intervention_boundary,
                cooldown_chunks=args.intervention_cooldown_chunks,
            )
        executed_residual = residual if intervene else np.zeros_like(residual)
        action_chunk = (
            agent.apply_residual(ref, residual) if intervene else ref.copy()
        )
        action_chunk = _validate_chunk(
            action_chunk,
            chunk_len=config.chunk_len,
            action_dim=config.action_dim,
            low=low,
            high=high,
            name="refined action chunk",
        )
        trace = {
            "step": int(step_id),
            "boundary_index": int(boundary_index),
            "q_advantage": float(q_advantage),
            "gate_status": status,
            "interventions_before": int(interventions),
            "intervention_env_steps_before": int(intervention_env_steps),
            "gate_threshold": float(threshold),
            "proposal_residual_rms": float(
                np.sqrt(np.mean(np.square(residual), dtype=np.float64))
            ),
            "proposal_residual_max": float(np.max(np.abs(residual))),
            "value_potential": (
                value_estimate.potential if value_estimate is not None else None
            ),
            "value_failure_probability": (
                value_estimate.failure_probability
                if value_estimate is not None
                else None
            ),
            "value_entropy": (
                value_estimate.entropy if value_estimate is not None else None
            ),
        }
        return (
            action_chunk,
            executed_residual,
            float(q_advantage),
            intervene,
            status,
            trace,
        )

    def finish_result(
        role: dict,
        *,
        residual_norms: list[float],
        q_advantages: list[float],
        intervention_steps: list[int],
        intervention_q_advantages: list[float],
        boundary_trace: list[dict],
    ) -> dict:
        interventions = len(intervention_steps)
        return {
            "success": bool(role["success"]),
            "return": float(role["return"]),
            "steps": int(role["steps"]),
            "residual_rms": (
                float(np.mean(residual_norms)) if residual_norms else 0.0
            ),
            "residual_max": (
                float(np.max(residual_norms)) if residual_norms else 0.0
            ),
            "intervention_chunks": interventions,
            "intervention_env_steps": int(role["intervention_env_steps"]),
            "intervention_steps": intervention_steps,
            "intervention_q_advantages": intervention_q_advantages,
            "grasp_reward_events": int(role["grasp_reward_events"]),
            "grasped": bool(role["grasp_reward_events"]),
            "mean_q_advantage": (
                float(np.mean(q_advantages)) if q_advantages else None
            ),
            "max_q_advantage": (
                float(np.max(q_advantages)) if q_advantages else None
            ),
            "final_budget_remaining": max(
                0,
                (
                    args.max_intervention_env_steps
                    - int(role["intervention_env_steps"])
                    if value_enabled
                    else args.max_intervention_chunks_per_episode - interventions
                ),
            ),
            "boundary_trace": boundary_trace,
        }

    def step_role(role: dict, role_env, *, tcp_points: list[np.ndarray] | None) -> None:
        action = np.asarray(role["chunk"][role["cursor"]], dtype=np.float32)
        if (
            action.shape != (config.action_dim,)
            or not np.all(np.isfinite(action))
            or np.any(action < low)
            or np.any(action > high)
        ):
            raise ValueError("Evaluation action violates the environment schema")
        observation, reward, terminated, truncated, info = role_env.step(action)
        role["obs"] = observation
        role["return"] += _scalar(reward)
        role["steps"] += 1
        if role.get("chunk_intervened", False):
            role["intervention_env_steps"] += 1
        role["cursor"] += 1
        if isinstance(info, dict):
            role["success"] = role["success"] or _done(info.get("success", False))
            role["grasp_reward_events"] += int(
                _done(info.get("grasp_reward_event", False))
            )
        role["done"] = bool(
            _done(terminated)
            or _done(truncated)
            or role["steps"] >= config.max_episode_steps
        )
        if tcp_points is not None:
            tcp_points.append(_tcp_position(observation, args.trajectory_tcp_key))

    def run_episode(seed: int, *, use_rl: bool) -> dict:
        observation, _ = reset_env_fresh_scene(
            env,
            seed=seed,
            operation=("frozen latent TD3 eval" if use_rl else "paired Base eval"),
        )
        rlt_policy.reset()
        role = {
            "obs": observation,
            "steps": 0,
            "return": 0.0,
            "success": False,
            "grasp_reward_events": 0,
            "done": False,
            "chunk": None,
            "cursor": 0,
            "chunk_intervened": False,
            "intervention_env_steps": 0,
        }
        residual_norms: list[float] = []
        q_advantages: list[float] = []
        intervention_steps: list[int] = []
        intervention_q_advantages: list[float] = []
        boundary_trace: list[dict] = []
        last_intervention_boundary: int | None = None
        recovery_active = False

        while not role["done"]:
            inference_seed = inference_seed_for_step(seed, role["steps"])
            if use_rl:
                ref, latent = _plan_context(
                    rlt_policy,
                    role["obs"],
                    config=config,
                    inference_seed=inference_seed,
                )
                ref = validate_reference(ref)
                (
                    action_chunk,
                    residual,
                    q_advantage,
                    intervene,
                    status,
                    trace,
                ) = choose_chunk(
                    role["obs"],
                    ref=ref,
                    latent=latent,
                    step_id=role["steps"],
                    interventions=len(intervention_steps),
                    intervention_env_steps=role["intervention_env_steps"],
                    recovery_active=recovery_active,
                    last_intervention_boundary=last_intervention_boundary,
                )
                recovery_active = intervene
                q_advantages.append(q_advantage)
                boundary_trace.append(trace)
                if intervene:
                    intervention_steps.append(role["steps"])
                    intervention_q_advantages.append(q_advantage)
                    last_intervention_boundary = role["steps"] // config.chunk_len
                    print(
                        {
                            "RL_INTERVENTION": {
                                "seed": seed,
                                "step": role["steps"],
                                "q_advantage": q_advantage,
                                "residual_rms": trace["proposal_residual_rms"],
                                "residual_max": trace["proposal_residual_max"],
                            }
                        },
                        flush=True,
                    )
                else:
                    print(
                        {
                            "RL_GATE": {
                                "seed": seed,
                                "step": role["steps"],
                                "status": status,
                                "q_advantage": q_advantage,
                            }
                        },
                        flush=True,
                    )
                residual_norms.append(
                    float(np.sqrt(np.mean(np.square(residual), dtype=np.float64)))
                )
            else:
                action_chunk = validate_reference(
                    rlt_policy.plan(
                        role["obs"],
                        chunk_len=config.chunk_len,
                        action_dim=config.action_dim,
                        inference_seed=inference_seed,
                    )
                )
                intervene = False

            role["chunk"] = action_chunk
            role["cursor"] = 0
            role["chunk_intervened"] = bool(intervene)
            for _ in range(len(action_chunk)):
                wall_start = time.perf_counter()
                step_role(role, env, tcp_points=None)
                if args.render_mode is not None and (
                    role["steps"] % args.render_every == 0 or role["done"]
                ):
                    env.render()
                if args.real_time:
                    remaining = target_step_seconds - (
                        time.perf_counter() - wall_start
                    )
                    if remaining > 0:
                        time.sleep(remaining)
                if role["done"]:
                    break

        if use_rl and not intervention_steps:
            print(
                {
                    "RL_NO_INTERVENTION": {
                        "seed": seed,
                        "max_q_advantage": (
                            float(np.max(q_advantages)) if q_advantages else None
                        ),
                        "required_q_advantage": args.min_q_advantage,
                    }
                },
                flush=True,
            )
        return finish_result(
            role,
            residual_norms=residual_norms,
            q_advantages=q_advantages,
            intervention_steps=intervention_steps,
            intervention_q_advantages=intervention_q_advantages,
            boundary_trace=boundary_trace,
        )

    def run_live_paired_episode(seed: int) -> tuple[dict, dict]:
        if base_env is None or base_policy is None:
            raise RuntimeError("Live trajectories require the paired Base environment")
        base_obs, _ = reset_env_fresh_scene(
            base_env, seed=seed, operation="live paired Base trajectory"
        )
        rlt_obs, _ = reset_env_fresh_scene(
            env, seed=seed, operation="live paired residual-RL trajectory"
        )
        base_policy.reset()
        rlt_policy.reset()
        np.testing.assert_allclose(
            raw_state_vector(base_obs),
            raw_state_vector(rlt_obs),
            rtol=0,
            atol=1e-6,
            err_msg="Base/RL initial states differ for the same seed",
        )
        base = {
            "obs": base_obs,
            "steps": 0,
            "return": 0.0,
            "success": False,
            "grasp_reward_events": 0,
            "done": False,
            "chunk": None,
            "cursor": 0,
            "chunk_intervened": False,
            "intervention_env_steps": 0,
        }
        rlt = {
            "obs": rlt_obs,
            "steps": 0,
            "return": 0.0,
            "success": False,
            "grasp_reward_events": 0,
            "done": False,
            "chunk": None,
            "cursor": 0,
            "chunk_intervened": False,
            "intervention_env_steps": 0,
        }
        residual_norms: list[float] = []
        q_advantages: list[float] = []
        intervention_steps: list[int] = []
        intervention_q_advantages: list[float] = []
        boundary_trace: list[dict] = []
        last_intervention_boundary: int | None = None
        recovery_active = False
        base_tcp = [_tcp_position(base_obs, args.trajectory_tcp_key)]
        rlt_tcp = [_tcp_position(rlt_obs, args.trajectory_tcp_key)]
        viewer = env.unwrapped.render_human()
        overlay = _LivePairedTrajectoryOverlay(
            viewer, line_width=args.trajectory_line_width
        )
        global_tick = 0

        try:
            while not (base["done"] and rlt["done"]):
                if not base["done"] and (
                    base["chunk"] is None or base["cursor"] >= len(base["chunk"])
                ):
                    base["chunk"] = validate_reference(
                        base_policy.plan(
                            base["obs"],
                            chunk_len=config.chunk_len,
                            action_dim=config.action_dim,
                            inference_seed=inference_seed_for_step(
                                seed, base["steps"]
                            ),
                        )
                    )
                    base["cursor"] = 0

                if not rlt["done"] and (
                    rlt["chunk"] is None or rlt["cursor"] >= len(rlt["chunk"])
                ):
                    ref, latent = _plan_context(
                        rlt_policy,
                        rlt["obs"],
                        config=config,
                        inference_seed=inference_seed_for_step(seed, rlt["steps"]),
                    )
                    ref = validate_reference(ref)
                    if (
                        not intervention_steps
                        and not base["done"]
                        and base["steps"] == rlt["steps"]
                        and base["cursor"] == 0
                    ):
                        np.testing.assert_allclose(
                            raw_state_vector(base["obs"]),
                            raw_state_vector(rlt["obs"]),
                            rtol=0,
                            atol=1e-6,
                            err_msg="Base/RL states diverged before intervention",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(base["chunk"], dtype=np.float32),
                            ref,
                            err_msg="Base/RL Pi0 chunks differ before intervention",
                        )
                    (
                        rlt["chunk"],
                        residual,
                        q_advantage,
                        intervene,
                        status,
                        trace,
                    ) = choose_chunk(
                        rlt["obs"],
                        ref=ref,
                        latent=latent,
                        step_id=rlt["steps"],
                        interventions=len(intervention_steps),
                        intervention_env_steps=rlt["intervention_env_steps"],
                        recovery_active=recovery_active,
                        last_intervention_boundary=last_intervention_boundary,
                    )
                    recovery_active = intervene
                    rlt["cursor"] = 0
                    rlt["chunk_intervened"] = bool(intervene)
                    q_advantages.append(q_advantage)
                    boundary_trace.append(trace)
                    residual_norms.append(
                        float(
                            np.sqrt(
                                np.mean(np.square(residual), dtype=np.float64)
                            )
                        )
                    )
                    if intervene:
                        intervention_steps.append(rlt["steps"])
                        intervention_q_advantages.append(q_advantage)
                        last_intervention_boundary = (
                            rlt["steps"] // config.chunk_len
                        )
                        print(
                            {
                                "RL_INTERVENTION": {
                                    "seed": seed,
                                    "step": rlt["steps"],
                                    "q_advantage": q_advantage,
                                    "residual_rms": trace[
                                        "proposal_residual_rms"
                                    ],
                                    "residual_max": trace[
                                        "proposal_residual_max"
                                    ],
                                }
                            },
                            flush=True,
                        )
                    else:
                        print(
                            {
                                "RL_GATE": {
                                    "seed": seed,
                                    "step": rlt["steps"],
                                    "status": status,
                                    "q_advantage": q_advantage,
                                }
                            },
                            flush=True,
                        )

                wall_start = time.perf_counter()
                if not base["done"]:
                    step_role(base, base_env, tcp_points=base_tcp)
                if not rlt["done"]:
                    step_role(rlt, env, tcp_points=rlt_tcp)
                if not intervention_steps:
                    if base["steps"] != rlt["steps"]:
                        raise RuntimeError(
                            "Base/RL step counts diverged before intervention"
                        )
                    np.testing.assert_allclose(
                        raw_state_vector(base["obs"]),
                        raw_state_vector(rlt["obs"]),
                        rtol=0,
                        atol=1e-6,
                        err_msg="Base/RL states diverged before intervention",
                    )
                    if (
                        base["done"] != rlt["done"]
                        or base["success"] != rlt["success"]
                    ):
                        raise RuntimeError(
                            "Base/RL outcomes diverged before intervention"
                        )

                global_tick += 1
                finished = base["done"] and rlt["done"]
                if global_tick % args.trajectory_update_every == 0 or finished:
                    overlay.update(
                        np.asarray(base_tcp, dtype=np.float32),
                        np.asarray(rlt_tcp, dtype=np.float32),
                    )
                if global_tick % args.render_every == 0 or finished:
                    if viewer.closed:
                        raise RuntimeError("SAPIEN viewer closed during rollout")
                    env.render()
                if args.real_time:
                    remaining = target_step_seconds - (
                        time.perf_counter() - wall_start
                    )
                    if remaining > 0:
                        time.sleep(remaining)

            if args.overlay_hold_seconds > 0:
                deadline = time.perf_counter() + args.overlay_hold_seconds
                while time.perf_counter() < deadline and not viewer.closed:
                    env.render()
                    time.sleep(
                        min(1.0 / 30.0, max(0.0, deadline - time.perf_counter()))
                    )
        finally:
            overlay.clear()

        if not intervention_steps:
            print(
                {
                    "RL_NO_INTERVENTION": {
                        "seed": seed,
                        "max_q_advantage": (
                            float(np.max(q_advantages)) if q_advantages else None
                        ),
                        "required_q_advantage": args.min_q_advantage,
                    }
                },
                flush=True,
            )
        return (
            finish_result(
                base,
                residual_norms=[],
                q_advantages=[],
                intervention_steps=[],
                intervention_q_advantages=[],
                boundary_trace=[],
            ),
            finish_result(
                rlt,
                residual_norms=residual_norms,
                q_advantages=q_advantages,
                intervention_steps=intervention_steps,
                intervention_q_advantages=intervention_q_advantages,
                boundary_trace=boundary_trace,
            ),
        )

    if args.live_paired_trajectories:
        print(
            {
                "live_paired_trajectories": {
                    "base_tcp": "blue",
                    "refined_rl_tcp": "orange",
                    "execution": "same-seed lockstep",
                }
            },
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb_enabled:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B logging requested but wandb is unavailable") from error
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            tags=list(args.wandb_tags),
            config={
                **vars(args),
                "evaluation_seeds": evaluation_seeds,
                "runtime_identity": agent.runtime_identity,
                "backend": backend,
            },
            mode="online",
            dir=str(output_dir),
        )
        if wandb_run is None:
            raise RuntimeError("W&B did not return a run handle")
        wandb_run.define_metric("eval/index")
        wandb_run.define_metric("eval/*", step_metric="eval/index")
        print({"wandb_run": wandb_run.id, "wandb_url": wandb_run.url}, flush=True)

    rows: list[dict] = []
    try:
        for index, seed in enumerate(evaluation_seeds):
            if args.live_paired_trajectories:
                base_result, rlt_result = run_live_paired_episode(seed)
            else:
                base_result = run_episode(seed, use_rl=False)
                rlt_result = run_episode(seed, use_rl=True)
            row = {
                "index": index,
                "seed": seed,
                "base": base_result,
                "rlt": rlt_result,
            }
            rows.append(row)
            print("pair", row, flush=True)
            with (output_dir / "paired_results.jsonl").open(
                "a", encoding="utf-8"
            ) as file:
                file.write(json.dumps(row) + "\n")

            if wandb_run is not None:
                cumulative = _aggregate_rows(rows, bootstrap_seed=None)
                wandb_run.log(
                    {
                        "eval/index": index,
                        "eval/seed": seed,
                        "eval/base_success": int(base_result["success"]),
                        "eval/rl_success": int(rlt_result["success"]),
                        "eval/paired_delta": int(rlt_result["success"])
                        - int(base_result["success"]),
                        "eval/base_grasped": int(base_result["grasped"]),
                        "eval/rl_grasped": int(rlt_result["grasped"]),
                        "eval/intervention_chunks": int(
                            rlt_result["intervention_chunks"]
                        ),
                        "eval/intervention_env_steps": int(
                            rlt_result["intervention_env_steps"]
                        ),
                        "eval/max_q_advantage": float(
                            rlt_result["max_q_advantage"] or 0.0
                        ),
                        "eval/cumulative_base_success_rate": cumulative[
                            "base_success_rate"
                        ],
                        "eval/cumulative_rl_success_rate": cumulative[
                            "rlt_success_rate"
                        ],
                        "eval/cumulative_base_grasp_rate": cumulative[
                            "base_grasp_rate"
                        ],
                        "eval/cumulative_rl_grasp_rate": cumulative[
                            "rlt_grasp_rate"
                        ],
                        "eval/cumulative_rescues": cumulative[
                            "base_fail_rlt_success"
                        ],
                        "eval/cumulative_regressions": cumulative[
                            "base_success_rlt_fail"
                        ],
                    },
                    step=index,
                )
    finally:
        if base_env is not None:
            base_env.close()
        env.close()

    summary = {
        "schema": "frozen_pi0_continuous_residual_paired_eval_v4",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backend": backend,
        "start_seed": args.start_seed if args.seeds is None else None,
        "evaluation_seeds": evaluation_seeds,
        "num_seeds": len(evaluation_seeds),
        **_aggregate_rows(rows, bootstrap_seed=args.bootstrap_seed),
        "representation": (
            "safe_four_endpoint_tokens_plus_frozen_distributional_value_features"
            if value_enabled
            else "mean_pooled_action_suffix"
        ),
        "live_lockstep_trajectories": bool(args.live_paired_trajectories),
        "trajectory_colors": (
            {"base": "blue", "refined_rl": "orange"}
            if args.live_paired_trajectories
            else None
        ),
        "intervention_policy": {
            "mode": "q_hysteresis_dynamic" if value_enabled else "fixed_chunk_budget",
            "max_chunks_per_episode": (
                None if value_enabled else args.max_intervention_chunks_per_episode
            ),
            "max_intervention_env_steps": (
                args.max_intervention_env_steps if value_enabled else None
            ),
            "cooldown_chunks": 0 if value_enabled else args.intervention_cooldown_chunks,
            "q_gate_min_advantage": args.min_q_advantage,
            "q_gate_keep_advantage": (
                args.intervention_keep_q_advantage if value_enabled else None
            ),
            "critic_conditioned_on_gate_state": False,
        },
        "value_checkpoint": (
            str(Path(args.value_checkpoint).resolve()) if value_enabled else None
        ),
        "runtime_identity": agent.runtime_identity,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        interval = summary["paired_delta_bootstrap_95ci"]
        wandb_run.summary.update(
            {
                "final/base_successes": summary["base_successes"],
                "final/base_success_rate": summary["base_success_rate"],
                "final/rl_successes": summary["rlt_successes"],
                "final/rl_success_rate": summary["rlt_success_rate"],
                "final/paired_delta": summary["paired_delta"],
                "final/paired_delta_ci_low": interval[0],
                "final/paired_delta_ci_high": interval[1],
                "final/rescues": summary["base_fail_rlt_success"],
                "final/regressions": summary["base_success_rlt_fail"],
                "final/base_grasp_rate": summary["base_grasp_rate"],
                "final/rl_grasp_rate": summary["rlt_grasp_rate"],
            }
        )
        wandb_run.finish()
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
