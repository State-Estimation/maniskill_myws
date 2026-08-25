"""Frozen-Pi0-latent Advantage-BC residual TD3 (mean or SAFE endpoint mode)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.value_model import SafeEndpointTokenEncoder


FROZEN_LATENT_PROTOCOL = "maniskill_frozen_pi0_action_suffix_mean_v1"
FROZEN_LATENT_DIM = 1024
FROZEN_LATENT_CHECKPOINT_SCHEMA = (
    "mean_frozen_latent_advantage_bc_v1"
)
LEGACY_V4_FROZEN_LATENT_CHECKPOINT_SCHEMA = (
    "lightweight_rlt_frozen_pi0_continuous_residual_v4"
)
SAFE_VALUE_GUIDED_CHECKPOINT_SCHEMA = (
    "safe_full_latent_value_guided_advantage_bc_v1"
)
MEAN_LATENT_ENCODER = "mean_projection"
SAFE_ENDPOINT_LATENT_ENCODER = "safe_endpoint_attention"


def make_runtime_identity(
    *,
    env_id: str,
    obs_mode: str,
    reward_mode: str,
    control_mode: str,
    sim_backend: str,
    render_backend: str,
    enhanced_determinism: bool,
    prompt: str,
    image_key: str,
    wrist_image_key: str,
    state_keys: list[str] | tuple[str, ...],
    resize: int,
    chunk_len: int,
    max_episode_steps: int,
    openpi_policy_identity_sha256: str,
    latent_protocol: str = FROZEN_LATENT_PROTOCOL,
    latent_dim: int = FROZEN_LATENT_DIM,
) -> dict[str, Any]:
    """Build the immutable deployment identity used by v2 runtime checks."""

    if len(openpi_policy_identity_sha256) != 64:
        raise ValueError("OpenPI policy identity must be a SHA-256 digest")
    identity: dict[str, Any] = {
        "schema": "lightweight_rlt_runtime_identity_v2",
        "env_id": str(env_id),
        "obs_mode": str(obs_mode),
        "reward_mode": str(reward_mode),
        "control_mode": str(control_mode),
        "sim_backend": str(sim_backend),
        "render_backend": str(render_backend),
        "enhanced_determinism": bool(enhanced_determinism),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_key": str(image_key),
        "wrist_image_key": str(wrist_image_key),
        "state_keys": [str(key) for key in state_keys],
        "resize": int(resize),
        "chunk_len": int(chunk_len),
        "max_episode_steps": int(max_episode_steps),
        "openpi_policy_identity_sha256": str(openpi_policy_identity_sha256),
        "frozen_latent_protocol": str(latent_protocol),
        "frozen_latent_dim": int(latent_dim),
    }
    return identity


def runtime_identity_sha256(identity: dict[str, Any]) -> str:
    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(num_layers)):
        layers.extend(
            [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        )
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class FrozenLatentRLConfig:
    state_dim: int
    latent_dim: int = FROZEN_LATENT_DIM
    action_dim: int = 8
    chunk_len: int = 50
    max_episode_steps: int = 500
    context_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    num_critics: int = 2
    arm_residual_fraction: float = 0.06
    gripper_residual_fraction: float = 0.10
    fixed_std: float = 0.05
    actor_residual_limit: float = 0.35
    target_noise: float = 0.02
    target_noise_clip: float = 0.05
    gamma: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    target_tau: float = 0.005
    actor_update_period: int = 2
    actor_l2_weight: float = 0.5
    actor_smoothness_weight: float = 0.2
    mc_loss_weight: float = 0.5
    conservative_q_weight: float = 1.0
    conservative_random_std: float = 0.35
    success_bc_min_return: float = 0.5
    actor_success_bc_weight: float = 2.0
    actor_success_bc_min_residual_rms: float = 1e-4
    actor_success_bc_min_q_advantage: float = 0.0
    exploration_knots: int = 6
    grad_clip_norm: float = 1.0
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    latent_protocol: str = FROZEN_LATENT_PROTOCOL
    latent_encoder: str = MEAN_LATENT_ENCODER

    def __post_init__(self) -> None:
        if self.action_dim != 8:
            raise ValueError("Frozen-latent residual RL requires action_dim=8")
        if min(
            self.state_dim,
            self.latent_dim,
            self.chunk_len,
            self.max_episode_steps,
            self.context_dim,
            self.hidden_dim,
            self.num_layers,
            self.num_critics,
            self.exploration_knots,
        ) <= 0:
            raise ValueError("Model dimensions and horizons must be positive")
        if self.num_critics < 2:
            raise ValueError("TD3 requires at least two critics")
        if self.exploration_knots > self.chunk_len:
            raise ValueError("exploration_knots cannot exceed chunk_len")
        if not 0.0 < self.arm_residual_fraction <= 0.5:
            raise ValueError("arm_residual_fraction must lie in (0, 0.5]")
        if not 0.0 < self.gripper_residual_fraction <= 0.5:
            raise ValueError("gripper_residual_fraction must lie in (0, 0.5]")
        if not 0.0 <= self.fixed_std <= 1.0:
            raise ValueError("fixed_std must lie in [0, 1]")
        if not 0.0 < self.actor_residual_limit <= 1.0:
            raise ValueError("actor_residual_limit must lie in (0, 1]")
        if not 0.0 <= self.target_noise_clip <= 1.0:
            raise ValueError("target_noise_clip must lie in [0, 1]")
        if not 0.0 <= self.target_noise <= 1.0:
            raise ValueError("target_noise must lie in [0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if min(self.actor_lr, self.critic_lr) <= 0.0:
            raise ValueError("Learning rates must be positive")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must lie in (0, 1]")
        if self.actor_update_period < 1:
            raise ValueError("actor_update_period must be positive")
        if min(
            self.actor_l2_weight,
            self.actor_smoothness_weight,
            self.mc_loss_weight,
            self.conservative_q_weight,
            self.conservative_random_std,
            self.actor_success_bc_weight,
            self.actor_success_bc_min_residual_rms,
            self.actor_success_bc_min_q_advantage,
            self.grad_clip_norm,
        ) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if self.conservative_random_std > 1.0:
            raise ValueError("conservative_random_std must lie in [0, 1]")
        if self.actor_success_bc_min_residual_rms > 1.0:
            raise ValueError(
                "actor_success_bc_min_residual_rms must lie in [0, 1]"
            )
        if not 0.0 < self.success_bc_min_return < 1.0:
            raise ValueError("success_bc_min_return must lie in (0, 1)")
        if self.action_low is None or self.action_high is None:
            raise ValueError("action_low and action_high are required")
        if len(self.action_low) != self.action_dim or len(self.action_high) != self.action_dim:
            raise ValueError("Action bounds must match action_dim")
        low = np.asarray(self.action_low, dtype=np.float32)
        high = np.asarray(self.action_high, dtype=np.float32)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low >= high):
            raise ValueError("Action bounds must be finite with low < high")
        valid_latent = (
            self.latent_protocol == FROZEN_LATENT_PROTOCOL
            and self.latent_encoder == MEAN_LATENT_ENCODER
        ) or (
            self.latent_protocol == SAFE_LATENT_PROTOCOL
            and self.latent_dim == SAFE_LATENT_DIM
            and self.latent_encoder == SAFE_ENDPOINT_LATENT_ENCODER
        )
        if not valid_latent:
            raise ValueError("Unsupported frozen latent protocol/encoder combination")

@dataclass(slots=True)
class FrozenLatentBatch:
    states: np.ndarray
    latents: np.ndarray
    ref_chunks: np.ndarray
    action_chunks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_states: np.ndarray
    next_latents: np.ndarray
    next_ref_chunks: np.ndarray
    durations: np.ndarray
    step_ids: np.ndarray
    mc_returns: np.ndarray


class FrozenLatentReplayBuffer:
    """Exact executed-action replay for mean frozen latents."""

    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int,
        latent_dim: int,
        chunk_len: int,
        action_dim: int = 8,
        seed: int = 0,
    ) -> None:
        if min(capacity, state_dim, latent_dim, chunk_len, action_dim) <= 0:
            raise ValueError("Replay dimensions and capacity must be positive")
        if action_dim != 8:
            raise ValueError("Frozen-latent replay requires action_dim=8")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self.latent_shape = (latent_dim,)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.latents = np.zeros((capacity, *self.latent_shape), dtype=np.float16)
        self.ref_chunks = np.zeros((capacity, chunk_len, action_dim), dtype=np.float32)
        self.action_chunks = np.zeros_like(self.ref_chunks)
        self.rewards = np.zeros((capacity, chunk_len), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.next_latents = np.zeros_like(self.latents)
        self.next_ref_chunks = np.zeros_like(self.ref_chunks)
        self.durations = np.zeros((capacity,), dtype=np.int32)
        self.step_ids = np.zeros((capacity,), dtype=np.int32)
        self.mc_returns = np.full((capacity,), np.nan, dtype=np.float32)
        self.pos = 0
        self.full = False
        self.last_load_was_exact = False
        self.last_loaded_snapshot_id: str | None = None
        self.last_migration_stats: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    @property
    def schema_version(self) -> int:
        return 3

    def _indices(self) -> np.ndarray:
        return np.arange(self.capacity if self.full else self.pos, dtype=np.int64)

    def _validate(self, name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    def add(
        self,
        *,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        action_chunk: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        next_state: np.ndarray,
        next_latent: np.ndarray,
        next_ref_chunk: np.ndarray,
        duration: int,
        step_id: int,
        mc_return: float = float("nan"),
    ) -> None:
        duration = int(duration)
        if not 1 <= duration <= self.chunk_len or int(step_id) < 0:
            raise ValueError("Invalid replay duration or step_id")
        idx = self.pos
        self.states[idx] = self._validate("state", state, (self.state_dim,))
        latent_value = self._validate("latent", latent, self.latent_shape)
        if np.any(np.abs(latent_value) > np.finfo(np.float16).max):
            raise ValueError("latent overflows float16 replay storage")
        self.latents[idx] = latent_value.astype(np.float16)
        self.ref_chunks[idx] = self._validate(
            "ref_chunk", ref_chunk, (self.chunk_len, self.action_dim)
        )
        action = self._validate(
            "action_chunk", action_chunk, (self.chunk_len, self.action_dim)
        )
        if not np.array_equal(action[duration:], self.ref_chunks[idx, duration:]):
            raise ValueError("Padded action suffix must equal reference suffix")
        self.action_chunks[idx] = action
        self.rewards[idx] = self._validate("rewards", rewards, (self.chunk_len,))
        self.dones[idx] = float(bool(done))
        self.next_states[idx] = self._validate("next_state", next_state, (self.state_dim,))
        next_value = self._validate("next_latent", next_latent, self.latent_shape)
        if np.any(np.abs(next_value) > np.finfo(np.float16).max):
            raise ValueError("next_latent overflows float16 replay storage")
        self.next_latents[idx] = next_value.astype(np.float16)
        self.next_ref_chunks[idx] = self._validate(
            "next_ref_chunk", next_ref_chunk, (self.chunk_len, self.action_dim)
        )
        if not np.isfinite(mc_return) and not np.isnan(mc_return):
            raise ValueError("mc_return must be finite or NaN")
        self.durations[idx] = duration
        self.step_ids[idx] = int(step_id)
        self.mc_returns[idx] = float(mc_return)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> FrozenLatentBatch:
        if len(self) == 0:
            raise ValueError("Cannot sample an empty replay")
        indices = self._rng.choice(
            self._indices(), size=int(batch_size), replace=len(self) < int(batch_size)
        )
        return self.batch(indices)

    def stratified_pool_counts(self, *, success_threshold: float) -> dict[str, int]:
        if not np.isfinite(success_threshold):
            raise ValueError("success_threshold must be finite")
        indices = self._indices()
        if indices.size == 0:
            return {"zero": 0, "successful_nonzero": 0, "failed_nonzero": 0}
        nonzero = np.any(
            np.abs(self.action_chunks[indices] - self.ref_chunks[indices]) > 1e-6,
            axis=(1, 2),
        )
        successful = np.isfinite(self.mc_returns[indices]) & (
            self.mc_returns[indices] > float(success_threshold)
        )
        return {
            "zero": int(np.count_nonzero(~nonzero)),
            "successful_nonzero": int(np.count_nonzero(nonzero & successful)),
            "failed_nonzero": int(np.count_nonzero(nonzero & ~successful)),
        }

    def has_stratified_support(
        self,
        batch_size: int,
        *,
        nonzero_fraction: float,
        nonzero_success_fraction: float,
        success_threshold: float,
    ) -> bool:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= nonzero_fraction <= 1.0:
            raise ValueError("nonzero_fraction must lie in [0, 1]")
        if not 0.0 <= nonzero_success_fraction <= 1.0:
            raise ValueError("nonzero_success_fraction must lie in [0, 1]")
        if not np.isfinite(success_threshold):
            raise ValueError("success_threshold must be finite")
        counts = self.stratified_pool_counts(success_threshold=success_threshold)
        nonzero_count = int(round(batch_size * float(nonzero_fraction)))
        requested = {
            "zero": batch_size - nonzero_count,
            "successful_nonzero": int(round(nonzero_count * float(nonzero_success_fraction))),
            "failed_nonzero": nonzero_count
            - int(round(nonzero_count * float(nonzero_success_fraction))),
        }
        return all(amount == 0 or counts[name] > 0 for name, amount in requested.items())

    def sample_stratified(
        self,
        batch_size: int,
        *,
        nonzero_fraction: float,
        nonzero_success_fraction: float,
        success_threshold: float,
    ) -> FrozenLatentBatch:
        if batch_size <= 0 or len(self) == 0:
            raise ValueError("Stratified replay requires data and positive batch")
        if not 0.0 <= nonzero_fraction <= 1.0:
            raise ValueError("nonzero_fraction must lie in [0, 1]")
        if not 0.0 <= nonzero_success_fraction <= 1.0:
            raise ValueError("nonzero_success_fraction must lie in [0, 1]")
        if not np.isfinite(success_threshold):
            raise ValueError("success_threshold must be finite")
        indices = self._indices()
        nonzero = np.any(
            np.abs(self.action_chunks[indices] - self.ref_chunks[indices]) > 1e-6,
            axis=(1, 2),
        )
        successful = np.isfinite(self.mc_returns[indices]) & (
            self.mc_returns[indices] > float(success_threshold)
        )
        pools = {
            "zero": indices[~nonzero],
            "successful nonzero": indices[nonzero & successful],
            "failed nonzero": indices[nonzero & ~successful],
        }
        nonzero_count = int(round(batch_size * float(nonzero_fraction)))
        counts = (
            batch_size - nonzero_count,
            int(round(nonzero_count * float(nonzero_success_fraction))),
        )
        counts += (nonzero_count - counts[1],)
        selected: list[np.ndarray] = []
        for name, count in zip(pools, counts, strict=True):
            if count == 0:
                selected.append(np.empty(0, dtype=np.int64))
            elif pools[name].size == 0:
                raise ValueError(f"Stratified replay requested an empty {name} pool")
            else:
                selected.append(
                    self._rng.choice(
                        pools[name], size=count, replace=pools[name].size < count
                    )
                )
        result = np.concatenate(selected)
        self._rng.shuffle(result)
        return self.batch(result)

    def batch(self, indices: np.ndarray) -> FrozenLatentBatch:
        idx = np.asarray(indices, dtype=np.int64)
        return FrozenLatentBatch(
            states=self.states[idx],
            latents=self.latents[idx].astype(np.float32),
            ref_chunks=self.ref_chunks[idx],
            action_chunks=self.action_chunks[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            next_states=self.next_states[idx],
            next_latents=self.next_latents[idx].astype(np.float32),
            next_ref_chunks=self.next_ref_chunks[idx],
            durations=self.durations[idx],
            step_ids=self.step_ids[idx],
            mc_returns=self.mc_returns[idx],
        )

    def save(self, path: str | Path, *, snapshot_id: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = self._indices()
        schema = 3
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
            payload: dict[str, Any] = {
                "schema_version": np.int32(schema),
                "capacity": np.int32(self.capacity),
                "pos": np.int32(self.pos),
                "full": np.bool_(self.full),
                "rng_state": np.asarray(json.dumps(self._rng.bit_generator.state)),
                "snapshot_id": np.asarray(snapshot_id or ""),
                "state_dim": np.int32(self.state_dim),
                "latent_dim": np.int32(self.latent_dim),
                "chunk_len": np.int32(self.chunk_len),
                "action_dim": np.int32(self.action_dim),
                "states": self.states[idx],
                "latents": self.latents[idx],
                "ref_chunks": self.ref_chunks[idx],
                "action_chunks": self.action_chunks[idx],
                "rewards": self.rewards[idx],
                "dones": self.dones[idx],
                "next_states": self.next_states[idx],
                "next_latents": self.next_latents[idx],
                "next_ref_chunks": self.next_ref_chunks[idx],
                "durations": self.durations[idx],
                "step_ids": self.step_ids[idx],
                "mc_returns": self.mc_returns[idx],
            }
            np.savez_compressed(file, **payload)
        temporary.replace(path)

    def load(self, path: str | Path) -> int:
        """Restore an exact replay snapshot for interrupted-run continuation."""

        self.last_load_was_exact = False
        self.last_loaded_snapshot_id = None
        self.last_migration_stats = None
        with np.load(path, allow_pickle=False) as data:
            source_schema = int(np.asarray(data["schema_version"]))
            if source_schema != 3:
                raise ValueError("Replay is not a mean Advantage-BC schema")
            source_size = int(np.asarray(data["states"]).shape[0])
            source_capacity = int(np.asarray(data["capacity"]))
            source_pos = int(np.asarray(data["pos"]))
            source_full = bool(np.asarray(data["full"]))
            source_latent_bins = 1
            if int(np.asarray(data["state_dim"])) != self.state_dim:
                raise ValueError("Replay state_dim does not match model")
            if int(np.asarray(data["latent_dim"])) != self.latent_dim:
                raise ValueError("Replay latent_dim does not match model")
            if int(np.asarray(data["chunk_len"])) != self.chunk_len:
                raise ValueError("Replay chunk_len does not match model")
            if int(np.asarray(data["action_dim"])) != self.action_dim:
                raise ValueError("Replay action_dim does not match model")
            if (
                source_size > source_capacity
                or source_pos < 0
                or source_pos >= source_capacity
            ):
                raise ValueError("Replay capacity/position metadata is invalid")
            expected = source_capacity if source_full else source_pos
            if source_size != expected:
                raise ValueError("Replay state and transition count disagree")
            names = (
                "states", "latents", "ref_chunks", "action_chunks", "rewards",
                "dones", "next_states", "next_latents", "next_ref_chunks",
                "durations", "step_ids", "mc_returns",
            )
            arrays = {name: np.asarray(data[name]) for name in names}
            expected_shapes = {
                "states": (source_size, self.state_dim),
                "latents": (
                    (source_size, self.latent_dim)
                    if source_latent_bins == 1
                    else (source_size, source_latent_bins, self.latent_dim)
                ),
                "ref_chunks": (source_size, self.chunk_len, self.action_dim),
                "action_chunks": (source_size, self.chunk_len, self.action_dim),
                "rewards": (source_size, self.chunk_len),
                "dones": (source_size,),
                "next_states": (source_size, self.state_dim),
                "next_latents": (
                    (source_size, self.latent_dim)
                    if source_latent_bins == 1
                    else (source_size, source_latent_bins, self.latent_dim)
                ),
                "next_ref_chunks": (
                    source_size,
                    self.chunk_len,
                    self.action_dim,
                ),
                "durations": (source_size,),
                "step_ids": (source_size,),
                "mc_returns": (source_size,),
            }
            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    raise ValueError(
                        f"Replay {name} shape {arrays[name].shape} != {shape}"
                    )
            expected_dtypes = {
                "states": np.dtype(np.float32),
                "latents": np.dtype(np.float16),
                "ref_chunks": np.dtype(np.float32),
                "action_chunks": np.dtype(np.float32),
                "rewards": np.dtype(np.float32),
                "dones": np.dtype(np.float32),
                "next_states": np.dtype(np.float32),
                "next_latents": np.dtype(np.float16),
                "next_ref_chunks": np.dtype(np.float32),
                "durations": np.dtype(np.int32),
                "step_ids": np.dtype(np.int32),
                "mc_returns": np.dtype(np.float32),
            }
            for name, dtype in expected_dtypes.items():
                if arrays[name].dtype != dtype:
                    raise TypeError(
                        f"Replay {name} dtype {arrays[name].dtype} != {dtype}"
                    )
            if not all(np.all(np.isfinite(arrays[name])) for name in names if name != "mc_returns"):
                raise ValueError("Replay contains NaN or Inf")
            if not np.all(np.isfinite(arrays["mc_returns"]) | np.isnan(arrays["mc_returns"])):
                raise ValueError("Replay mc_returns contain invalid values")
            expected_latent_shape = (
                (source_size, self.latent_dim)
                if source_latent_bins == 1
                else (source_size, source_latent_bins, self.latent_dim)
            )
            if (
                arrays["latents"].shape != expected_latent_shape
                or arrays["next_latents"].shape != expected_latent_shape
            ):
                raise ValueError("Replay latent arrays have incompatible shapes")
            if np.any(arrays["durations"] < 1) or np.any(
                arrays["durations"] > self.chunk_len
            ):
                raise ValueError("Replay contains an invalid duration")
            if np.any(arrays["step_ids"] < 0):
                raise ValueError("Replay contains a negative step_id")
            if np.any((arrays["dones"] != 0.0) & (arrays["dones"] != 1.0)):
                raise ValueError("Replay dones must be binary")
            suffix = np.arange(self.chunk_len)[None, :] >= arrays[
                "durations"
            ][:, None]
            if not np.array_equal(
                arrays["action_chunks"][suffix],
                arrays["ref_chunks"][suffix],
            ):
                raise ValueError("Replay padded action suffix differs from reference")
            loaded_snapshot_id = (
                str(np.asarray(data["snapshot_id"]).item())
                if "snapshot_id" in data.files
                else ""
            )
            self.last_loaded_snapshot_id = loaded_snapshot_id or None
            exact_layout = bool(
                source_schema == 3
                and source_capacity == self.capacity
                and "rng_state" in data.files
            )
            if exact_layout:
                for name, target in (
                    ("states", self.states), ("latents", self.latents),
                    ("ref_chunks", self.ref_chunks), ("action_chunks", self.action_chunks),
                    ("rewards", self.rewards), ("dones", self.dones),
                    ("next_states", self.next_states), ("next_latents", self.next_latents),
                    ("next_ref_chunks", self.next_ref_chunks), ("durations", self.durations),
                    ("step_ids", self.step_ids), ("mc_returns", self.mc_returns),
                ):
                    target[:source_size] = arrays[name]
                self.pos = source_pos
                self.full = source_full
                self._rng.bit_generator.state = json.loads(
                    str(np.asarray(data["rng_state"]).item())
                )
                self.last_load_was_exact = True
                return source_size
            raise ValueError(
                "Resume replay must match schema, capacity, and RNG layout exactly"
            )


class FrozenLatentContextEncoder(nn.Module):
    def __init__(self, config: FrozenLatentRLConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        self.config = config
        self.latent = (
            SafeEndpointTokenEncoder(h)
            if config.latent_encoder == SAFE_ENDPOINT_LATENT_ENCODER
            else nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.Linear(config.latent_dim, h),
                nn.GELU(),
            )
        )
        self.state = nn.Sequential(
            nn.Linear(config.state_dim + 1, h), nn.LayerNorm(h), nn.GELU()
        )
        self.reference = nn.Sequential(
            nn.Linear(config.chunk_len * config.action_dim, h), nn.LayerNorm(h), nn.GELU()
        )
        self.fusion = _mlp(3 * h, config.context_dim, h, 1)

    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        ref_chunk: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        remaining = 1.0 - step_ids.to(state.dtype).reshape(-1, 1) / float(
            self.config.max_episode_steps
        )
        state_time = torch.cat([state, remaining.clamp(0.0, 1.0)], dim=-1)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("Frozen latent must have shape [B, latent_dim]")
        latent_feature = self.latent(latent)
        return self.fusion(
            torch.cat(
                [
                    self.state(state_time),
                    latent_feature,
                    self.reference(ref_chunk.flatten(start_dim=1)),
                ],
                dim=-1,
            )
        )


class ContinuousResidualActor(nn.Module):
    def __init__(self, config: FrozenLatentRLConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(
            config.context_dim,
            config.exploration_knots * config.action_dim,
            config.hidden_dim,
            config.num_layers,
        )
        output = self.trunk[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _expand_knots(self, knots: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            knots.permute(0, 2, 1), size=self.config.chunk_len,
            mode="linear", align_corners=True,
        ).permute(0, 2, 1)

    def smooth_noise(
        self,
        batch_size: int,
        *,
        std: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        knots = torch.randn(
            batch_size,
            self.config.exploration_knots,
            self.config.action_dim,
            dtype=dtype,
            device=device,
        )
        return self._expand_knots(knots) * float(std)

    def mean(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.trunk(context)
        knots = self.config.actor_residual_limit * torch.tanh(
            raw.reshape(-1, self.config.exploration_knots, self.config.action_dim)
        )
        return self._expand_knots(knots)

    def sample(
        self,
        context: torch.Tensor,
        *,
        deterministic: bool,
        noise_std: float | None = None,
    ) -> torch.Tensor:
        mean = self.mean(context)
        std = self.config.fixed_std if noise_std is None else float(noise_std)
        if not 0.0 <= std <= 1.0:
            raise ValueError("noise_std must lie in [0, 1]")
        if deterministic or std == 0.0:
            return mean
        noise = self.smooth_noise(
            mean.shape[0], std=std, dtype=mean.dtype, device=mean.device
        )
        return (mean + noise).clamp(-1.0, 1.0)


class _QHead(nn.Module):
    def __init__(self, config: FrozenLatentRLConfig) -> None:
        super().__init__()
        action_size = config.chunk_len * config.action_dim
        self.value = _mlp(config.context_dim, 1, config.hidden_dim, config.num_layers)
        self.advantage = _mlp(
            config.context_dim + action_size,
            1,
            config.hidden_dim,
            config.num_layers,
        )

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        flat = residual.flatten(start_dim=1)
        raw = self.advantage(torch.cat([context, flat], dim=-1)).squeeze(-1)
        zero = self.advantage(torch.cat([context, torch.zeros_like(flat)], dim=-1)).squeeze(-1)
        return self.value(context).squeeze(-1) + raw - zero


class ContinuousResidualCritic(nn.Module):
    def __init__(self, config: FrozenLatentRLConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList([_QHead(config) for _ in range(config.num_critics)])

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(context, residual) for head in self.heads], dim=-1)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


class FrozenLatentResidualAgent:
    """TD3 agent with a frozen Pi0 parent and one continuous residual actor."""

    checkpoint_version = 4

    def __init__(
        self,
        config: FrozenLatentRLConfig,
        *,
        device: str | torch.device = "cpu",
        runtime_identity: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.runtime_identity = copy.deepcopy(runtime_identity)
        self.context = FrozenLatentContextEncoder(config).to(self.device)
        self.target_context = FrozenLatentContextEncoder(config).to(self.device)
        self.actor = ContinuousResidualActor(config).to(self.device)
        self.target_actor = ContinuousResidualActor(config).to(self.device)
        self.critic = ContinuousResidualCritic(config).to(self.device)
        self.target_critic = ContinuousResidualCritic(config).to(self.device)
        self.target_context.load_state_dict(self.context.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(
            [*self.context.parameters(), *self.critic.parameters()], lr=config.critic_lr
        )
        low = np.asarray(config.action_low, dtype=np.float32)
        high = np.asarray(config.action_high, dtype=np.float32)
        fractions = np.full((config.action_dim,), config.arm_residual_fraction, dtype=np.float32)
        fractions[-1] = config.gripper_residual_fraction
        self.action_low = low
        self.action_high = high
        self.residual_scale = (high - low) * fractions
        self._action_low_tensor = self._tensor(low).view(1, 1, -1)
        self._action_high_tensor = self._tensor(high).view(1, 1, -1)
        self._residual_scale_tensor = self._tensor(self.residual_scale).view(1, 1, -1)
        self.total_updates = 0
        self.actor_updates = 0
        self.critic_updates = 0
        self.snapshot_id: str | None = None

    def assert_runtime_identity(self, actual: dict[str, Any]) -> None:
        if self.runtime_identity is None:
            raise ValueError("Checkpoint has no frozen runtime identity")
        if actual != self.runtime_identity:
            keys = sorted(set(actual) | set(self.runtime_identity))
            mismatches = {
                key: (self.runtime_identity.get(key), actual.get(key))
                for key in keys
                if self.runtime_identity.get(key) != actual.get(key)
            }
            raise ValueError(f"RL checkpoint runtime identity mismatch: {mismatches}")

    def _tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def normalized_residual(
        self,
        ref_chunk: np.ndarray,
        action_chunk: np.ndarray,
        *,
        duration: int | None = None,
    ) -> np.ndarray:
        ref = np.asarray(ref_chunk, dtype=np.float32)
        action = np.asarray(action_chunk, dtype=np.float32)
        expected = (self.config.chunk_len, self.config.action_dim)
        if ref.shape != expected or action.shape != expected:
            raise ValueError(f"Action chunks must have shape {expected}")
        residual = (action - ref) / self.residual_scale[None]
        if duration is not None:
            residual[int(duration):] = 0.0
        return np.clip(residual, -1.0, 1.0).astype(np.float32)

    def apply_residual(self, ref_chunk: np.ndarray, residual: np.ndarray) -> np.ndarray:
        ref = np.asarray(ref_chunk, dtype=np.float32)
        normalized = np.asarray(residual, dtype=np.float32)
        expected = (self.config.chunk_len, self.config.action_dim)
        if ref.shape != expected or normalized.shape != expected:
            raise ValueError(f"Reference and residual must have shape {expected}")
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(normalized)):
            raise ValueError("Reference or residual contains NaN or Inf")
        action = ref + np.clip(normalized, -1.0, 1.0) * self.residual_scale[None]
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def _effective_residual(self, ref_chunk: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(
            ref_chunk + requested.clamp(-1.0, 1.0) * self._residual_scale_tensor,
            self._action_low_tensor,
            self._action_high_tensor,
        )
        return (action - ref_chunk) / self._residual_scale_tensor

    def _encode(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        ref: torch.Tensor,
        steps: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        return (self.target_context if target else self.context)(state, latent, ref, steps)

    def _batch_residual(
        self, batch: FrozenLatentBatch
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = self._tensor(batch.states)
        latent = self._tensor(batch.latents)
        ref = self._tensor(batch.ref_chunks)
        action = self._tensor(batch.action_chunks)
        rewards = self._tensor(batch.rewards)
        done = self._tensor(batch.dones)
        next_state = self._tensor(batch.next_states)
        next_latent = self._tensor(batch.next_latents)
        next_ref = self._tensor(batch.next_ref_chunks)
        duration = self._tensor(batch.durations, torch.long)
        steps = self._tensor(batch.step_ids, torch.long)
        mc_return = self._tensor(batch.mc_returns)
        residual = torch.clamp((action - ref) / self._residual_scale_tensor, -1.0, 1.0)
        time = torch.arange(self.config.chunk_len, device=self.device).view(1, -1)
        residual = torch.where((time < duration.view(-1, 1)).unsqueeze(-1), residual, 0.0)
        return (
            state,
            latent,
            ref,
            residual,
            rewards,
            done,
            next_state,
            next_latent,
            next_ref,
            duration,
            steps,
            mc_return,
        )

    def update(
        self,
        batch: FrozenLatentBatch,
        *,
        update_actor: bool = True,
        freeze_policy_context: bool = False,
    ) -> dict[str, float]:
        (
            state,
            latent,
            ref,
            residual,
            rewards,
            done,
            next_state,
            next_latent,
            next_ref,
            duration,
            steps,
            mc_return,
        ) = self._batch_residual(batch)
        with torch.no_grad():
            next_steps = (steps + duration).clamp(max=self.config.max_episode_steps)
            next_context = self._encode(next_state, next_latent, next_ref, next_steps, target=True)
            next_residual = self.target_actor.mean(next_context)
            if self.config.target_noise:
                noise = self.target_actor.smooth_noise(
                    next_residual.shape[0], std=self.config.target_noise,
                    dtype=next_residual.dtype, device=next_residual.device,
                ).clamp(-self.config.target_noise_clip, self.config.target_noise_clip)
                next_residual = (next_residual + noise).clamp(-1.0, 1.0)
            next_residual = self._effective_residual(next_ref, next_residual)
            next_q = self.target_critic(next_context, next_residual).min(dim=-1).values
            reward_steps = torch.arange(self.config.chunk_len, device=self.device)
            mask = reward_steps.view(1, -1) < duration.view(-1, 1)
            macro_reward = (rewards * mask).sum(dim=-1)
            discount = torch.pow(
                torch.as_tensor(self.config.gamma, device=self.device),
                duration.to(rewards.dtype) / float(self.config.chunk_len),
            )
            target_q = macro_reward + (1.0 - done) * discount * next_q

        base_context = self._encode(state, latent, ref, steps, target=False)
        critic_context = base_context.detach() if freeze_policy_context else base_context
        q = self.critic(critic_context, residual)
        td_loss = F.smooth_l1_loss(q, target_q[:, None].expand_as(q))
        finite_mc = torch.isfinite(mc_return)
        mc_loss = q.new_zeros(())
        if bool(finite_mc.any()):
            mc_loss = F.smooth_l1_loss(
                q[finite_mc], mc_return[finite_mc, None].expand_as(q[finite_mc])
            )
        conservative_q_loss = q.new_zeros(())
        if self.config.conservative_q_weight > 0.0:
            with torch.no_grad():
                actor_proposal = self.actor.mean(base_context.detach())
                random_proposal = self.actor.smooth_noise(
                    state.shape[0], std=self.config.conservative_random_std,
                    dtype=state.dtype, device=self.device,
                ).clamp(-1.0, 1.0)
            base_q = self.critic(critic_context, torch.zeros_like(residual))
            positive_advantages = []
            for proposal in (actor_proposal, random_proposal, -random_proposal):
                proposal_q = self.critic(critic_context, self._effective_residual(ref, proposal))
                positive_advantages.append((proposal_q - base_q).clamp_min(0.0))
            conservative_q_loss = torch.stack(positive_advantages).square().mean()
        critic_loss = (
            td_loss
            + self.config.mc_loss_weight * mc_loss
            + self.config.conservative_q_weight * conservative_q_loss
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = [*self.context.parameters(), *self.critic.parameters()]
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(critic_parameters, self.config.grad_clip_norm)
        self.critic_opt.step()
        if not freeze_policy_context:
            _soft_update(self.target_context, self.context, self.config.target_tau)
        _soft_update(self.target_critic, self.critic, self.config.target_tau)
        self.total_updates += 1
        self.critic_updates += 1
        metrics: dict[str, float] = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "mc_loss": float(mc_loss.detach().cpu()),
            "conservative_q_loss": float(conservative_q_loss.detach().cpu()),
            "q_mean": float(q.mean().detach().cpu()),
            "target_q": float(target_q.mean().detach().cpu()),
            "actor_updated": 0.0,
            "actor_success_bc_loss": 0.0,
            "actor_success_bc_samples": 0.0,
            "policy_context_frozen": float(freeze_policy_context),
        }
        if update_actor and self.total_updates % self.config.actor_update_period == 0:
            with torch.no_grad():
                actor_context = self._encode(state, latent, ref, steps, target=False)
                critic_actor_context = actor_context.detach()
            predicted = self.actor.mean(actor_context)
            effective = self._effective_residual(ref, predicted)
            self.critic.requires_grad_(False)
            actor_q_values = self.critic(critic_actor_context, effective).min(dim=-1).values
            actor_q = actor_q_values.mean()
            l2 = predicted.square().mean()
            smoothness = (
                (predicted[:, 1:] - predicted[:, :-1]).square().mean()
                if self.config.chunk_len > 1
                else predicted.new_zeros(())
            )
            actor_loss = (
                -actor_q
                + self.config.actor_l2_weight * l2
                + self.config.actor_smoothness_weight * smoothness
            )
            success_bc = actor_q.new_zeros(())
            success_bc_samples = 0
            success_bc_advantage = actor_q.new_zeros(())
            if self.config.actor_success_bc_weight > 0.0:
                time = torch.arange(self.config.chunk_len, device=self.device).view(1, -1)
                valid = (time < duration.view(-1, 1)).unsqueeze(-1)
                valid_count = (
                    valid.to(predicted.dtype).sum(dim=(1, 2)).clamp_min(1.0)
                    * float(self.config.action_dim)
                )
                per_sample_bc = (
                    (effective - residual).square() * valid.to(predicted.dtype)
                ).sum(dim=(1, 2)) / valid_count
                executed_rms = torch.sqrt(
                    (residual.square() * valid.to(residual.dtype)).sum(dim=(1, 2))
                    / valid_count
                )
                success_anchor = (
                    finite_mc
                    & (mc_return > self.config.success_bc_min_return)
                    & (
                        executed_rms
                        > self.config.actor_success_bc_min_residual_rms
                    )
                )
                if self.config.actor_success_bc_min_q_advantage > 0.0:
                    with torch.no_grad():
                        executed_q = self.critic(
                            critic_actor_context, residual
                        ).min(dim=-1).values
                        zero_q = self.critic(
                            critic_actor_context, torch.zeros_like(residual)
                        ).min(dim=-1).values
                        executed_advantage = executed_q - zero_q
                    success_anchor &= (
                        executed_advantage
                        >= self.config.actor_success_bc_min_q_advantage
                    )
                    if bool(success_anchor.any()):
                        success_bc_advantage = executed_advantage[
                            success_anchor
                        ].mean()
                success_bc = (
                    per_sample_bc[success_anchor].mean()
                    if bool(success_anchor.any())
                    else actor_q.new_zeros(())
                )
                success_bc_samples = int(success_anchor.sum().detach().cpu())
                actor_loss = (
                    actor_loss + self.config.actor_success_bc_weight * success_bc
                )
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()
            self.critic.requires_grad_(True)
            _soft_update(self.target_actor, self.actor, self.config.target_tau)
            self.actor_updates += 1
            metrics.update(
                actor_updated=1.0,
                actor_loss=float(actor_loss.detach().cpu()),
                actor_q=float(actor_q.detach().cpu()),
                actor_l2=float(l2.detach().cpu()),
                actor_smoothness=float(smoothness.detach().cpu()),
                actor_success_bc_loss=float(success_bc.detach().cpu()),
                actor_success_bc_samples=float(success_bc_samples),
                actor_success_bc_advantage=float(
                    success_bc_advantage.detach().cpu()
                ),
            )
        metrics.update(
            total_updates=float(self.total_updates),
            critic_updates=float(self.critic_updates),
            actor_updates=float(self.actor_updates),
        )
        return metrics

    def actor_success_bc_metrics(
        self, batch: FrozenLatentBatch, *, update: bool = False
    ) -> dict[str, float]:
        """Evaluate or fit on successful, nonzero executed residuals."""
        state = self._tensor(batch.states)
        latent = self._tensor(batch.latents)
        ref = self._tensor(batch.ref_chunks)
        action = self._tensor(batch.action_chunks)
        duration = self._tensor(batch.durations, torch.long)
        steps = self._tensor(batch.step_ids, torch.long)
        mc_return = self._tensor(batch.mc_returns)
        target = self._effective_residual(
            ref,
            torch.clamp(
                (action - ref) / self._residual_scale_tensor, -1.0, 1.0
            ),
        )
        time = torch.arange(self.config.chunk_len, device=self.device).view(1, -1)
        valid = (time < duration[:, None]).unsqueeze(-1)
        weight = valid.to(target.dtype)
        count = weight.sum(dim=(1, 2)).clamp_min(1.0) * float(self.config.action_dim)
        target_rms = torch.sqrt((target.square() * weight).sum(dim=(1, 2)) / count)
        anchors = (
            torch.isfinite(mc_return)
            & (mc_return > self.config.success_bc_min_return)
            & (target_rms > self.config.actor_success_bc_min_residual_rms)
        )
        if not bool(anchors.any()):
            raise ValueError("No successful nonzero residual in actor BC batch")
        context = self._encode(state, latent, ref, steps, target=False)
        prediction = self._effective_residual(ref, self.actor.mean(context))
        squared_error = ((prediction - target).square() * weight).sum(dim=(1, 2)) / count
        loss = squared_error[anchors].mean()
        if update:
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()
            self.actor_updates += 1
        with torch.no_grad():
            predicted_flat = (prediction * weight).flatten(start_dim=1)
            target_flat = (target * weight).flatten(start_dim=1)
            cosine = F.cosine_similarity(predicted_flat, target_flat, dim=-1, eps=1e-8)
            predicted_rms = torch.sqrt((prediction.square() * weight).sum(dim=(1, 2)) / count)
        return {
            "actor_bc_loss": float(loss.detach().cpu()),
            "actor_bc_cosine": float(cosine[anchors].mean().cpu()),
            "actor_bc_predicted_rms": float(predicted_rms[anchors].mean().cpu()),
            "actor_bc_target_rms": float(target_rms[anchors].mean().cpu()),
            "actor_bc_samples": float(anchors.sum().cpu()),
            "actor_updates": float(self.actor_updates),
        }

    def sync_target_actor(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())

    @torch.no_grad()
    def select_residual(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
        deterministic: bool,
        noise_std: float | None = None,
    ) -> np.ndarray:
        state_t = self._tensor(np.asarray(state, dtype=np.float32)[None])
        latent_t = self._tensor(np.asarray(latent, dtype=np.float32)[None])
        ref_t = self._tensor(np.asarray(ref_chunk, dtype=np.float32)[None])
        steps_t = self._tensor(np.asarray([step_id]), torch.long)
        context = self._encode(state_t, latent_t, ref_t, steps_t, target=False)
        return (
            self.actor.sample(
                context, deterministic=deterministic, noise_std=noise_std
            )[0]
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def select_chunk(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
        deterministic: bool,
        noise_std: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = self.select_residual(
            state, latent, ref_chunk, step_id=step_id,
            deterministic=deterministic, noise_std=noise_std,
        )
        return self.apply_residual(ref_chunk, residual), residual

    @torch.no_grad()
    def conservative_advantage(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        residual: np.ndarray,
        *,
        step_id: int,
    ) -> float:
        state_t = self._tensor(np.asarray(state, dtype=np.float32)[None])
        latent_t = self._tensor(np.asarray(latent, dtype=np.float32)[None])
        ref_t = self._tensor(np.asarray(ref_chunk, dtype=np.float32)[None])
        residual_t = self._tensor(np.asarray(residual, dtype=np.float32)[None])
        steps_t = self._tensor(np.asarray([step_id]), torch.long)
        context = self._encode(state_t, latent_t, ref_t, steps_t, target=False)
        effective = self._effective_residual(ref_t, residual_t)
        candidate_q = self.critic(context, effective)
        base_q = self.critic(context, torch.zeros_like(effective))
        return float((candidate_q - base_q).min(dim=-1).values[0].cpu())

    @torch.no_grad()
    def executed_action_advantages(self, batch: FrozenLatentBatch) -> np.ndarray:
        state = self._tensor(batch.states)
        latent = self._tensor(batch.latents)
        ref = self._tensor(batch.ref_chunks)
        action = self._tensor(batch.action_chunks)
        duration = self._tensor(batch.durations, torch.long)
        steps = self._tensor(batch.step_ids, torch.long)
        residual = torch.clamp((action - ref) / self._residual_scale_tensor, -1.0, 1.0)
        time = torch.arange(self.config.chunk_len, device=self.device).view(1, -1)
        residual = torch.where((time < duration[:, None]).unsqueeze(-1), residual, 0.0)
        context = self._encode(state, latent, ref, steps, target=False)
        advantage = self.critic(context, residual) - self.critic(
            context, torch.zeros_like(residual)
        )
        return advantage.min(dim=-1).values.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path, *, snapshot_id: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = (
            SAFE_VALUE_GUIDED_CHECKPOINT_SCHEMA
            if self.config.latent_protocol == SAFE_LATENT_PROTOCOL
            else FROZEN_LATENT_CHECKPOINT_SCHEMA
        )
        payload = {
            "schema": schema,
            "checkpoint_version": self.checkpoint_version,
            "config": asdict(self.config),
            "runtime_identity": copy.deepcopy(self.runtime_identity),
            "context": self.context.state_dict(),
            "target_context": self.target_context.state_dict(),
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
            "snapshot_id": snapshot_id,
        }
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
        temporary.replace(path)
        self.snapshot_id = snapshot_id or None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> FrozenLatentResidualAgent:
        payload = torch.load(path, map_location=device, weights_only=False)
        schema = payload.get("schema")
        version = int(payload.get("checkpoint_version", -1))
        accepted = {
            (FROZEN_LATENT_CHECKPOINT_SCHEMA, 4),
            (LEGACY_V4_FROZEN_LATENT_CHECKPOINT_SCHEMA, 4),
            (SAFE_VALUE_GUIDED_CHECKPOINT_SCHEMA, 4),
        }
        if (schema, version) not in accepted:
            raise ValueError("Checkpoint is not a maintained frozen-Pi0 residual agent")
        raw_config = dict(payload["config"])
        if "outcome_success_threshold" in raw_config:
            raw_config["success_bc_min_return"] = raw_config.pop(
                "outcome_success_threshold"
            )
        if int(raw_config.get("temporal_latent_bins", 1)) != 1:
            raise ValueError("Temporal checkpoints are not supported on this branch")
        if float(raw_config.get("q_uncertainty_beta", 0.0)) != 0.0:
            raise ValueError("Q-uncertainty checkpoints are not supported on this branch")
        config_fields = {field.name for field in fields(FrozenLatentRLConfig)}
        ignored = set(raw_config) - config_fields
        unexpected = ignored - {
            "temporal_latent_bins",
            "temporal_adapter_dim",
            "q_uncertainty_beta",
        }
        if unexpected:
            raise ValueError(f"Checkpoint has unsupported fields: {sorted(unexpected)}")
        raw_config = {key: value for key, value in raw_config.items() if key in config_fields}
        agent = cls(
            FrozenLatentRLConfig(**raw_config),
            device=device,
            runtime_identity=payload.get("runtime_identity"),
        )
        for name in (
            "context",
            "target_context",
            "actor",
            "target_actor",
            "critic",
            "target_critic",
        ):
            state = payload.get(name)
            if state is None:
                raise ValueError(f"Checkpoint is missing {name}")
            getattr(agent, name).load_state_dict(state)
        if "actor_opt" in payload:
            agent.actor_opt.load_state_dict(payload["actor_opt"])
        if "critic_opt" in payload:
            agent.critic_opt.load_state_dict(payload["critic_opt"])
        agent.total_updates = int(payload.get("total_updates", 0))
        agent.critic_updates = int(payload.get("critic_updates", 0))
        agent.actor_updates = int(payload.get("actor_updates", 0))
        snapshot_id = payload.get("snapshot_id")
        if snapshot_id is not None and not isinstance(snapshot_id, str):
            raise ValueError("Checkpoint snapshot_id must be a string or null")
        agent.snapshot_id = snapshot_id or None
        return agent
