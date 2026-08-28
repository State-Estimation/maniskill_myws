"""V-gated residual policy with a supervised chunk-value improvement scorer."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.latent_actor import (
    SAFE_ENDPOINT_LATENT_ENCODER,
    ContinuousResidualActor,
    FrozenLatentContextEncoder,
)


VALUE_GUIDED_BANDIT_CHECKPOINT_SCHEMA = "safe_value_guided_chunk_bandit_v1"
VALUE_GUIDED_BANDIT_REPLAY_SCHEMA = 4


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(num_layers)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ValueGuidedBanditConfig:
    state_dim: int
    action_dim: int = 8
    chunk_len: int = 10
    max_episode_steps: int = 500
    latent_dim: int = SAFE_LATENT_DIM
    latent_protocol: str = SAFE_LATENT_PROTOCOL
    latent_encoder: str = SAFE_ENDPOINT_LATENT_ENCODER
    context_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    num_scorers: int = 5
    candidate_count: int = 12
    exploration_knots: int = 6
    arm_residual_fraction: float = 0.06
    gripper_residual_fraction: float = 0.10
    fixed_std: float = 0.10
    actor_residual_limit: float = 0.35
    scorer_lr: float = 1e-4
    actor_lr: float = 1e-4
    actor_update_period: int = 2
    actor_context_trainable: bool = False
    actor_hypotheses: int = 1
    actor_hypothesis_loss_weight: float = 1.0
    actor_value_objective_weight: float = 1.0
    actor_l2_weight: float = 0.5
    actor_smoothness_weight: float = 0.2
    actor_awr_weight: float = 2.0
    actor_awr_temperature: float = 0.10
    actor_awr_min_advantage: float = 0.01
    actor_success_bc_weight: float = 2.0
    actor_success_bc_min_value_improvement: float = -0.02
    actor_success_bc_min_residual_rms: float = 1e-4
    actor_success_bc_requires_gate: bool = False
    actor_success_credit_mode: str = "chunk_threshold"
    actor_deployment_policy: str = "actor_mean"
    retrieval_max_step_distance: int = 50
    retrieval_min_similarity: float = 0.0
    bootstrap_probability: float = 0.8
    grad_clip_norm: float = 1.0
    selection_uncertainty_penalty: float = 1.0
    selection_residual_penalty: float = 0.01
    selection_min_advantage: float = 0.01
    deterministic_actor_on_gate: bool = False
    # Persistent actor intervention is safest when authorized by a high-confidence
    # immediate entry.  False preserves the legacy confirmed-entry behavior.
    actor_requires_immediate_gate: bool = False
    actor_safety_enabled: bool = False
    actor_safety_min_value_improvement: float = -0.02
    actor_safety_cooldown_chunks: int = 3
    actor_max_consecutive_chunks: int = 0
    actor_throttle_cooldown_chunks: int = 3
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.action_dim != 8:
            raise ValueError("Value-guided bandit requires action_dim=8")
        if self.chunk_len != 10:
            raise ValueError("Value-guided bandit requires 10-step chunks")
        if (
            min(
                self.state_dim,
                self.max_episode_steps,
                self.context_dim,
                self.hidden_dim,
                self.num_layers,
                self.num_scorers,
                self.candidate_count,
                self.exploration_knots,
                self.actor_hypotheses,
            )
            <= 0
        ):
            raise ValueError("Model dimensions and counts must be positive")
        if self.num_scorers < 2:
            raise ValueError("At least two scorer heads are required")
        if self.candidate_count < 3:
            raise ValueError("Candidates must include base, mean, and exploration")
        if self.exploration_knots > self.chunk_len:
            raise ValueError("exploration_knots cannot exceed chunk_len")
        if (
            self.latent_protocol != SAFE_LATENT_PROTOCOL
            or self.latent_dim != SAFE_LATENT_DIM
            or self.latent_encoder != SAFE_ENDPOINT_LATENT_ENCODER
        ):
            raise ValueError("Value-guided bandit requires the SAFE endpoint latent")
        if not 0.0 < self.arm_residual_fraction <= 0.5:
            raise ValueError("arm_residual_fraction must lie in (0, 0.5]")
        if not 0.0 < self.gripper_residual_fraction <= 0.5:
            raise ValueError("gripper_residual_fraction must lie in (0, 0.5]")
        if not 0.0 <= self.fixed_std <= 1.0:
            raise ValueError("fixed_std must lie in [0, 1]")
        if not 0.0 < self.actor_residual_limit <= 1.0:
            raise ValueError("actor_residual_limit must lie in (0, 1]")
        if min(self.scorer_lr, self.actor_lr, self.actor_awr_temperature) <= 0.0:
            raise ValueError("Learning rates and AWR temperature must be positive")
        if self.actor_update_period <= 0:
            raise ValueError("actor_update_period must be positive")
        if not 0.0 < self.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must lie in (0, 1]")
        if (
            min(
                self.actor_l2_weight,
                self.actor_smoothness_weight,
                self.actor_hypothesis_loss_weight,
                self.actor_value_objective_weight,
                self.actor_awr_weight,
                self.actor_awr_min_advantage,
                self.actor_success_bc_weight,
                self.actor_success_bc_min_residual_rms,
                self.grad_clip_norm,
                self.selection_uncertainty_penalty,
                self.selection_residual_penalty,
                self.selection_min_advantage,
            )
            < 0.0
        ):
            raise ValueError("Loss, selection, and clipping values must be non-negative")
        if not np.isfinite(self.actor_success_bc_min_value_improvement):
            raise ValueError("Success-BC value-improvement threshold must be finite")
        if not np.isfinite(self.actor_safety_min_value_improvement):
            raise ValueError("Actor safety threshold must be finite")
        if self.actor_safety_cooldown_chunks < 0:
            raise ValueError("Actor safety cooldown must be non-negative")
        if self.actor_max_consecutive_chunks < 0:
            raise ValueError("Actor consecutive-chunk cap must be non-negative")
        if self.actor_throttle_cooldown_chunks < 0:
            raise ValueError("Actor throttle cooldown must be non-negative")
        if self.actor_success_bc_min_residual_rms > 1.0:
            raise ValueError("Success-BC residual RMS threshold must lie in [0, 1]")
        if self.actor_success_credit_mode not in (
            "chunk_threshold",
            "best_positive_burst",
        ):
            raise ValueError("Unsupported actor success-credit mode")
        if self.actor_deployment_policy not in ("actor_mean", "success_retrieval"):
            raise ValueError("Unsupported actor deployment policy")
        if self.retrieval_max_step_distance < 0:
            raise ValueError("Retrieval step distance must be non-negative")
        if not -1.0 <= self.retrieval_min_similarity <= 1.0:
            raise ValueError("Retrieval minimum similarity must lie in [-1, 1]")
        if self.action_low is None or self.action_high is None:
            raise ValueError("action_low and action_high are required")
        if len(self.action_low) != self.action_dim or len(self.action_high) != self.action_dim:
            raise ValueError("Action bounds must match action_dim")
        low = np.asarray(self.action_low, dtype=np.float32)
        high = np.asarray(self.action_high, dtype=np.float32)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Action bounds must be finite")
        if np.any(low >= high):
            raise ValueError("Every action lower bound must be below its upper bound")


@dataclass(frozen=True)
class VGateConfig:
    enter_failure_probability: float = 0.65
    immediate_failure_probability: float = 0.85
    exit_failure_probability: float = 0.25
    enter_confirm_chunks: int = 2
    exit_confirm_chunks: int = 2
    min_active_chunks: int = 1
    ema_alpha: float = 0.5
    immediate_max_entropy: float = 1.5
    latest_entry_step: int = 400
    max_intervention_env_steps: int = 500

    def __post_init__(self) -> None:
        probabilities = (
            self.enter_failure_probability,
            self.immediate_failure_probability,
            self.exit_failure_probability,
            self.ema_alpha,
        )
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("VGate probabilities and EMA alpha must lie in [0, 1]")
        if self.exit_failure_probability >= self.enter_failure_probability:
            raise ValueError("VGate exit probability must be below enter probability")
        if self.immediate_failure_probability < self.enter_failure_probability:
            raise ValueError("Immediate VGate threshold must be at least the enter threshold")
        if (
            min(
                self.enter_confirm_chunks,
                self.exit_confirm_chunks,
                self.min_active_chunks,
            )
            <= 0
        ):
            raise ValueError("VGate confirmation and hold counts must be positive")
        if (
            min(
                self.immediate_max_entropy,
                self.latest_entry_step,
                self.max_intervention_env_steps,
            )
            < 0
        ):
            raise ValueError("VGate limits must be non-negative")


@dataclass(frozen=True)
class VGateDecision:
    active: bool
    event: str
    smoothed_failure_probability: float
    risk_confirmations: int
    recovery_confirmations: int
    active_chunks: int
    intervention_env_steps: int


def update_actor_gate_authorization(
    authorized: bool,
    decision: VGateDecision,
    *,
    require_immediate_entry: bool,
) -> bool:
    """Track whether a persistent actor may act during the current gate episode.

    A confirmed entry is deliberately not enough when ``require_immediate_entry``
    is enabled: delayed, low-confidence entries are a common source of regressions
    against a base policy that would otherwise have succeeded.
    """

    if not decision.active:
        return False
    if decision.event == "ENTER_IMMEDIATE":
        return True
    if decision.event == "ENTER_CONFIRMED":
        return not require_immediate_entry
    return bool(authorized)


@dataclass(frozen=True)
class ActorChunkThrottleDecision:
    allowed: bool
    event: str
    consecutive_chunks: int
    cooldown_remaining: int


class ActorChunkThrottle:
    """Bound deterministic actor persistence consistently in train and eval."""

    def __init__(self, *, max_consecutive_chunks: int, cooldown_chunks: int) -> None:
        if max_consecutive_chunks < 0:
            raise ValueError("Actor consecutive-chunk cap must be non-negative")
        if cooldown_chunks < 0:
            raise ValueError("Actor throttle cooldown must be non-negative")
        self.max_consecutive_chunks = int(max_consecutive_chunks)
        self.cooldown_chunks = int(cooldown_chunks)
        self.reset()

    def reset(self) -> None:
        self.consecutive_chunks = 0
        self.cooldown_remaining = 0

    def decide(self, *, requested: bool) -> ActorChunkThrottleDecision:
        requested = bool(requested)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.consecutive_chunks = 0
            return ActorChunkThrottleDecision(
                allowed=False,
                event="COOLDOWN" if requested else "IDLE_COOLDOWN",
                consecutive_chunks=0,
                cooldown_remaining=self.cooldown_remaining,
            )
        if not requested:
            self.consecutive_chunks = 0
            return ActorChunkThrottleDecision(
                allowed=False,
                event="IDLE",
                consecutive_chunks=0,
                cooldown_remaining=0,
            )
        if (
            self.max_consecutive_chunks > 0
            and self.consecutive_chunks >= self.max_consecutive_chunks
        ):
            self.consecutive_chunks = 0
            self.cooldown_remaining = max(self.cooldown_chunks - 1, 0)
            return ActorChunkThrottleDecision(
                allowed=False,
                event="THROTTLED",
                consecutive_chunks=0,
                cooldown_remaining=self.cooldown_remaining,
            )
        self.consecutive_chunks += 1
        return ActorChunkThrottleDecision(
            allowed=True,
            event="ALLOW",
            consecutive_chunks=self.consecutive_chunks,
            cooldown_remaining=0,
        )


class VGate:
    """State machine that uses only frozen V outputs for entry and exit."""

    def __init__(self, config: VGateConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.smoothed_failure_probability: float | None = None
        self.risk_confirmations = 0
        self.recovery_confirmations = 0
        self.active_chunks = 0
        self.intervention_env_steps = 0

    def decide(
        self,
        *,
        failure_probability: float,
        entropy: float,
        step_id: int,
    ) -> VGateDecision:
        if not np.isfinite(failure_probability) or not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure_probability must be finite and lie in [0, 1]")
        if not np.isfinite(entropy) or entropy < 0.0:
            raise ValueError("entropy must be finite and non-negative")
        if step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.smoothed_failure_probability is None:
            self.smoothed_failure_probability = float(failure_probability)
        else:
            alpha = self.config.ema_alpha
            self.smoothed_failure_probability = (
                alpha * float(failure_probability)
                + (1.0 - alpha) * self.smoothed_failure_probability
            )
        risk = self.smoothed_failure_probability
        event = "HOLD" if self.active else "IDLE"

        if self.active:
            self.active_chunks += 1
            if self.intervention_env_steps >= self.config.max_intervention_env_steps:
                self.active = False
                self.recovery_confirmations = 0
                event = "EXIT_BUDGET"
            elif (
                self.active_chunks >= self.config.min_active_chunks
                and risk <= self.config.exit_failure_probability
            ):
                self.recovery_confirmations += 1
                if self.recovery_confirmations >= self.config.exit_confirm_chunks:
                    self.active = False
                    self.recovery_confirmations = 0
                    event = "EXIT_RECOVERED"
                else:
                    event = "HOLD_EXIT_CONFIRM"
            else:
                self.recovery_confirmations = 0
        else:
            self.active_chunks = 0
            self.recovery_confirmations = 0
            if self.intervention_env_steps >= self.config.max_intervention_env_steps:
                self.risk_confirmations = 0
                event = "IDLE_BUDGET_EXHAUSTED"
            elif step_id > self.config.latest_entry_step:
                self.risk_confirmations = 0
                event = "IDLE_TOO_LATE"
            elif (
                risk >= self.config.immediate_failure_probability
                and entropy <= self.config.immediate_max_entropy
            ):
                self.active = True
                self.active_chunks = 1
                self.risk_confirmations = 0
                event = "ENTER_IMMEDIATE"
            elif risk >= self.config.enter_failure_probability:
                self.risk_confirmations += 1
                if self.risk_confirmations >= self.config.enter_confirm_chunks:
                    self.active = True
                    self.active_chunks = 1
                    self.risk_confirmations = 0
                    event = "ENTER_CONFIRMED"
                else:
                    event = "IDLE_ENTER_CONFIRM"
            else:
                self.risk_confirmations = 0

        return VGateDecision(
            active=bool(self.active),
            event=event,
            smoothed_failure_probability=float(risk),
            risk_confirmations=int(self.risk_confirmations),
            recovery_confirmations=int(self.recovery_confirmations),
            active_chunks=int(self.active_chunks),
            intervention_env_steps=int(self.intervention_env_steps),
        )

    def observe_execution(self, *, duration: int, intervened: bool) -> None:
        if duration <= 0:
            raise ValueError("Executed duration must be positive")
        if intervened:
            self.intervention_env_steps += int(duration)


def value_improvement_target(
    *,
    current_potential: float,
    next_potential: float | None,
    terminal: bool,
    success: bool,
    failure_value: float,
) -> float:
    """Observed one-chunk improvement in the frozen V utility space.

    This deliberately excludes environment rewards. The distributional V support
    defines terminal success as zero and terminal failure as ``failure_value``.
    """

    values = (current_potential, failure_value)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Value improvement inputs must be finite")
    if failure_value >= 0.0:
        raise ValueError("failure_value must be negative")
    if terminal:
        terminal_value = 0.0 if success else float(failure_value)
        return terminal_value - float(current_potential)
    if next_potential is None or not np.isfinite(next_potential):
        raise ValueError("Non-terminal improvement requires a finite next potential")
    if success:
        raise ValueError("A non-terminal transition cannot already be terminal success")
    return float(next_potential) - float(current_potential)


@dataclass(slots=True)
class ValueBanditBatch:
    states: np.ndarray
    latents: np.ndarray
    ref_chunks: np.ndarray
    residuals: np.ndarray
    value_improvements: np.ndarray
    environment_returns: np.ndarray
    durations: np.ndarray
    step_ids: np.ndarray
    gate_active: np.ndarray
    exploration_active: np.ndarray
    terminal_success: np.ndarray
    terminal_failure: np.ndarray
    episode_ids: np.ndarray
    outcome_successes: np.ndarray
    success_credited: np.ndarray


def linear_curriculum_value(
    step: int,
    *,
    start_step: int,
    anneal_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    """Hold a value until ``start_step``, then linearly anneal it."""

    if step < 0 or start_step < 0:
        raise ValueError("Curriculum steps must be non-negative")
    if anneal_steps <= 0:
        raise ValueError("Curriculum anneal_steps must be positive")
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        raise ValueError("Curriculum values must be finite")
    progress = np.clip((step - start_step) / float(anneal_steps), 0.0, 1.0)
    return float(start_value + progress * (end_value - start_value))


class PersistentResidualExplorer:
    """Generate temporally correlated residual noise across chunk boundaries."""

    def __init__(
        self,
        action_dim: int,
        *,
        correlation: float,
        seed: int,
        gripper_scale: float = 0.5,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("Persistent explorer action_dim must be positive")
        if not 0.0 <= correlation < 1.0:
            raise ValueError("Persistent explorer correlation must lie in [0, 1)")
        if not 0.0 <= gripper_scale <= 1.0:
            raise ValueError("Persistent explorer gripper scale must lie in [0, 1]")
        self.action_dim = int(action_dim)
        self.correlation = float(correlation)
        self.gripper_scale = float(gripper_scale)
        self._innovation_scale = float(np.sqrt(1.0 - correlation**2))
        self._rng = np.random.default_rng(seed)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def start_burst(self, *, total_steps: int, std: float) -> None:
        if total_steps <= 0:
            raise ValueError("Exploration burst length must be positive")
        if not 0.0 <= std <= 1.0:
            raise ValueError("Persistent explorer std must lie in [0, 1]")
        self.reset()

    def sample(self, chunk_len: int, *, std: float) -> np.ndarray:
        if chunk_len <= 0:
            raise ValueError("Persistent explorer chunk_len must be positive")
        if not 0.0 <= std <= 1.0:
            raise ValueError("Persistent explorer std must lie in [0, 1]")
        values = np.empty((chunk_len, self.action_dim), dtype=np.float32)
        if self._state is None:
            self._state = self._rng.normal(size=self.action_dim).astype(np.float32)
        for index in range(chunk_len):
            innovation = self._rng.normal(size=self.action_dim).astype(np.float32)
            self._state = (
                self.correlation * self._state + self._innovation_scale * innovation
            ).astype(np.float32)
            values[index] = self._state
        values *= float(std)
        values[:, -1] *= self.gripper_scale
        return values


class SmoothKnotResidualExplorer:
    """Generate one smooth low-dimensional residual path for a whole burst."""

    def __init__(
        self,
        action_dim: int,
        *,
        knot_count: int,
        seed: int,
        gripper_scale: float = 0.5,
    ) -> None:
        if min(action_dim, knot_count) <= 0:
            raise ValueError("Smooth explorer dimensions must be positive")
        if not 0.0 <= gripper_scale <= 1.0:
            raise ValueError("Smooth explorer gripper scale must lie in [0, 1]")
        self.action_dim = int(action_dim)
        self.knot_count = int(knot_count)
        self.gripper_scale = float(gripper_scale)
        self._rng = np.random.default_rng(seed)
        self._trajectory: np.ndarray | None = None
        self._position = 0
        self._std: float | None = None

    def reset(self) -> None:
        self._trajectory = None
        self._position = 0
        self._std = None

    def start_burst(self, *, total_steps: int, std: float) -> None:
        if total_steps < self.knot_count:
            raise ValueError("Smooth burst must contain at least one step per knot")
        if not 0.0 <= std <= 1.0:
            raise ValueError("Smooth explorer std must lie in [0, 1]")
        knots = self._rng.normal(size=(self.knot_count, self.action_dim)).astype(np.float32)
        knot_positions = np.linspace(0.0, float(total_steps - 1), self.knot_count)
        step_positions = np.arange(total_steps, dtype=np.float32)
        trajectory = np.empty((total_steps, self.action_dim), dtype=np.float32)
        for action_index in range(self.action_dim):
            trajectory[:, action_index] = np.interp(
                step_positions,
                knot_positions,
                knots[:, action_index],
            )
        trajectory *= float(std)
        trajectory[:, -1] *= self.gripper_scale
        self._trajectory = trajectory
        self._position = 0
        self._std = float(std)

    def sample(self, chunk_len: int, *, std: float) -> np.ndarray:
        if chunk_len <= 0:
            raise ValueError("Smooth explorer chunk length must be positive")
        if self._trajectory is None or self._std is None:
            raise RuntimeError("Smooth exploration burst was not started")
        if not 0.0 <= std <= 1.0:
            raise ValueError("Smooth explorer std must lie in [0, 1]")
        end = self._position + int(chunk_len)
        if end > len(self._trajectory):
            raise RuntimeError("Smooth exploration burst is exhausted")
        values = self._trajectory[self._position : end].copy()
        self._position = end
        return values


@dataclass(frozen=True)
class ExplorationBurstDecision:
    explore: bool
    event: str
    bursts_started: int
    chunks_remaining: int
    cooldown_remaining: int


class ExplorationBurstSchedule:
    """Bound persistent noise while allowing the learned actor to remain active."""

    def __init__(
        self,
        *,
        burst_chunks: int,
        max_bursts: int,
        cooldown_chunks: int,
    ) -> None:
        if min(burst_chunks, max_bursts) <= 0 or cooldown_chunks < 0:
            raise ValueError("Exploration burst sizes must be positive and cooldown non-negative")
        self.burst_chunks = int(burst_chunks)
        self.max_bursts = int(max_bursts)
        self.cooldown_chunks = int(cooldown_chunks)
        self.bursts_started = 0
        self.chunks_remaining = 0
        self.cooldown_remaining = 0

    def decide(self, *, gate_active: bool, start_requested: bool) -> ExplorationBurstDecision:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        event = "IDLE"
        explore = False
        if not gate_active:
            if self.chunks_remaining:
                self.chunks_remaining = 0
                self.cooldown_remaining = self.cooldown_chunks
                event = "CANCEL_GATE_EXIT"
        elif self.chunks_remaining > 0:
            explore = True
            self.chunks_remaining -= 1
            event = "BURST_HOLD"
        elif self.cooldown_remaining > 0:
            event = "COOLDOWN"
        elif self.bursts_started >= self.max_bursts:
            event = "BUDGET_EXHAUSTED"
        elif start_requested:
            explore = True
            self.bursts_started += 1
            self.chunks_remaining = self.burst_chunks - 1
            event = "BURST_START"
        else:
            event = "START_SKIPPED"
        if explore and self.chunks_remaining == 0:
            self.cooldown_remaining = self.cooldown_chunks
            event = "BURST_END" if event == "BURST_HOLD" else "BURST_SINGLE"
        return ExplorationBurstDecision(
            explore=explore,
            event=event,
            bursts_started=self.bursts_started,
            chunks_remaining=self.chunks_remaining,
            cooldown_remaining=self.cooldown_remaining,
        )


class ValueBanditReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int,
        latent_dim: int,
        chunk_len: int,
        action_dim: int,
        seed: int,
    ) -> None:
        if min(capacity, state_dim, latent_dim, chunk_len, action_dim) <= 0:
            raise ValueError("Replay dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.latents = np.empty((capacity, latent_dim), dtype=np.float16)
        self.ref_chunks = np.empty((capacity, chunk_len, action_dim), dtype=np.float32)
        self.residuals = np.empty((capacity, chunk_len, action_dim), dtype=np.float32)
        self.value_improvements = np.empty((capacity,), dtype=np.float32)
        self.environment_returns = np.empty((capacity,), dtype=np.float32)
        self.durations = np.empty((capacity,), dtype=np.int32)
        self.step_ids = np.empty((capacity,), dtype=np.int32)
        self.gate_active = np.empty((capacity,), dtype=np.bool_)
        self.exploration_active = np.empty((capacity,), dtype=np.bool_)
        self.terminal_success = np.empty((capacity,), dtype=np.bool_)
        self.terminal_failure = np.empty((capacity,), dtype=np.bool_)
        self.episode_ids = np.empty((capacity,), dtype=np.int64)
        self.outcome_successes = np.empty((capacity,), dtype=np.int8)
        self.success_credited = np.empty((capacity,), dtype=np.bool_)
        self.pos = 0
        self.full = False
        self._rng = np.random.default_rng(seed)
        self.last_load_was_exact = False
        self.last_loaded_snapshot_id: str | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def _indices(self) -> np.ndarray:
        return np.arange(len(self), dtype=np.int64)

    @staticmethod
    def _finite(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"Replay {name} must have finite shape {shape}")
        return array

    def add(
        self,
        *,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        residual: np.ndarray,
        value_improvement: float,
        environment_return: float,
        duration: int,
        step_id: int,
        gate_active: bool,
        exploration_active: bool,
        terminal_success: bool,
        terminal_failure: bool,
        episode_id: int = -1,
        outcome_success: int = -1,
    ) -> int:
        if not 1 <= int(duration) <= self.chunk_len:
            raise ValueError("Replay duration is outside the chunk")
        if int(step_id) < 0:
            raise ValueError("Replay step_id must be non-negative")
        if terminal_success and terminal_failure:
            raise ValueError("A transition cannot be terminal success and failure")
        if episode_id < -1:
            raise ValueError("Replay episode_id must be -1 or non-negative")
        if outcome_success not in (-1, 0, 1):
            raise ValueError("Replay outcome_success must be pending, failure, or success")
        scalars = (value_improvement, environment_return)
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("Replay scalar targets must be finite")
        index = self.pos
        self.states[index] = self._finite("state", state, (self.state_dim,))
        latent_array = self._finite("latent", latent, (self.latent_dim,))
        if np.any(np.abs(latent_array) > np.finfo(np.float16).max):
            raise ValueError("Replay latent overflows float16 storage")
        self.latents[index] = latent_array.astype(np.float16)
        self.ref_chunks[index] = self._finite(
            "ref_chunk", ref_chunk, (self.chunk_len, self.action_dim)
        )
        residual_array = self._finite("residual", residual, (self.chunk_len, self.action_dim))
        if np.any(np.abs(residual_array) > 1.0 + 1e-6):
            raise ValueError("Replay residual lies outside [-1, 1]")
        self.residuals[index] = np.clip(residual_array, -1.0, 1.0)
        self.value_improvements[index] = float(value_improvement)
        self.environment_returns[index] = float(environment_return)
        self.durations[index] = int(duration)
        self.step_ids[index] = int(step_id)
        self.gate_active[index] = bool(gate_active)
        self.exploration_active[index] = bool(exploration_active)
        self.terminal_success[index] = bool(terminal_success)
        self.terminal_failure[index] = bool(terminal_failure)
        self.episode_ids[index] = int(episode_id)
        self.outcome_successes[index] = int(outcome_success)
        self.success_credited[index] = False
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
        return int(index)

    def finalize_episode(
        self,
        episode_id: int,
        *,
        success: bool,
        success_credit_mode: str = "chunk_threshold",
        min_burst_improvement: float = 0.0,
        scorer_return_mode: str = "one_step",
        scorer_return_gamma: float = 0.99,
        scorer_trace_chunks: int = 5,
    ) -> int:
        if episode_id < 0:
            raise ValueError("Finalized episode_id must be non-negative")
        indices = self._indices()
        selected = indices[self.episode_ids[indices] == int(episode_id)]
        if not len(selected):
            raise ValueError(f"Replay has no transitions for episode {episode_id}")
        if np.any(self.outcome_successes[selected] != -1):
            raise ValueError(f"Replay episode {episode_id} was already finalized")
        if success_credit_mode not in ("chunk_threshold", "best_positive_burst"):
            raise ValueError("Unsupported success-credit mode")
        if scorer_return_mode not in (
            "one_step",
            "episode_mc",
            "bounded_reward_trace",
        ):
            raise ValueError("Unsupported scorer return mode")
        if not 0.0 <= scorer_return_gamma <= 1.0:
            raise ValueError("Scorer return gamma must lie in [0, 1]")
        if scorer_trace_chunks <= 0:
            raise ValueError("Scorer trace chunks must be positive")
        if not np.isfinite(min_burst_improvement):
            raise ValueError("Minimum burst improvement must be finite")
        self.outcome_successes[selected] = int(bool(success))
        self.success_credited[selected] = False
        if scorer_return_mode == "episode_mc":
            order = selected[np.argsort(self.step_ids[selected])]
            running = 0.0
            for index in order[::-1]:
                discount = float(scorer_return_gamma) ** (
                    int(self.durations[index]) / float(self.chunk_len)
                )
                running = float(self.value_improvements[index]) + discount * running
                self.value_improvements[index] = running
        elif scorer_return_mode == "bounded_reward_trace":
            order = selected[np.argsort(self.step_ids[selected])]
            for position, index in enumerate(order):
                if not self.exploration_active[index]:
                    continue
                stop = min(len(order), position + int(scorer_trace_chunks) + 1)
                for future_position in range(position + 1, stop):
                    future = order[future_position]
                    reward = float(self.environment_returns[future])
                    if reward > 0.0:
                        lag = future_position - position
                        self.value_improvements[index] += (
                            float(scorer_return_gamma) ** lag
                        ) * reward
        if success:
            nonzero = np.any(np.abs(self.residuals[selected]) > 1e-6, axis=(1, 2))
            exploratory = selected[nonzero & self.exploration_active[selected]]
            if success_credit_mode == "chunk_threshold":
                self.success_credited[exploratory] = True
            elif len(exploratory):
                order = exploratory[np.argsort(self.step_ids[exploratory])]
                groups: list[np.ndarray] = []
                start = 0
                for offset in range(1, len(order)):
                    if self.step_ids[order[offset]] != (
                        self.step_ids[order[offset - 1]] + self.chunk_len
                    ):
                        groups.append(order[start:offset])
                        start = offset
                groups.append(order[start:])
                scores = np.asarray(
                    [np.sum(self.value_improvements[group]) for group in groups],
                    dtype=np.float32,
                )
                best = int(np.argmax(scores))
                if float(scores[best]) > max(0.0, float(min_burst_improvement)):
                    self.success_credited[groups[best]] = True
        return int(len(selected))

    def batch(self, indices: np.ndarray) -> ValueBanditBatch:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("Replay indices are invalid")
        return ValueBanditBatch(
            states=self.states[indices].astype(np.float32, copy=True),
            latents=self.latents[indices].astype(np.float32),
            ref_chunks=self.ref_chunks[indices].copy(),
            residuals=self.residuals[indices].copy(),
            value_improvements=self.value_improvements[indices].copy(),
            environment_returns=self.environment_returns[indices].copy(),
            durations=self.durations[indices].copy(),
            step_ids=self.step_ids[indices].copy(),
            gate_active=self.gate_active[indices].copy(),
            exploration_active=self.exploration_active[indices].copy(),
            terminal_success=self.terminal_success[indices].copy(),
            terminal_failure=self.terminal_failure[indices].copy(),
            episode_ids=self.episode_ids[indices].copy(),
            outcome_successes=self.outcome_successes[indices].copy(),
            success_credited=self.success_credited[indices].copy(),
        )

    def sample(
        self,
        batch_size: int,
        *,
        nonzero_fraction: float = 0.5,
        nonzero_success_fraction: float = 0.5,
    ) -> ValueBanditBatch:
        if batch_size <= 0 or len(self) == 0:
            raise ValueError("Cannot sample an empty replay or non-positive batch")
        if not 0.0 <= nonzero_fraction <= 1.0:
            raise ValueError("nonzero_fraction must lie in [0, 1]")
        if not 0.0 <= nonzero_success_fraction <= 1.0:
            raise ValueError("nonzero_success_fraction must lie in [0, 1]")
        indices = self._indices()
        nonzero = np.any(np.abs(self.residuals[indices]) > 1e-6, axis=(1, 2))
        zero_pool = indices[~nonzero]
        nonzero_pool = indices[nonzero]
        requested_nonzero = int(round(batch_size * nonzero_fraction))
        successful_mask = nonzero & self.success_credited[indices]
        successful_nonzero = indices[successful_mask]
        other_nonzero = indices[nonzero & ~successful_mask]
        if requested_nonzero and len(nonzero_pool) and len(zero_pool):
            requested_success = int(round(requested_nonzero * nonzero_success_fraction))
            if requested_success and len(successful_nonzero) and len(other_nonzero):
                selected_success = self._rng.choice(
                    successful_nonzero, size=requested_success, replace=True
                )
                selected_other = self._rng.choice(
                    other_nonzero,
                    size=requested_nonzero - requested_success,
                    replace=True,
                )
                selected_nonzero = np.concatenate([selected_success, selected_other])
            else:
                selected_nonzero = self._rng.choice(
                    nonzero_pool, size=requested_nonzero, replace=True
                )
            selected_zero = self._rng.choice(
                zero_pool, size=batch_size - requested_nonzero, replace=True
            )
            selected = np.concatenate([selected_zero, selected_nonzero])
            self._rng.shuffle(selected)
        else:
            selected = self._rng.choice(indices, size=batch_size, replace=True)
        return self.batch(selected)

    def pool_counts(self) -> dict[str, int]:
        indices = self._indices()
        if not len(indices):
            return {
                "zero": 0,
                "nonzero": 0,
                "positive_nonzero": 0,
                "successful_nonzero": 0,
                "failed_nonzero": 0,
                "pending": 0,
            }
        nonzero = np.any(np.abs(self.residuals[indices]) > 1e-6, axis=(1, 2))
        positive = self.value_improvements[indices] > 0.0
        outcome = self.outcome_successes[indices]
        successful = nonzero & self.success_credited[indices]
        return {
            "zero": int(np.count_nonzero(~nonzero)),
            "nonzero": int(np.count_nonzero(nonzero)),
            "positive_nonzero": int(np.count_nonzero(nonzero & positive)),
            "successful_nonzero": int(np.count_nonzero(successful)),
            "failed_nonzero": int(np.count_nonzero(nonzero & (outcome == 0))),
            "pending": int(np.count_nonzero(outcome == -1)),
        }

    def save(self, path: str | Path, *, snapshot_id: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = len(self)
        payload: dict[str, Any] = {
            "schema_version": np.asarray(VALUE_GUIDED_BANDIT_REPLAY_SCHEMA),
            "capacity": np.asarray(self.capacity),
            "pos": np.asarray(self.pos),
            "full": np.asarray(self.full),
            "rng_state": np.asarray(json.dumps(self._rng.bit_generator.state)),
            "snapshot_id": np.asarray(snapshot_id or ""),
            "state_dim": np.asarray(self.state_dim),
            "latent_dim": np.asarray(self.latent_dim),
            "chunk_len": np.asarray(self.chunk_len),
            "action_dim": np.asarray(self.action_dim),
        }
        for name in ValueBanditBatch.__dataclass_fields__:
            payload[name] = getattr(self, name)[:size]
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
        try:
            np.savez_compressed(temporary, **payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: str | Path) -> int:
        with np.load(path, allow_pickle=False) as data:
            schema = int(data["schema_version"])
            if schema not in (1, 2, 3, VALUE_GUIDED_BANDIT_REPLAY_SCHEMA):
                raise ValueError("Unsupported value-bandit replay schema")
            expected = {
                "capacity": self.capacity,
                "state_dim": self.state_dim,
                "latent_dim": self.latent_dim,
                "chunk_len": self.chunk_len,
                "action_dim": self.action_dim,
            }
            mismatches = {
                key: (expected_value, int(data[key]))
                for key, expected_value in expected.items()
                if int(data[key]) != expected_value
            }
            if mismatches:
                raise ValueError(f"Value-bandit replay layout mismatch: {mismatches}")
            source_size = int(data["capacity"]) if bool(data["full"]) else int(data["pos"])
            legacy_fields = {"episode_ids", "outcome_successes"}
            for name in ValueBanditBatch.__dataclass_fields__:
                if schema == 1 and name in legacy_fields:
                    getattr(self, name)[:source_size] = -1
                elif schema < 3 and name == "exploration_active":
                    # Older replay did not distinguish exploratory residuals
                    # from deterministic actor continuations. Preserve load
                    # compatibility without inventing background-probe labels.
                    self.exploration_active[:source_size] = data["gate_active"]
                elif schema < 4 and name == "success_credited":
                    self.success_credited[:source_size] = (
                        self.exploration_active[:source_size]
                        & (self.outcome_successes[:source_size] == 1)
                    )
                else:
                    getattr(self, name)[:source_size] = data[name]
            self.pos = int(data["pos"])
            self.full = bool(data["full"])
            self._rng.bit_generator.state = json.loads(str(data["rng_state"].item()))
            snapshot = str(data["snapshot_id"].item())
            self.last_loaded_snapshot_id = snapshot or None
            self.last_load_was_exact = True
            return source_size


class ValueDeltaScorer(nn.Module):
    """Ensemble predicting the observed one-chunk frozen-V improvement."""

    def __init__(self, config: ValueGuidedBanditConfig) -> None:
        super().__init__()
        input_dim = config.context_dim + config.chunk_len * config.action_dim
        self.heads = nn.ModuleList(
            [
                _mlp(input_dim, 1, config.hidden_dim, config.num_layers)
                for _ in range(config.num_scorers)
            ]
        )

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        value = torch.cat([context, residual.flatten(start_dim=1)], dim=-1)
        return torch.stack([head(value).squeeze(-1) for head in self.heads], dim=-1)


class MultiHypothesisResidualActor(nn.Module):
    """Context-selected residual modes trained with winner-take-all regression."""

    def __init__(self, config: ValueGuidedBanditConfig) -> None:
        super().__init__()
        self.config = config
        self.features = _mlp(
            config.context_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.num_layers,
        )
        self.residual_head = nn.Linear(
            config.hidden_dim,
            config.actor_hypotheses * config.exploration_knots * config.action_dim,
        )
        self.selection_head = nn.Linear(config.hidden_dim, config.actor_hypotheses)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.selection_head.weight)
        nn.init.zeros_(self.selection_head.bias)

    def _expand_knots(self, knots: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            knots.permute(0, 2, 1),
            size=self.config.chunk_len,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

    def components(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(context)
        raw = self.residual_head(features).reshape(
            -1,
            self.config.actor_hypotheses,
            self.config.exploration_knots,
            self.config.action_dim,
        )
        knots = self.config.actor_residual_limit * torch.tanh(raw)
        batch_size = knots.shape[0]
        means = self._expand_knots(
            knots.reshape(
                batch_size * self.config.actor_hypotheses,
                self.config.exploration_knots,
                self.config.action_dim,
            )
        ).reshape(
            batch_size,
            self.config.actor_hypotheses,
            self.config.chunk_len,
            self.config.action_dim,
        )
        return means, self.selection_head(features)

    def all_means(self, context: torch.Tensor) -> torch.Tensor:
        return self.components(context)[0]

    def selection_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.components(context)[1]

    def mean(self, context: torch.Tensor) -> torch.Tensor:
        means, logits = self.components(context)
        selected = logits.argmax(dim=-1)
        batch = torch.arange(means.shape[0], device=means.device)
        return means[batch, selected]


class ValueGuidedBanditAgent:
    checkpoint_version = 6

    def __init__(
        self,
        config: ValueGuidedBanditConfig,
        *,
        device: str | torch.device = "cpu",
        runtime_identity: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.runtime_identity = copy.deepcopy(runtime_identity)
        self.context = FrozenLatentContextEncoder(config).to(self.device)
        self.actor_context = (
            FrozenLatentContextEncoder(config).to(self.device)
            if config.actor_context_trainable
            else None
        )
        self.actor = (
            MultiHypothesisResidualActor(config)
            if config.actor_hypotheses > 1
            else ContinuousResidualActor(config)
        ).to(self.device)
        self.scorer = ValueDeltaScorer(config).to(self.device)
        self.scorer_opt = torch.optim.AdamW(
            [*self.context.parameters(), *self.scorer.parameters()],
            lr=config.scorer_lr,
        )
        actor_parameters = list(self.actor.parameters())
        if self.actor_context is not None:
            actor_parameters.extend(self.actor_context.parameters())
        self.actor_opt = torch.optim.AdamW(actor_parameters, lr=config.actor_lr)
        low = np.asarray(config.action_low, dtype=np.float32)
        high = np.asarray(config.action_high, dtype=np.float32)
        fractions = np.full(config.action_dim, config.arm_residual_fraction, np.float32)
        fractions[-1] = config.gripper_residual_fraction
        self.action_low = low
        self.action_high = high
        self.residual_scale = (high - low) * fractions
        self._action_low_tensor = self._tensor(low).view(1, 1, -1)
        self._action_high_tensor = self._tensor(high).view(1, 1, -1)
        self._residual_scale_tensor = self._tensor(self.residual_scale).view(1, 1, -1)
        self.total_updates = 0
        self.scorer_updates = 0
        self.actor_updates = 0
        self.actor_context_initialized = self.actor_context is None
        self.success_memory_contexts = torch.empty(
            (0, config.context_dim), dtype=torch.float32, device=self.device
        )
        self.success_memory_residuals = torch.empty(
            (0, config.chunk_len, config.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.success_memory_step_ids = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        self.success_memory_next_indices = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        self._retrieval_next_index = -1
        self._retrieval_last_step: int | None = None
        self.snapshot_id: str | None = None

    def _tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _encode_actor_context(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        ref_chunk: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.actor_context if self.actor_context is not None else self.context
        return encoder(state, latent, ref_chunk, step_ids)

    def _initialize_actor_context_from_scorer(self) -> bool:
        if self.actor_context is None or self.actor_context_initialized:
            return False
        self.actor_context.load_state_dict(self.context.state_dict())
        self.actor_context_initialized = True
        return True

    def assert_runtime_identity(self, actual: dict[str, Any]) -> None:
        if self.runtime_identity != actual:
            expected = self.runtime_identity or {}
            mismatches = {
                key: (expected.get(key), actual.get(key))
                for key in sorted(set(expected) | set(actual))
                if expected.get(key) != actual.get(key)
            }
            raise ValueError(f"Bandit checkpoint runtime identity mismatch: {mismatches}")

    def _effective_residual(self, ref_chunk: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(
            ref_chunk + requested.clamp(-1.0, 1.0) * self._residual_scale_tensor,
            self._action_low_tensor,
            self._action_high_tensor,
        )
        return (action - ref_chunk) / self._residual_scale_tensor

    def apply_residual(self, ref_chunk: np.ndarray, residual: np.ndarray) -> np.ndarray:
        ref = np.asarray(ref_chunk, dtype=np.float32)
        normalized = np.asarray(residual, dtype=np.float32)
        expected = (self.config.chunk_len, self.config.action_dim)
        if ref.shape != expected or normalized.shape != expected:
            raise ValueError(f"Reference and residual must have shape {expected}")
        action = ref + np.clip(normalized, -1.0, 1.0) * self.residual_scale[None]
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def normalized_residual(
        self, ref_chunk: np.ndarray, action_chunk: np.ndarray
    ) -> np.ndarray:
        """Express an alternative action chunk inside the configured trust region."""

        ref = np.asarray(ref_chunk, dtype=np.float32)
        action = np.asarray(action_chunk, dtype=np.float32)
        expected = (self.config.chunk_len, self.config.action_dim)
        if ref.shape != expected or action.shape != expected:
            raise ValueError(f"Reference and action chunks must have shape {expected}")
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(action)):
            raise ValueError("Reference or action chunk contains NaN or Inf")
        return np.clip(
            (action - ref) / self.residual_scale[None], -1.0, 1.0
        ).astype(np.float32)

    def vla_tangent_residuals(
        self,
        ref_chunk: np.ndarray,
        alternative_chunk: np.ndarray,
        *,
        scales: Sequence[float],
    ) -> np.ndarray:
        """Scale a sampled VLA action direction inside the residual trust region."""

        scale_array = np.asarray(tuple(scales), dtype=np.float32)
        if (
            scale_array.ndim != 1
            or scale_array.size == 0
            or not np.all(np.isfinite(scale_array))
            or np.any(scale_array <= 0.0)
        ):
            raise ValueError("VLA tangent scales must be finite and positive")
        direction = self.normalized_residual(ref_chunk, alternative_chunk)
        return np.clip(
            scale_array[:, None, None] * direction[None], -1.0, 1.0
        ).astype(np.float32)

    @torch.no_grad()
    def propose_actor_residual(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
        exploration_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the actor residual, optionally shifted by correlated behavior noise."""

        state_t = self._tensor(np.asarray(state, dtype=np.float32)[None])
        latent_t = self._tensor(np.asarray(latent, dtype=np.float32)[None])
        ref_t = self._tensor(np.asarray(ref_chunk, dtype=np.float32)[None])
        steps_t = self._tensor(np.asarray([step_id]), torch.long)
        mean = self.actor.mean(
            self._encode_actor_context(state_t, latent_t, ref_t, steps_t)
        )
        requested = mean
        if exploration_noise is not None:
            noise = np.asarray(exploration_noise, dtype=np.float32)
            expected = (self.config.chunk_len, self.config.action_dim)
            if noise.shape != expected or not np.all(np.isfinite(noise)):
                raise ValueError(f"Exploration noise must have finite shape {expected}")
            requested = requested + self._tensor(noise[None])
        return self._effective_residual(ref_t, requested)[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def refresh_success_memory(self, replay: ValueBanditReplayBuffer) -> int:
        """Encode causally credited residual prototypes for deployment retrieval."""

        indices = replay._indices()
        selected = indices[replay.success_credited[indices]]
        if not len(selected):
            self.success_memory_contexts = torch.empty(
                (0, self.config.context_dim), dtype=torch.float32, device=self.device
            )
            self.success_memory_residuals = torch.empty(
                (0, self.config.chunk_len, self.config.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.success_memory_step_ids = torch.empty(
                (0,), dtype=torch.long, device=self.device
            )
            self.success_memory_next_indices = torch.empty(
                (0,), dtype=torch.long, device=self.device
            )
            self.reset_deployment_state()
            return 0
        batch = replay.batch(selected)
        contexts = self._encode_actor_context(
            self._tensor(batch.states),
            self._tensor(batch.latents),
            self._tensor(batch.ref_chunks),
            self._tensor(batch.step_ids, torch.long),
        )
        self.success_memory_contexts = F.normalize(contexts, dim=-1)
        self.success_memory_residuals = self._tensor(batch.residuals)
        self.success_memory_step_ids = self._tensor(batch.step_ids, torch.long)
        positions = {
            (int(episode), int(step)): position
            for position, (episode, step) in enumerate(
                zip(batch.episode_ids, batch.step_ids, strict=True)
            )
        }
        next_indices = np.asarray(
            [
                positions.get(
                    (int(episode), int(step) + self.config.chunk_len), -1
                )
                for episode, step in zip(
                    batch.episode_ids, batch.step_ids, strict=True
                )
            ],
            dtype=np.int64,
        )
        self.success_memory_next_indices = self._tensor(next_indices, torch.long)
        self.reset_deployment_state()
        return int(len(selected))

    def reset_deployment_state(self) -> None:
        self._retrieval_next_index = -1
        self._retrieval_last_step = None

    @torch.no_grad()
    def propose_success_retrieval_residual(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        if not len(self.success_memory_contexts):
            residual = self.propose_actor_residual(
                state, latent, ref_chunk, step_id=step_id
            )
            return residual, {
                "source": "ACTOR_MEAN_NO_MEMORY",
                "memory_index": -1,
                "similarity": float("nan"),
            }
        continuation = bool(
            self._retrieval_next_index >= 0
            and self._retrieval_last_step is not None
            and int(step_id) == self._retrieval_last_step + self.config.chunk_len
        )
        if continuation:
            memory_index = self._retrieval_next_index
            similarity = float("nan")
            source = "SUCCESS_RETRIEVAL_CONTINUE"
        else:
            context = self._encode_actor_context(
                self._tensor(np.asarray(state, dtype=np.float32)[None]),
                self._tensor(np.asarray(latent, dtype=np.float32)[None]),
                self._tensor(np.asarray(ref_chunk, dtype=np.float32)[None]),
                self._tensor(np.asarray([step_id]), torch.long),
            )
            similarities = self.success_memory_contexts @ F.normalize(context, dim=-1)[0]
            if self.config.retrieval_max_step_distance > 0:
                nearby = (
                    torch.abs(self.success_memory_step_ids - int(step_id))
                    <= self.config.retrieval_max_step_distance
                )
                if bool(nearby.any()):
                    similarities = similarities.masked_fill(~nearby, -torch.inf)
            memory_index = int(torch.argmax(similarities).item())
            similarity = float(similarities[memory_index].cpu())
            source = "SUCCESS_RETRIEVAL_SEARCH"
        if similarity < self.config.retrieval_min_similarity:
            residual = self.propose_actor_residual(
                state, latent, ref_chunk, step_id=step_id
            )
            source = "ACTOR_MEAN_LOW_RETRIEVAL_SIMILARITY"
            self.reset_deployment_state()
        else:
            residual = (
                self.success_memory_residuals[memory_index]
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            self._retrieval_next_index = int(
                self.success_memory_next_indices[memory_index].item()
            )
            self._retrieval_last_step = int(step_id)
        return residual, {
            "source": source,
            "memory_index": memory_index,
            "similarity": similarity,
        }

    def propose_deployment_residual(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        if self.config.actor_deployment_policy == "success_retrieval":
            return self.propose_success_retrieval_residual(
                state, latent, ref_chunk, step_id=step_id
            )
        return self.propose_actor_residual(
            state, latent, ref_chunk, step_id=step_id
        ), {
            "source": "ACTOR_MEAN",
            "memory_index": -1,
            "similarity": float("nan"),
        }

    @torch.no_grad()
    def propose_candidates(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        step_id: int,
        noise_std: float,
        seed: int,
    ) -> np.ndarray:
        if not 0.0 <= noise_std <= 1.0:
            raise ValueError("Candidate noise_std must lie in [0, 1]")
        state_t = self._tensor(np.asarray(state, dtype=np.float32)[None])
        latent_t = self._tensor(np.asarray(latent, dtype=np.float32)[None])
        ref_t = self._tensor(np.asarray(ref_chunk, dtype=np.float32)[None])
        steps_t = self._tensor(np.asarray([step_id]), torch.long)
        actor_context = self._encode_actor_context(state_t, latent_t, ref_t, steps_t)
        mean = self.actor.mean(actor_context)
        candidates = [torch.zeros_like(mean), mean]
        remaining = self.config.candidate_count - 2
        pairs = (remaining + 1) // 2
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        knots = torch.randn(
            pairs,
            self.config.exploration_knots,
            self.config.action_dim,
            generator=generator,
            device=self.device,
            dtype=mean.dtype,
        )
        noise = self.actor._expand_knots(knots) * float(noise_std)
        centered = mean.expand(pairs, -1, -1)
        symmetric = torch.cat([centered + noise, centered - noise], dim=0)
        candidates.append(symmetric[:remaining].clamp(-1.0, 1.0))
        requested = torch.cat(candidates, dim=0)
        ref_batch = ref_t.expand(len(requested), -1, -1)
        effective = self._effective_residual(ref_batch, requested)
        return effective.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def score_candidates(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        ref_chunk: np.ndarray,
        residuals: np.ndarray,
        *,
        step_id: int,
    ) -> dict[str, np.ndarray]:
        residual_array = np.asarray(residuals, dtype=np.float32)
        expected_tail = (self.config.chunk_len, self.config.action_dim)
        if residual_array.ndim != 3 or residual_array.shape[1:] != expected_tail:
            raise ValueError("Candidate residuals have an invalid shape")
        count = len(residual_array)
        state_t = self._tensor(np.repeat(np.asarray(state)[None], count, axis=0))
        latent_t = self._tensor(np.repeat(np.asarray(latent)[None], count, axis=0))
        ref_t = self._tensor(np.repeat(np.asarray(ref_chunk)[None], count, axis=0))
        steps_t = self._tensor(np.full(count, step_id), torch.long)
        residual_t = self._tensor(residual_array)
        context = self.context(state_t, latent_t, ref_t, steps_t)
        raw = self.scorer(context, residual_t)
        base = raw[0:1]
        advantages = raw - base
        mean = advantages.mean(dim=-1)
        std = advantages.std(dim=-1, unbiased=False)
        rms = torch.sqrt(residual_t.square().mean(dim=(1, 2)))
        lcb = (
            mean
            - self.config.selection_uncertainty_penalty * std
            - self.config.selection_residual_penalty * rms
        )
        return {
            "raw_heads": raw.cpu().numpy().astype(np.float32),
            "advantage_heads": advantages.cpu().numpy().astype(np.float32),
            "advantage_mean": mean.cpu().numpy().astype(np.float32),
            "advantage_std": std.cpu().numpy().astype(np.float32),
            "residual_rms": rms.cpu().numpy().astype(np.float32),
            "lcb": lcb.cpu().numpy().astype(np.float32),
        }

    def choose_candidate(
        self,
        scores: dict[str, np.ndarray],
        *,
        allow_exploration: bool,
    ) -> tuple[int, str]:
        lcb = np.asarray(scores["lcb"], dtype=np.float32)
        if lcb.ndim != 1 or len(lcb) != self.config.candidate_count:
            raise ValueError("Candidate scores have an invalid shape")
        selected = int(np.argmax(lcb))
        if selected != 0 and lcb[selected] >= self.config.selection_min_advantage:
            return selected, "LCB_SELECTED"
        if allow_exploration:
            mean = np.asarray(scores["advantage_mean"], dtype=np.float32)
            std = np.asarray(scores["advantage_std"], dtype=np.float32)
            residual_rms = np.asarray(scores["residual_rms"], dtype=np.float32)
            ucb = (
                mean
                + self.config.selection_uncertainty_penalty * std
                - self.config.selection_residual_penalty * residual_rms
            )
            return 1 + int(np.argmax(ucb[1:])), "UCB_EXPLORE"
        return 0, "BASE_FALLBACK"

    def update(
        self,
        batch: ValueBanditBatch,
        *,
        update_actor: bool,
    ) -> dict[str, float]:
        state = self._tensor(batch.states)
        latent = self._tensor(batch.latents)
        ref = self._tensor(batch.ref_chunks)
        residual = self._tensor(batch.residuals)
        target = self._tensor(batch.value_improvements)
        steps = self._tensor(batch.step_ids, torch.long)
        context = self.context(state, latent, ref, steps)
        raw = self.scorer(context, residual)
        masks = torch.rand_like(raw) < self.config.bootstrap_probability
        empty_heads = ~masks.any(dim=0)
        if bool(empty_heads.any()):
            masks[0, empty_heads] = True
        squared = (raw - target[:, None]).square()
        scorer_loss = (squared * masks).sum() / masks.sum().clamp_min(1)
        self.scorer_opt.zero_grad(set_to_none=True)
        scorer_loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                [*self.context.parameters(), *self.scorer.parameters()],
                self.config.grad_clip_norm,
            )
        self.scorer_opt.step()
        self.total_updates += 1
        self.scorer_updates += 1
        metrics = {
            "scorer_loss": float(scorer_loss.detach().cpu()),
            "scorer_prediction": float(raw.mean().detach().cpu()),
            "scorer_target": float(target.mean().detach().cpu()),
            "scorer_disagreement": float(raw.std(dim=-1, unbiased=False).mean().detach().cpu()),
            "actor_updated": 0.0,
            "actor_awr_samples": 0.0,
            "actor_awr_loss": 0.0,
            "actor_success_bc_loss": 0.0,
            "actor_success_bc_samples": 0.0,
            "actor_hypothesis_loss": 0.0,
            "actor_hypothesis_active_heads": 0.0,
        }

        if update_actor and self.total_updates % self.config.actor_update_period == 0:
            self._initialize_actor_context_from_scorer()
            if self.actor_context is None:
                with torch.no_grad():
                    actor_context = self.context(state, latent, ref, steps)
            else:
                actor_context = self.actor_context(state, latent, ref, steps)
            hypothesis_logits = None
            all_effective = None
            if isinstance(self.actor, MultiHypothesisResidualActor):
                all_predicted, hypothesis_logits = self.actor.components(actor_context)
                selected_hypotheses = hypothesis_logits.argmax(dim=-1)
                batch_indices = torch.arange(state.shape[0], device=self.device)
                predicted = all_predicted[batch_indices, selected_hypotheses]
                all_effective = self._effective_residual(
                    ref[:, None]
                    .expand(-1, self.config.actor_hypotheses, -1, -1)
                    .reshape(-1, self.config.chunk_len, self.config.action_dim),
                    all_predicted.reshape(
                        -1, self.config.chunk_len, self.config.action_dim
                    ),
                ).reshape_as(all_predicted)
                regularized_predictions = all_predicted
            else:
                predicted = self.actor.mean(actor_context)
                regularized_predictions = predicted[:, None]
            effective = self._effective_residual(ref, predicted)
            zero = torch.zeros_like(effective)
            self.scorer.requires_grad_(False)
            with torch.no_grad():
                scorer_actor_context = self.context(state, latent, ref, steps)
            proposed_raw = self.scorer(scorer_actor_context, effective)
            base_raw = self.scorer(scorer_actor_context, zero)
            proposed_advantages = proposed_raw - base_raw
            conservative_advantage = proposed_advantages.min(dim=-1).values
            actor_objective = conservative_advantage.mean()
            l2 = regularized_predictions.square().mean()
            smoothness = (
                regularized_predictions[:, :, 1:] - regularized_predictions[:, :, :-1]
            ).square().mean()
            actor_loss = (
                -self.config.actor_value_objective_weight * actor_objective
                + self.config.actor_l2_weight * l2
                + self.config.actor_smoothness_weight * smoothness
            )
            with torch.no_grad():
                realized_advantage = target - base_raw.mean(dim=-1)
                valid_steps = torch.arange(self.config.chunk_len, device=self.device).view(
                    1, -1
                ) < self._tensor(batch.durations, torch.long).view(-1, 1)
                valid = valid_steps.unsqueeze(-1)
                valid_count = valid.sum(dim=(1, 2)).clamp_min(1) * self.config.action_dim
                executed_rms = torch.sqrt((residual.square() * valid).sum(dim=(1, 2)) / valid_count)
                anchors = (realized_advantage >= self.config.actor_awr_min_advantage) & (
                    executed_rms > 1e-5
                )
                weights = torch.exp(
                    (realized_advantage / self.config.actor_awr_temperature).clamp(max=5.0)
                )
            if all_effective is not None:
                hypothesis_errors = (
                    (all_effective - residual[:, None]).square() * valid[:, None]
                ).sum(dim=(2, 3)) / valid_count[:, None]
                with torch.no_grad():
                    winner_hypotheses = hypothesis_errors.argmin(dim=-1)
                    preferred = torch.arange(state.shape[0], device=self.device) % (
                        self.config.actor_hypotheses
                    )
                    minimum = hypothesis_errors.min(dim=-1).values
                    preferred_is_tied = torch.isclose(
                        hypothesis_errors[
                            torch.arange(state.shape[0], device=self.device), preferred
                        ],
                        minimum,
                        rtol=1e-5,
                        atol=1e-8,
                    )
                    winner_hypotheses = torch.where(
                        preferred_is_tied, preferred, winner_hypotheses
                    )
                per_sample_bc = hypothesis_errors[
                    torch.arange(state.shape[0], device=self.device), winner_hypotheses
                ]
            else:
                winner_hypotheses = None
                per_sample_bc = (
                    (effective - residual).square() * valid
                ).sum(dim=(1, 2)) / valid_count
            if bool(anchors.any()):
                awr_loss = (per_sample_bc[anchors] * weights[anchors]).sum() / weights[
                    anchors
                ].sum().clamp_min(1e-6)
                actor_loss = actor_loss + self.config.actor_awr_weight * awr_loss
                awr_samples = int(anchors.sum().detach().cpu())
            else:
                awr_loss = actor_loss.new_zeros(())
                awr_samples = 0
            successful_anchors = self._tensor(batch.success_credited, torch.bool) & (
                executed_rms > self.config.actor_success_bc_min_residual_rms
            )
            if self.config.actor_success_credit_mode == "chunk_threshold":
                successful_anchors = successful_anchors & (
                    target >= self.config.actor_success_bc_min_value_improvement
                )
            if self.config.actor_success_bc_requires_gate:
                successful_anchors = successful_anchors & self._tensor(
                    batch.gate_active, torch.bool
                )
            if bool(successful_anchors.any()):
                success_bc = per_sample_bc[successful_anchors].mean()
                actor_loss = actor_loss + self.config.actor_success_bc_weight * success_bc
                success_bc_samples = int(successful_anchors.sum().detach().cpu())
            else:
                success_bc = actor_loss.new_zeros(())
                success_bc_samples = 0
            if hypothesis_logits is not None and winner_hypotheses is not None:
                hypothesis_anchors = successful_anchors
                if self.config.actor_awr_weight > 0.0:
                    hypothesis_anchors = hypothesis_anchors | anchors
                if bool(hypothesis_anchors.any()):
                    hypothesis_loss = F.cross_entropy(
                        hypothesis_logits[hypothesis_anchors],
                        winner_hypotheses[hypothesis_anchors],
                    )
                    actor_loss = (
                        actor_loss
                        + self.config.actor_hypothesis_loss_weight * hypothesis_loss
                    )
                    active_hypotheses = int(
                        winner_hypotheses[hypothesis_anchors].unique().numel()
                    )
                else:
                    hypothesis_loss = actor_loss.new_zeros(())
                    active_hypotheses = 0
            else:
                hypothesis_loss = actor_loss.new_zeros(())
                active_hypotheses = 0
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_clip_norm > 0:
                actor_parameters = list(self.actor.parameters())
                if self.actor_context is not None:
                    actor_parameters.extend(self.actor_context.parameters())
                nn.utils.clip_grad_norm_(actor_parameters, self.config.grad_clip_norm)
            self.actor_opt.step()
            self.scorer.requires_grad_(True)
            self.actor_updates += 1
            metrics.update(
                actor_updated=1.0,
                actor_loss=float(actor_loss.detach().cpu()),
                actor_value_advantage=float(actor_objective.detach().cpu()),
                actor_l2=float(l2.detach().cpu()),
                actor_smoothness=float(smoothness.detach().cpu()),
                actor_awr_loss=float(awr_loss.detach().cpu()),
                actor_awr_samples=float(awr_samples),
                actor_success_bc_loss=float(success_bc.detach().cpu()),
                actor_success_bc_samples=float(success_bc_samples),
                actor_hypothesis_loss=float(hypothesis_loss.detach().cpu()),
                actor_hypothesis_active_heads=float(active_hypotheses),
                realized_advantage_mean=float(realized_advantage.mean().detach().cpu()),
            )
        metrics.update(
            total_updates=float(self.total_updates),
            scorer_updates=float(self.scorer_updates),
            actor_updates=float(self.actor_updates),
        )
        return metrics

    def save(self, path: str | Path, *, snapshot_id: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": VALUE_GUIDED_BANDIT_CHECKPOINT_SCHEMA,
            "checkpoint_version": self.checkpoint_version,
            "config": asdict(self.config),
            "runtime_identity": copy.deepcopy(self.runtime_identity),
            "context": self.context.state_dict(),
            "actor_context": (
                self.actor_context.state_dict()
                if self.actor_context is not None
                else None
            ),
            "actor": self.actor.state_dict(),
            "scorer": self.scorer.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "scorer_opt": self.scorer_opt.state_dict(),
            "total_updates": self.total_updates,
            "scorer_updates": self.scorer_updates,
            "actor_updates": self.actor_updates,
            "actor_context_initialized": self.actor_context_initialized,
            "success_memory": {
                "contexts": self.success_memory_contexts.detach().cpu(),
                "residuals": self.success_memory_residuals.detach().cpu(),
                "step_ids": self.success_memory_step_ids.detach().cpu(),
                "next_indices": self.success_memory_next_indices.detach().cpu(),
            },
            "snapshot_id": snapshot_id,
        }
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
        temporary.replace(path)
        self.snapshot_id = snapshot_id

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "ValueGuidedBanditAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != VALUE_GUIDED_BANDIT_CHECKPOINT_SCHEMA:
            raise ValueError("Checkpoint is not a maintained value-guided bandit")
        config_payload = dict(payload["config"])
        known = {field.name for field in fields(ValueGuidedBanditConfig)}
        unknown = sorted(set(config_payload) - known)
        if unknown:
            raise ValueError(f"Unknown value-bandit config fields: {unknown}")
        agent = cls(
            ValueGuidedBanditConfig(**config_payload),
            device=device,
            runtime_identity=payload.get("runtime_identity"),
        )
        agent.context.load_state_dict(payload["context"])
        if agent.actor_context is not None:
            actor_context_payload = payload.get("actor_context")
            if actor_context_payload is None:
                raise ValueError("Checkpoint is missing the trainable actor context")
            agent.actor_context.load_state_dict(actor_context_payload)
        agent.actor.load_state_dict(payload["actor"])
        agent.scorer.load_state_dict(payload["scorer"])
        agent.actor_opt.load_state_dict(payload["actor_opt"])
        agent.scorer_opt.load_state_dict(payload["scorer_opt"])
        agent.total_updates = int(payload.get("total_updates", 0))
        agent.scorer_updates = int(payload.get("scorer_updates", 0))
        agent.actor_updates = int(payload.get("actor_updates", 0))
        if agent.actor_context is not None:
            if "actor_context_initialized" not in payload:
                raise ValueError("Checkpoint is missing actor-context initialization state")
            agent.actor_context_initialized = bool(payload["actor_context_initialized"])
        memory = payload.get("success_memory")
        if memory is not None:
            agent.success_memory_contexts = memory["contexts"].to(agent.device)
            agent.success_memory_residuals = memory["residuals"].to(agent.device)
            agent.success_memory_step_ids = memory["step_ids"].to(agent.device)
            agent.success_memory_next_indices = memory.get(
                "next_indices",
                torch.full_like(memory["step_ids"], -1),
            ).to(agent.device)
        agent.snapshot_id = payload.get("snapshot_id")
        return agent
