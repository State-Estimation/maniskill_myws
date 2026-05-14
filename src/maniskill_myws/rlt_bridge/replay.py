from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clients import RLTFeatures, SOURCE_BASE
from .policy import RLTPlan


@dataclass
class ChunkTransitionBuilder:
    """Build one openpi-RLT replay transition from an executed action chunk."""

    chunk_len: int = 10
    action_dim: int = 7
    collection_phase: str = "online"
    plan: RLTPlan | None = None
    actions: list[np.ndarray] = field(default_factory=list)
    ref_actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    source_chunk: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.plan = None
        self.actions.clear()
        self.ref_actions.clear()
        self.rewards.clear()
        self.source_chunk.clear()

    def begin(self, plan: RLTPlan) -> None:
        self.reset()
        self.plan = plan

    def append_step(
        self,
        *,
        action: np.ndarray,
        ref_action: np.ndarray,
        reward: float,
        source: int = SOURCE_BASE,
    ) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32)[: self.action_dim])
        self.ref_actions.append(np.asarray(ref_action, dtype=np.float32)[: self.action_dim])
        self.rewards.append(float(reward))
        self.source_chunk.append(int(source))

    def ready(self) -> bool:
        return self.plan is not None and len(self.actions) >= self.chunk_len

    def to_transition(
        self,
        *,
        next_features: RLTFeatures,
        done: bool,
        success: bool | int = False,
        intervention_flag: bool = False,
    ) -> dict[str, Any]:
        if self.plan is None:
            raise RuntimeError("Cannot build transition before begin(plan).")

        action_chunk = _pad_rows(self.actions, self.chunk_len, self.action_dim)
        rewards = _pad_rewards(self.rewards, self.chunk_len)
        source_chunk = _pad_source(self.source_chunk, self.chunk_len, int(self.plan.actor_result.source))
        source = _collapse_source(source_chunk)

        return {
            "z_rl": np.asarray(self.plan.features.z_rl, dtype=np.float16),
            "proprio": np.asarray(self.plan.features.proprio, dtype=np.float32),
            "ref_chunk": np.asarray(self.plan.features.ref_chunk, dtype=np.float16),
            "action_chunk": np.asarray(action_chunk, dtype=np.float16),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "done": bool(done),
            "next_z_rl": np.asarray(next_features.z_rl, dtype=np.float16),
            "next_proprio": np.asarray(next_features.proprio, dtype=np.float32),
            "next_ref_chunk": np.asarray(next_features.ref_chunk, dtype=np.float16),
            "source": np.asarray(source, dtype=np.uint8),
            "source_chunk": np.asarray(source_chunk, dtype=np.uint8),
            "collection_phase": self.collection_phase,
            "success": np.asarray(int(bool(success)), dtype=np.int8),
            "intervention_flag": np.asarray(bool(intervention_flag), dtype=np.bool_),
            "episode_id": np.asarray(self.plan.episode_id, dtype=np.int32),
            "step_id": np.asarray(self.plan.step_id, dtype=np.int32),
        }


def _pad_rows(values: list[np.ndarray], chunk_len: int, action_dim: int) -> np.ndarray:
    out = np.zeros((chunk_len, action_dim), dtype=np.float32)
    if not values:
        return out
    rows = np.stack(values[:chunk_len], axis=0).astype(np.float32, copy=False)
    out[: rows.shape[0], : rows.shape[1]] = rows[:, :action_dim]
    if rows.shape[0] < chunk_len:
        out[rows.shape[0] :] = out[rows.shape[0] - 1]
    return out


def _pad_rewards(values: list[float], chunk_len: int) -> np.ndarray:
    out = np.zeros((chunk_len,), dtype=np.float32)
    if not values:
        return out
    rewards = np.asarray(values[:chunk_len], dtype=np.float32)
    out[: rewards.shape[0]] = rewards
    return out


def _pad_source(values: list[int], chunk_len: int, fallback: int) -> np.ndarray:
    out = np.full((chunk_len,), int(fallback), dtype=np.uint8)
    if not values:
        return out
    arr = np.asarray(values[:chunk_len], dtype=np.uint8)
    out[: arr.shape[0]] = arr
    return out


def _collapse_source(source_chunk: np.ndarray) -> int:
    unique = set(int(x) for x in np.asarray(source_chunk).reshape(-1))
    if len(unique) == 1:
        return next(iter(unique))
    return 3
