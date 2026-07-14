from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import Any

import numpy as np


class TransitionSource(enum.IntEnum):
    BASE = 0
    RLT = 1
    HUMAN = 2
    MIXED = 3


@dataclass(slots=True)
class ChunkTransition:
    obs: np.ndarray
    ref_chunk: np.ndarray
    action_chunk: np.ndarray
    rewards: np.ndarray
    done: bool
    next_obs: np.ndarray
    next_ref_chunk: np.ndarray
    images: np.ndarray | None = None
    next_images: np.ndarray | None = None
    source: int = int(TransitionSource.RLT)
    source_chunk: np.ndarray | None = None
    episode_id: int = 0
    step_id: int = 0
    success: int = 0


@dataclass(slots=True)
class ChunkReplayBatch:
    obs: np.ndarray
    ref_chunks: np.ndarray
    action_chunks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_obs: np.ndarray
    next_ref_chunks: np.ndarray
    source_chunks: np.ndarray
    images: np.ndarray | None = None
    next_images: np.ndarray | None = None


def validate_pd_joint_pos_action_dim(action_dim: int) -> None:
    if int(action_dim) != 8:
        raise ValueError(
            f"ManiSkill RLT currently expects pd_joint_pos action_dim=8, got {action_dim}."
        )


def pad_or_trim_chunk(
    chunk: np.ndarray,
    *,
    chunk_len: int,
    action_dim: int,
    pad_mode: str = "last",
) -> np.ndarray:
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected action chunk [T, A], got {array.shape}")
    if array.shape[1] < action_dim:
        raise ValueError(f"Chunk action_dim {array.shape[1]} is smaller than required {action_dim}")
    array = array[:, :action_dim]
    if array.shape[0] >= chunk_len:
        return array[:chunk_len].astype(np.float32, copy=False)
    out = np.zeros((chunk_len, action_dim), dtype=np.float32)
    out[: array.shape[0]] = array
    if array.shape[0] > 0 and pad_mode == "last":
        out[array.shape[0] :] = array[-1]
    return out


def pad_rewards(rewards: list[float] | np.ndarray, *, chunk_len: int) -> np.ndarray:
    array = np.asarray(rewards, dtype=np.float32).reshape(-1)
    out = np.zeros((chunk_len,), dtype=np.float32)
    n = min(chunk_len, array.shape[0])
    if n > 0:
        out[:n] = array[:n]
    return out


class ChunkReplayBuffer:
    """CPU replay buffer for fixed-length RLT chunk transitions."""

    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int,
        action_dim: int = 8,
        chunk_len: int = 10,
        image_shape: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> None:
        validate_pd_joint_pos_action_dim(action_dim)
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.image_shape = tuple(int(x) for x in image_shape) if image_shape is not None else None
        self._rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.ref_chunks = np.zeros(
            (self.capacity, self.chunk_len, self.action_dim), dtype=np.float32
        )
        self.action_chunks = np.zeros(
            (self.capacity, self.chunk_len, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.chunk_len), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_ref_chunks = np.zeros(
            (self.capacity, self.chunk_len, self.action_dim), dtype=np.float32
        )
        self.source_chunks = np.zeros((self.capacity, self.chunk_len), dtype=np.uint8)
        self.episode_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.step_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.successes = np.zeros((self.capacity,), dtype=np.int8)
        self.images = (
            np.zeros((self.capacity, *self.image_shape), dtype=np.uint8)
            if self.image_shape is not None
            else None
        )
        self.next_images = (
            np.zeros((self.capacity, *self.image_shape), dtype=np.uint8)
            if self.image_shape is not None
            else None
        )
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, transition: ChunkTransition | dict[str, Any]) -> None:
        if isinstance(transition, dict):
            transition = ChunkTransition(**transition)
        idx = self.pos
        self.obs[idx] = np.asarray(transition.obs, dtype=np.float32).reshape(self.state_dim)
        self.ref_chunks[idx] = pad_or_trim_chunk(
            transition.ref_chunk, chunk_len=self.chunk_len, action_dim=self.action_dim
        )
        self.action_chunks[idx] = pad_or_trim_chunk(
            transition.action_chunk, chunk_len=self.chunk_len, action_dim=self.action_dim
        )
        self.rewards[idx] = pad_rewards(transition.rewards, chunk_len=self.chunk_len)
        self.dones[idx] = float(transition.done)
        self.next_obs[idx] = np.asarray(transition.next_obs, dtype=np.float32).reshape(
            self.state_dim
        )
        self.next_ref_chunks[idx] = pad_or_trim_chunk(
            transition.next_ref_chunk, chunk_len=self.chunk_len, action_dim=self.action_dim
        )
        if transition.source_chunk is None:
            self.source_chunks[idx] = np.full(
                (self.chunk_len,), int(transition.source), dtype=np.uint8
            )
        else:
            self.source_chunks[idx] = np.asarray(transition.source_chunk, dtype=np.uint8).reshape(
                self.chunk_len
            )
        self.episode_ids[idx] = int(transition.episode_id)
        self.step_ids[idx] = int(transition.step_id)
        self.successes[idx] = int(transition.success)
        if self.images is not None and self.next_images is not None:
            if transition.images is None or transition.next_images is None:
                raise ValueError("Visual ChunkReplayBuffer requires images and next_images")
            self.images[idx] = np.asarray(transition.images, dtype=np.uint8).reshape(
                self.image_shape
            )
            self.next_images[idx] = np.asarray(transition.next_images, dtype=np.uint8).reshape(
                self.image_shape
            )
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> ChunkReplayBatch:
        size = len(self)
        if size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, size, size=min(int(batch_size), size))
        return ChunkReplayBatch(
            obs=self.obs[idx],
            ref_chunks=self.ref_chunks[idx],
            action_chunks=self.action_chunks[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            next_obs=self.next_obs[idx],
            next_ref_chunks=self.next_ref_chunks[idx],
            source_chunks=self.source_chunks[idx],
            images=self.images[idx] if self.images is not None else None,
            next_images=self.next_images[idx] if self.next_images is not None else None,
        )
