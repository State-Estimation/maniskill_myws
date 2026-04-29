from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .h5_replay import OfflineReplayData


@dataclass
class ReplayBatch:
    obs: np.ndarray
    actions: np.ndarray
    base_actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_base_actions: np.ndarray
    dones: np.ndarray
    mc_returns: np.ndarray
    images: np.ndarray | None = None
    next_images: np.ndarray | None = None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        image_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.image_shape = tuple(int(x) for x in image_shape) if image_shape is not None else None
        self.obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.mc_returns = np.zeros((capacity,), dtype=np.float32)
        self.images = (
            np.zeros((capacity, *self.image_shape), dtype=np.uint8)
            if self.image_shape is not None
            else None
        )
        self.next_images = (
            np.zeros((capacity, *self.image_shape), dtype=np.uint8)
            if self.image_shape is not None
            else None
        )
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        base_action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_base_action: np.ndarray,
        done: bool,
        mc_return: float | None = None,
        images: np.ndarray | None = None,
        next_images: np.ndarray | None = None,
    ) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32).reshape(self.state_dim)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.base_actions[self.pos] = np.asarray(base_action, dtype=np.float32).reshape(
            self.action_dim
        )
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(self.state_dim)
        self.next_base_actions[self.pos] = np.asarray(next_base_action, dtype=np.float32).reshape(
            self.action_dim
        )
        if self.image_shape is not None:
            if images is None or next_images is None:
                raise ValueError("Visual ReplayBuffer requires images and next_images")
            assert self.images is not None and self.next_images is not None
            self.images[self.pos] = np.asarray(images, dtype=np.uint8).reshape(self.image_shape)
            self.next_images[self.pos] = np.asarray(next_images, dtype=np.uint8).reshape(
                self.image_shape
            )
        self.dones[self.pos] = float(done)
        self.mc_returns[self.pos] = float(reward if mc_return is None else mc_return)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def add_offline_data(self, data: OfflineReplayData) -> None:
        for i in range(data.size):
            self.add(
                data.obs[i],
                data.actions[i],
                data.base_actions[i],
                float(data.rewards[i]),
                data.next_obs[i],
                data.next_base_actions[i],
                bool(data.dones[i]),
                float(data.mc_returns[i]),
                images=data.images[i] if data.images is not None else None,
                next_images=data.next_images[i] if data.next_images is not None else None,
            )

    def sample(self, batch_size: int) -> ReplayBatch:
        size = len(self)
        if size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = np.random.randint(0, size, size=int(batch_size))
        return ReplayBatch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            base_actions=self.base_actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            next_base_actions=self.next_base_actions[idx],
            dones=self.dones[idx],
            mc_returns=self.mc_returns[idx],
            images=self.images[idx] if self.images is not None else None,
            next_images=self.next_images[idx] if self.next_images is not None else None,
        )

    def save(self, path: str | Path, **metadata) -> None:
        size = len(self)
        payload: dict[str, np.ndarray] = dict(
            size=np.asarray(size, dtype=np.int64),
            capacity=np.asarray(self.capacity, dtype=np.int64),
            state_dim=np.asarray(self.state_dim, dtype=np.int64),
            action_dim=np.asarray(self.action_dim, dtype=np.int64),
            image_shape=np.asarray(self.image_shape or (), dtype=np.int64),
            obs=self.obs[:size].copy(),
            actions=self.actions[:size].copy(),
            base_actions=self.base_actions[:size].copy(),
            rewards=self.rewards[:size].copy(),
            next_obs=self.next_obs[:size].copy(),
            next_base_actions=self.next_base_actions[:size].copy(),
            dones=self.dones[:size].copy(),
            mc_returns=self.mc_returns[:size].copy(),
        )
        if self.images is not None and self.next_images is not None:
            payload["images"] = self.images[:size].copy()
            payload["next_images"] = self.next_images[:size].copy()
        for key, value in metadata.items():
            payload[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(str(path), **payload)

    def load(self, path: str | Path) -> dict[str, object]:
        data = np.load(str(path), allow_pickle=False)
        size = int(np.asarray(data["size"]).item())
        state_dim = int(np.asarray(data["state_dim"]).item())
        action_dim = int(np.asarray(data["action_dim"]).item())
        image_shape_arr = np.asarray(data["image_shape"]) if "image_shape" in data.files else np.asarray(())
        image_shape = tuple(int(x) for x in image_shape_arr.reshape(-1))
        if state_dim != self.state_dim or action_dim != self.action_dim:
            raise ValueError(
                f"Buffer shape mismatch: file ({state_dim}, {action_dim}) "
                f"!= current ({self.state_dim}, {self.action_dim})"
            )
        if image_shape != (self.image_shape or ()):
            raise ValueError(
                f"Buffer image shape mismatch: file {image_shape or None} "
                f"!= current {self.image_shape}"
            )
        if size > self.capacity:
            raise ValueError(
                f"Saved buffer size {size} exceeds current capacity {self.capacity}"
            )
        self.pos = 0
        self.full = False
        self.obs[:size] = data["obs"]
        self.actions[:size] = data["actions"]
        self.base_actions[:size] = data["base_actions"]
        self.rewards[:size] = data["rewards"]
        self.next_obs[:size] = data["next_obs"]
        self.next_base_actions[:size] = data["next_base_actions"]
        self.dones[:size] = data["dones"]
        if "mc_returns" in data.files:
            self.mc_returns[:size] = data["mc_returns"]
        else:
            self.mc_returns[:size] = 0.0
        if self.images is not None and self.next_images is not None:
            if "images" not in data.files or "next_images" not in data.files:
                raise ValueError("Saved buffer does not contain visual observations")
            self.images[:size] = data["images"]
            self.next_images[:size] = data["next_images"]
        self.pos = size % self.capacity
        self.full = size == self.capacity
        meta: dict[str, object] = {}
        for key in data.files:
            if not key.startswith("meta_"):
                continue
            value = np.asarray(data[key])
            meta[key[5:]] = value.item() if value.shape == () else value
        return meta


def concat_batches(parts: list[ReplayBatch]) -> ReplayBatch:
    return ReplayBatch(
        obs=np.concatenate([p.obs for p in parts], axis=0),
        actions=np.concatenate([p.actions for p in parts], axis=0),
        base_actions=np.concatenate([p.base_actions for p in parts], axis=0),
        rewards=np.concatenate([p.rewards for p in parts], axis=0),
        next_obs=np.concatenate([p.next_obs for p in parts], axis=0),
        next_base_actions=np.concatenate([p.next_base_actions for p in parts], axis=0),
        dones=np.concatenate([p.dones for p in parts], axis=0),
        mc_returns=np.concatenate([p.mc_returns for p in parts], axis=0),
        images=(
            np.concatenate([p.images for p in parts if p.images is not None], axis=0)
            if all(p.images is not None for p in parts)
            else None
        ),
        next_images=(
            np.concatenate([p.next_images for p in parts if p.next_images is not None], axis=0)
            if all(p.next_images is not None for p in parts)
            else None
        ),
    )


def sample_offline_online(
    offline: ReplayBuffer,
    online: ReplayBuffer,
    batch_size: int,
    *,
    offline_fraction: float = 0.5,
) -> ReplayBatch:
    if len(online) == 0:
        return offline.sample(batch_size)
    n_offline = int(round(batch_size * offline_fraction))
    n_online = int(batch_size - n_offline)
    if batch_size > 1:
        n_offline = max(1, n_offline)
        n_online = max(1, n_online)
    else:
        n_offline = 1
        n_online = 0
    if n_online <= 0:
        return offline.sample(n_offline)
    return concat_batches([offline.sample(n_offline), online.sample(n_online)])
