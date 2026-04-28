from __future__ import annotations

from dataclasses import dataclass

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


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
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
        self.dones[self.pos] = float(done)
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
        )


def concat_batches(parts: list[ReplayBatch]) -> ReplayBatch:
    return ReplayBatch(
        obs=np.concatenate([p.obs for p in parts], axis=0),
        actions=np.concatenate([p.actions for p in parts], axis=0),
        base_actions=np.concatenate([p.base_actions for p in parts], axis=0),
        rewards=np.concatenate([p.rewards for p in parts], axis=0),
        next_obs=np.concatenate([p.next_obs for p in parts], axis=0),
        next_base_actions=np.concatenate([p.next_base_actions for p in parts], axis=0),
        dones=np.concatenate([p.dones for p in parts], axis=0),
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
