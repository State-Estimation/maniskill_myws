from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import glob

import numpy as np


@dataclass(frozen=True)
class OfflineReplayData:
    obs: np.ndarray
    actions: np.ndarray
    base_actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_base_actions: np.ndarray
    dones: np.ndarray

    @property
    def state_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


def find_h5_files(*, h5_dir: str | None = None, h5_glob: str | None = None) -> list[Path]:
    if h5_glob:
        pattern = h5_glob
    elif h5_dir:
        pattern = str(Path(h5_dir).expanduser().resolve() / "**" / "*.h5")
    else:
        raise ValueError("Need one of h5_dir or h5_glob")
    return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]


def _h5_get(root: Any, path: str) -> Any:
    cur = root
    parts = [p for p in path.strip("/").split("/") if p]
    if parts and parts[0] not in cur and "obs" in cur:
        parts = ["obs", *parts]
    for part in parts:
        cur = cur[part]
    return cur


def _build_state(traj_group: Any, t: int, state_keys: Sequence[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in state_keys:
        arr = np.asarray(_h5_get(traj_group, key))[t]
        parts.append(arr.astype(np.float32, copy=False).reshape(-1))
    if not parts:
        raise ValueError("state_keys is empty")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _trajectory_success(traj_group: Any) -> bool:
    if "success" not in traj_group:
        return True
    success = np.asarray(traj_group["success"], dtype=bool)
    return bool(success[-1]) if success.size else False


def load_h5_replay(
    files: Sequence[Path],
    *,
    state_keys: Sequence[str],
    actions_key: str = "actions",
    reward_key: str = "rewards",
    success_only: bool = True,
    reward_from_success: bool = False,
    terminal_success_reward: bool = True,
    base_action_mode: str = "action",
    max_transitions: int | None = None,
) -> OfflineReplayData:
    """
    Load RecordEpisode trajectories into PLD replay arrays.

    Existing teleop/replayed trajectories usually do not store the frozen VLA
    action separately. For PLD warm-starting we default to treating the recorded
    action as the local base action, which gives the critic successful high-value
    actions while online training later supplies real base-policy actions.
    """
    try:
        import h5py
    except Exception as e:  # pragma: no cover
        raise RuntimeError("h5py is required to load ManiSkill trajectory files") from e

    if base_action_mode not in {"action", "zero"}:
        raise ValueError("base_action_mode must be 'action' or 'zero'")

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    base_action_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    next_base_action_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []

    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            traj_names = sorted(k for k in f.keys() if k.startswith("traj_"))
            groups = [f[k] for k in traj_names] if traj_names else [f]
            for g in groups:
                if success_only and not _trajectory_success(g):
                    continue

                actions = np.asarray(_h5_get(g, actions_key), dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[:, None]
                t_count = int(actions.shape[0])
                if t_count <= 0:
                    continue

                if reward_from_success and "success" in g:
                    rewards = np.asarray(g["success"], dtype=np.float32)
                elif reward_key and reward_key in g:
                    rewards = np.asarray(g[reward_key], dtype=np.float32)
                elif "success" in g:
                    rewards = np.asarray(g["success"], dtype=np.float32)
                else:
                    rewards = np.zeros((t_count,), dtype=np.float32)
                rewards = rewards.reshape(-1)[:t_count]
                if terminal_success_reward and "success" in g and rewards.size:
                    success = np.asarray(g["success"], dtype=bool)
                    if success.size and success[-1] and float(np.max(rewards)) <= 0.0:
                        rewards[-1] = 1.0

                terminated = (
                    np.asarray(g["terminated"], dtype=bool)
                    if "terminated" in g
                    else np.zeros((t_count,), dtype=bool)
                )
                truncated = (
                    np.asarray(g["truncated"], dtype=bool)
                    if "truncated" in g
                    else np.zeros((t_count,), dtype=bool)
                )
                dones = np.logical_or(terminated[:t_count], truncated[:t_count])

                if base_action_mode == "action":
                    base_actions = actions
                else:
                    base_actions = np.zeros_like(actions)
                next_base_actions = np.concatenate([base_actions[1:], base_actions[-1:]], axis=0)

                states = np.stack([_build_state(g, t, state_keys) for t in range(t_count + 1)])
                obs_rows.append(states[:-1])
                next_obs_rows.append(states[1:])
                action_rows.append(actions)
                base_action_rows.append(base_actions)
                next_base_action_rows.append(next_base_actions)
                reward_rows.append(rewards.astype(np.float32, copy=False))
                done_rows.append(dones.astype(np.float32, copy=False))

    if not obs_rows:
        raise ValueError("No transitions loaded from H5 files")

    data = OfflineReplayData(
        obs=np.concatenate(obs_rows, axis=0),
        actions=np.concatenate(action_rows, axis=0),
        base_actions=np.concatenate(base_action_rows, axis=0),
        rewards=np.concatenate(reward_rows, axis=0),
        next_obs=np.concatenate(next_obs_rows, axis=0),
        next_base_actions=np.concatenate(next_base_action_rows, axis=0),
        dones=np.concatenate(done_rows, axis=0),
    )
    if max_transitions is not None and data.size > max_transitions:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(data.size, size=max_transitions, replace=False))
        data = OfflineReplayData(
            obs=data.obs[idx],
            actions=data.actions[idx],
            base_actions=data.base_actions[idx],
            rewards=data.rewards[idx],
            next_obs=data.next_obs[idx],
            next_base_actions=data.next_base_actions[idx],
            dones=data.dones[idx],
        )
    return data
