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
    mc_returns: np.ndarray
    images: np.ndarray | None = None
    next_images: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])

    @property
    def image_shape(self) -> tuple[int, ...] | None:
        if self.images is None:
            return None
        return tuple(int(x) for x in self.images.shape[1:])


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
        arr = np.asarray(_h5_get(traj_group, key)[t])
        parts.append(arr.astype(np.float32, copy=False).reshape(-1))
    if not parts:
        raise ValueError("state_keys is empty")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _resize_hwc_nearest(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return image
    size = int(size)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, size).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def _prepare_image(image: np.ndarray, image_size: int | None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {arr.shape}")
    if arr.shape[-1] not in {1, 3, 4} and arr.shape[0] in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    arr = _resize_hwc_nearest(arr, image_size)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _build_images(
    traj_group: Any, t: int, image_keys: Sequence[str], image_size: int | None
) -> np.ndarray:
    images = [_prepare_image(np.asarray(_h5_get(traj_group, key)[t]), image_size) for key in image_keys]
    return np.stack(images, axis=0)


def _trajectory_success(traj_group: Any) -> bool:
    if "success" not in traj_group:
        return True
    success = np.asarray(traj_group["success"], dtype=bool)
    return bool(success[-1]) if success.size else False


def _discounted_return_to_go(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + float(gamma) * running * (1.0 - float(dones[i]))
        returns[i] = running
    return returns


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
    mc_return_gamma: float = 0.99,
    image_keys: Sequence[str] | None = None,
    image_size: int | None = None,
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
    mc_return_rows: list[np.ndarray] = []
    image_rows: list[np.ndarray] = []
    next_image_rows: list[np.ndarray] = []
    use_images = bool(image_keys)
    loaded_transitions = 0

    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            traj_names = sorted(k for k in f.keys() if k.startswith("traj_"))
            groups = [f[k] for k in traj_names] if traj_names else [f]
            for g in groups:
                if max_transitions is not None and loaded_transitions >= max_transitions:
                    break
                if success_only and not _trajectory_success(g):
                    continue

                actions = np.asarray(_h5_get(g, actions_key), dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[:, None]
                original_t_count = int(actions.shape[0])
                t_count = original_t_count
                if t_count <= 0:
                    continue
                if max_transitions is not None:
                    t_count = min(t_count, int(max_transitions) - loaded_transitions)
                    actions = actions[:t_count]
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
                if (
                    terminal_success_reward
                    and "success" in g
                    and rewards.size
                    and t_count == original_t_count
                ):
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
                if use_images:
                    image_seq = np.stack(
                        [_build_images(g, t, image_keys or [], image_size) for t in range(t_count + 1)],
                        axis=0,
                    )
                    image_rows.append(image_seq[:-1])
                    next_image_rows.append(image_seq[1:])
                action_rows.append(actions)
                base_action_rows.append(base_actions)
                next_base_action_rows.append(next_base_actions)
                reward_rows.append(rewards.astype(np.float32, copy=False))
                done_rows.append(dones.astype(np.float32, copy=False))
                mc_return_rows.append(
                    _discounted_return_to_go(rewards, dones, mc_return_gamma)
                )
                loaded_transitions += t_count
        if max_transitions is not None and loaded_transitions >= max_transitions:
            break

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
        mc_returns=np.concatenate(mc_return_rows, axis=0),
        images=np.concatenate(image_rows, axis=0) if image_rows else None,
        next_images=np.concatenate(next_image_rows, axis=0) if next_image_rows else None,
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
            mc_returns=data.mc_returns[idx],
            images=data.images[idx] if data.images is not None else None,
            next_images=data.next_images[idx] if data.next_images is not None else None,
        )
    return data
