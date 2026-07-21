from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .replay import ChunkReplayBuffer, ChunkTransition, TransitionSource, pad_or_trim_chunk
from .state import prepare_rgb_image


@dataclass(frozen=True, slots=True)
class RolloutLoadStats:
    files: int
    episodes: int
    transitions: int
    env_steps: int


def _trajectory_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("traj_")
    try:
        return int(suffix), name
    except ValueError:
        return 2**63 - 1, name


def _obs_path(key: str) -> str:
    normalized = str(key).strip("/")
    return normalized if normalized.startswith("obs/") else f"obs/{normalized}"


def _read_state(trajectory: Any, state_keys: Sequence[str], index: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in state_keys:
        path = _obs_path(key)
        if path not in trajectory:
            raise KeyError(f"Trajectory is missing state key '{path}'")
        parts.append(np.asarray(trajectory[path][index], dtype=np.float32).reshape(-1))
    if not parts:
        raise ValueError("state_keys must not be empty")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _read_images(
    trajectory: Any,
    image_keys: Sequence[str],
    index: int,
    image_size: int | None,
) -> np.ndarray:
    images: list[np.ndarray] = []
    for key in image_keys:
        path = _obs_path(key)
        if path not in trajectory:
            raise KeyError(f"Trajectory is missing image key '{path}'")
        images.append(prepare_rgb_image(trajectory[path][index], image_size))
    if not images:
        raise ValueError("image_keys must not be empty when visual observations are requested")
    return np.stack(images, axis=0)


def _load_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in rollout metadata: {metadata_path}")
    return metadata


def _validate_metadata(
    path: Path,
    metadata: dict[str, Any] | None,
    *,
    expected_env_id: str | None,
    expected_control_mode: str | None,
    expected_reward_mode: str | None,
) -> None:
    if metadata is None:
        return
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info, dict) else {}
    checks = (
        ("env_id", env_info.get("env_id"), expected_env_id),
        ("control_mode", env_kwargs.get("control_mode"), expected_control_mode),
        ("reward_mode", env_kwargs.get("reward_mode"), expected_reward_mode),
    )
    for field, actual, expected in checks:
        if expected is not None and actual is not None and str(actual) != str(expected):
            raise ValueError(
                f"Rollout metadata mismatch for {path}: {field}={actual!r}, "
                f"expected {expected!r}"
            )


def load_rollout_h5_into_replay(
    buffer: ChunkReplayBuffer,
    paths: Sequence[str | Path],
    *,
    state_keys: Sequence[str],
    image_keys: Sequence[str] | None = None,
    image_size: int | None = None,
    action_key: str = "actions",
    reward_key: str = "rewards",
    max_transitions: int | None = None,
    expected_env_id: str | None = None,
    expected_control_mode: str | None = None,
    expected_reward_mode: str | None = None,
    validate_metadata: bool = True,
) -> RolloutLoadStats:
    """Load ManiSkill RecordEpisode H5 trajectories as Base-policy chunks.

    Contiguous executed Base actions are used as both the reference and action
    chunks. Episode-final partial chunks are padded by ChunkReplayBuffer.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("Loading rollout warmup data requires h5py") from error

    rollout_paths = [Path(path).expanduser().resolve() for path in paths]
    if not rollout_paths:
        return RolloutLoadStats(files=0, episodes=0, transitions=0, env_steps=0)
    if max_transitions is not None and max_transitions < 0:
        raise ValueError("max_transitions must be non-negative")
    transition_limit = buffer.capacity if max_transitions is None else int(max_transitions)
    if transition_limit > buffer.capacity:
        raise ValueError(
            f"Requested {transition_limit} rollout transitions, but buffer capacity is "
            f"{buffer.capacity}"
        )
    if len(buffer) != 0:
        raise ValueError("Rollout prefill currently requires an empty replay buffer")

    action_path = str(action_key).strip("/")
    reward_path = str(reward_key).strip("/")
    transitions = 0
    episodes = 0
    env_steps = 0
    files_loaded = 0

    for path in rollout_paths:
        if transitions >= transition_limit:
            break
        if not path.is_file():
            raise FileNotFoundError(f"Rollout H5 not found: {path}")
        if validate_metadata:
            _validate_metadata(
                path,
                _load_metadata(path),
                expected_env_id=expected_env_id,
                expected_control_mode=expected_control_mode,
                expected_reward_mode=expected_reward_mode,
            )

        with h5py.File(path, "r") as h5_file:
            trajectory_names = sorted(
                (name for name in h5_file if name.startswith("traj_")),
                key=_trajectory_sort_key,
            )
            if not trajectory_names:
                raise ValueError(f"Rollout H5 contains no traj_* groups: {path}")
            files_loaded += 1

            for trajectory_name in trajectory_names:
                if transitions >= transition_limit:
                    break
                trajectory = h5_file[trajectory_name]
                if action_path not in trajectory:
                    raise KeyError(
                        f"Trajectory '{trajectory_name}' in {path} is missing "
                        f"action key '{action_path}'"
                    )
                if reward_path not in trajectory:
                    raise KeyError(
                        f"Trajectory '{trajectory_name}' in {path} is missing "
                        f"reward key '{reward_path}'"
                    )

                actions = np.asarray(trajectory[action_path], dtype=np.float32)
                rewards = np.asarray(trajectory[reward_path], dtype=np.float32).reshape(-1)
                if actions.ndim != 2 or actions.shape[1] < buffer.action_dim:
                    raise ValueError(
                        f"Expected actions [T, >={buffer.action_dim}] in "
                        f"{path}:{trajectory_name}, got {actions.shape}"
                    )
                trajectory_steps = int(actions.shape[0])
                if rewards.shape[0] != trajectory_steps:
                    raise ValueError(
                        f"Reward/action length mismatch in {path}:{trajectory_name}: "
                        f"{rewards.shape[0]} != {trajectory_steps}"
                    )
                if trajectory_steps == 0:
                    continue

                terminal = np.zeros((trajectory_steps,), dtype=bool)
                for done_key in ("terminated", "truncated"):
                    if done_key in trajectory:
                        done_values = np.asarray(trajectory[done_key], dtype=bool).reshape(-1)
                        if done_values.shape[0] != trajectory_steps:
                            raise ValueError(
                                f"{done_key}/action length mismatch in "
                                f"{path}:{trajectory_name}"
                            )
                        terminal |= done_values
                success_values = (
                    np.asarray(trajectory["success"], dtype=bool).reshape(-1)
                    if "success" in trajectory
                    else np.zeros((trajectory_steps,), dtype=bool)
                )
                if success_values.shape[0] != trajectory_steps:
                    raise ValueError(
                        f"success/action length mismatch in {path}:{trajectory_name}"
                    )

                episode_id = episodes
                episodes += 1
                for start in range(0, trajectory_steps, buffer.chunk_len):
                    if transitions >= transition_limit:
                        break
                    end = min(start + buffer.chunk_len, trajectory_steps)
                    ref_chunk = actions[start:end, : buffer.action_dim]
                    next_ref_chunk = (
                        np.zeros((buffer.chunk_len, buffer.action_dim), dtype=np.float32)
                        if end >= trajectory_steps or bool(np.any(terminal[start:end]))
                        else pad_or_trim_chunk(
                            actions[end : end + buffer.chunk_len],
                            chunk_len=buffer.chunk_len,
                            action_dim=buffer.action_dim,
                        )
                    )
                    obs = _read_state(trajectory, state_keys, start)
                    next_obs = _read_state(trajectory, state_keys, end)
                    if obs.shape != (buffer.state_dim,) or next_obs.shape != (
                        buffer.state_dim,
                    ):
                        raise ValueError(
                            f"State dimension mismatch in {path}:{trajectory_name}: "
                            f"got {obs.shape}/{next_obs.shape}, expected "
                            f"({buffer.state_dim},)"
                        )

                    images = None
                    next_images = None
                    if buffer.image_shape is not None:
                        if image_keys is None:
                            raise ValueError(
                                "Visual replay buffer requires rollout image_keys"
                            )
                        images = _read_images(trajectory, image_keys, start, image_size)
                        next_images = _read_images(trajectory, image_keys, end, image_size)
                        if tuple(images.shape) != buffer.image_shape:
                            raise ValueError(
                                f"Image shape mismatch in {path}:{trajectory_name}: "
                                f"got {tuple(images.shape)}, expected {buffer.image_shape}"
                            )

                    buffer.add(
                        ChunkTransition(
                            obs=obs,
                            ref_chunk=ref_chunk,
                            action_chunk=ref_chunk,
                            rewards=rewards[start:end],
                            done=bool(np.any(terminal[start:end])) or end >= trajectory_steps,
                            next_obs=next_obs,
                            next_ref_chunk=next_ref_chunk,
                            images=images,
                            next_images=next_images,
                            source=int(TransitionSource.BASE),
                            source_chunk=np.full(
                                (buffer.chunk_len,),
                                int(TransitionSource.BASE),
                                dtype=np.uint8,
                            ),
                            episode_id=episode_id,
                            step_id=start,
                            success=int(np.any(success_values[start:end])),
                        )
                    )
                    transitions += 1
                    env_steps += end - start

    return RolloutLoadStats(
        files=files_loaded,
        episodes=episodes,
        transitions=transitions,
        env_steps=env_steps,
    )
