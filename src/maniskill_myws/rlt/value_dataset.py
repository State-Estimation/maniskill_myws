"""HDF5 rollout data for the lightweight distributional V_base model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from .value_model import return_bin_target


VALUE_ROLLOUT_SCHEMA = "safe_visual_base_value_rollouts_v1"


@dataclass(frozen=True)
class ValueEpisodeInfo:
    name: str
    seed: int
    success: bool
    env_steps: int
    boundaries: int


def read_value_dataset_metadata(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as file:
        schema = str(file.attrs.get("schema", ""))
        if schema != VALUE_ROLLOUT_SCHEMA:
            raise ValueError(f"Unsupported value rollout schema {schema!r}")
        metadata = json.loads(str(file.attrs["metadata_json"]))
    if not isinstance(metadata, dict):
        raise ValueError("Value rollout metadata must be a JSON object")
    return metadata


def scan_value_episodes(path: str | Path) -> list[ValueEpisodeInfo]:
    import h5py

    episodes: list[ValueEpisodeInfo] = []
    with h5py.File(path, "r") as file:
        if str(file.attrs.get("schema", "")) != VALUE_ROLLOUT_SCHEMA:
            raise ValueError("Value rollout file has an unsupported schema")
        for name in sorted(file.keys()):
            group = file[name]
            boundaries = int(group["latents"].shape[0])
            if boundaries <= 0:
                raise ValueError(f"Value rollout episode {name} is empty")
            expected = {
                "images": boundaries,
                "states": boundaries,
                "ref_chunks": boundaries,
                "step_ids": boundaries,
            }
            for key, count in expected.items():
                if int(group[key].shape[0]) != count:
                    raise ValueError(f"Value rollout {name}/{key} length mismatch")
            episodes.append(
                ValueEpisodeInfo(
                    name=name,
                    seed=int(group.attrs["seed"]),
                    success=bool(group.attrs["success"]),
                    env_steps=int(group.attrs["env_steps"]),
                    boundaries=boundaries,
                )
            )
    if not episodes:
        raise ValueError("Value rollout dataset is empty")
    seeds = [episode.seed for episode in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Value rollout episode seeds must be unique")
    return episodes


def stratified_value_split(
    episodes: Sequence[ValueEpisodeInfo],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ValueEpisodeInfo], list[ValueEpisodeInfo]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    train: list[ValueEpisodeInfo] = []
    validation: list[ValueEpisodeInfo] = []
    for outcome in (False, True):
        group = [episode for episode in episodes if episode.success is outcome]
        if len(group) < 2:
            raise ValueError(
                "Value training requires at least two success and two failure episodes"
            )
        rng.shuffle(group)
        count = max(1, int(round(len(group) * validation_fraction)))
        count = min(count, len(group) - 1)
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


class ValueBoundaryDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        episodes: Sequence[ValueEpisodeInfo],
        *,
        max_remaining_chunks: int,
    ) -> None:
        self.path = Path(path)
        self.episodes = list(episodes)
        self.max_remaining_chunks = int(max_remaining_chunks)
        self._index = [
            (episode_index, boundary_index)
            for episode_index, episode in enumerate(self.episodes)
            for boundary_index in range(episode.boundaries)
        ]
        self._handle = None

    def __len__(self) -> int:
        return len(self._index)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _file(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r", rdcc_nbytes=128 * 1024 * 1024)
        return self._handle

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        episode_index, boundary_index = self._index[index]
        episode = self.episodes[episode_index]
        group = self._file()[episode.name]
        target = return_bin_target(
            success=episode.success,
            boundary_index=boundary_index,
            boundary_count=episode.boundaries,
            max_remaining_chunks=self.max_remaining_chunks,
        )
        return {
            "images": np.asarray(group["images"][boundary_index], dtype=np.uint8),
            "state": np.asarray(group["states"][boundary_index], dtype=np.float32),
            "latent": np.asarray(group["latents"][boundary_index], dtype=np.float32),
            "ref_chunk": np.asarray(
                group["ref_chunks"][boundary_index], dtype=np.float32
            ),
            "step_id": np.int64(group["step_ids"][boundary_index]),
            "target": np.int64(target),
            "episode_index": np.int64(episode_index),
            "boundary_index": np.int64(boundary_index),
            "success": np.bool_(episode.success),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()
