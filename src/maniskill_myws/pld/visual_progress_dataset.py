from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from .h5_replay import _h5_get, _prepare_image, _trajectory_success


@dataclass(frozen=True)
class VisualEpisode:
    file_index: int
    group_name: str
    length: int
    success: bool


def scan_visual_episodes(files: Sequence[Path]) -> list[VisualEpisode]:
    """Scan trajectory metadata without reading robot/environment state."""
    try:
        import h5py
    except Exception as e:  # pragma: no cover
        raise RuntimeError("h5py is required for visual progress training") from e

    episodes: list[VisualEpisode] = []
    for file_index, path in enumerate(files):
        with h5py.File(path, "r") as h5_file:
            names = sorted(name for name in h5_file if name.startswith("traj_"))
            if not names:
                names = [""]
            for name in names:
                group = h5_file[name] if name else h5_file
                if "actions" not in group:
                    continue
                # RecordEpisode stores T actions and T+1 observations.
                length = int(group["actions"].shape[0])
                if length <= 0:
                    continue
                episodes.append(
                    VisualEpisode(
                        file_index=file_index,
                        group_name=name,
                        length=length,
                        success=_trajectory_success(group),
                    )
                )
    if not episodes:
        raise ValueError("No non-empty trajectories found for visual progress training")
    return episodes


def split_visual_episodes(
    episodes: Sequence[VisualEpisode],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[VisualEpisode], list[VisualEpisode]]:
    """Stratified episode-level split; frames from one trial never leak across splits."""
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    train: list[VisualEpisode] = []
    validation: list[VisualEpisode] = []
    for success in (False, True):
        group = [episode for episode in episodes if episode.success is success]
        rng.shuffle(group)
        if fraction <= 0.0 or len(group) <= 1:
            n_validation = 0
        else:
            n_validation = max(1, int(round(len(group) * fraction)))
            n_validation = min(n_validation, len(group) - 1)
        validation.extend(group[:n_validation])
        train.extend(group[n_validation:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if not train:
        raise ValueError("Episode split produced an empty visual-progress training set")
    return train, validation


class H5VisualProgressPairDataset(Dataset):
    """Pure-RGB sequence pairs for self-supervised task-progress learning.

    Each item contains two successful sequences for temporal cycle consistency
    and one failed sequence for trajectory-level preference/outcome learning.
    No proprioception, object pose, contact, reward, or dense environment signal
    is returned by this dataset.
    """

    def __init__(
        self,
        files: Sequence[Path],
        episodes: Sequence[VisualEpisode],
        *,
        image_keys: Sequence[str],
        image_size: int,
        context_frames: int,
        context_stride: int,
        sequence_points: int,
        samples_per_epoch: int,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if not image_keys:
            raise ValueError("At least one RGB image key is required")
        if context_frames <= 0 or context_stride <= 0:
            raise ValueError("context_frames and context_stride must be positive")
        if sequence_points < 3:
            raise ValueError("sequence_points must be at least 3")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.files = [Path(path) for path in files]
        self.episodes = list(episodes)
        self.success_episodes = [episode for episode in episodes if episode.success]
        self.failure_episodes = [episode for episode in episodes if not episode.success]
        if len(self.success_episodes) < 2:
            raise ValueError("Visual alignment requires at least two successful episodes")
        if not self.failure_episodes:
            raise ValueError("Visual outcome calibration requires at least one failed episode")
        self.image_keys = list(image_keys)
        self.image_size = int(image_size)
        self.context_frames = int(context_frames)
        self.context_stride = int(context_stride)
        self.sequence_points = int(sequence_points)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

    def _group(self, episode: VisualEpisode):
        try:
            import h5py
        except Exception as e:  # pragma: no cover
            raise RuntimeError("h5py is required for visual progress training") from e
        handle = self._handles.get(episode.file_index)
        if handle is None:
            # A larger raw-chunk cache matters for gzip-compressed RGB datasets.
            handle = h5py.File(
                self.files[episode.file_index],
                "r",
                rdcc_nbytes=128 * 1024 * 1024,
                rdcc_nslots=100_003,
            )
            self._handles[episode.file_index] = handle
        return handle[episode.group_name] if episode.group_name else handle

    def _sample_points(
        self,
        episode: VisualEpisode,
        rng: np.random.Generator,
    ) -> np.ndarray:
        count = self.sequence_points
        if self.deterministic:
            points = np.linspace(0, episode.length, count).round().astype(np.int64)
        else:
            # Preserve exact start/end anchors and jitter only interior strata.
            edges = np.linspace(0, episode.length + 1, count + 1)
            points = np.empty(count, dtype=np.int64)
            points[0] = 0
            points[-1] = episode.length
            for i in range(1, count - 1):
                low = max(1, int(np.floor(edges[i])))
                high = min(episode.length, int(np.ceil(edges[i + 1])))
                points[i] = low if high <= low else int(rng.integers(low, high))
            points = np.maximum.accumulate(points)
        return np.clip(points, 0, episode.length)

    def _read_sequence(
        self,
        episode: VisualEpisode,
        points: np.ndarray,
    ) -> np.ndarray:
        group = self._group(episode)
        offsets = (
            np.arange(self.context_frames - 1, -1, -1, dtype=np.int64)
            * self.context_stride
        )
        context_indices = np.clip(points[:, None] - offsets[None, :], 0, episode.length)
        flat_indices = context_indices.reshape(-1)
        unique_indices, inverse = np.unique(flat_indices, return_inverse=True)

        views: list[np.ndarray] = []
        for key in self.image_keys:
            dataset = _h5_get(group, key)
            # One sorted HDF5 selection amortizes gzip chunk decompression across
            # nearby context/sequence frames.
            selected = np.asarray(dataset[unique_indices])
            prepared = np.stack(
                [_prepare_image(frame, self.image_size) for frame in selected], axis=0
            )
            gathered = prepared[inverse].reshape(
                self.sequence_points,
                self.context_frames,
                self.image_size,
                self.image_size,
                3,
            )
            views.append(gathered)
        # (sequence, context, views, H, W, C)
        return np.ascontiguousarray(np.stack(views, axis=2), dtype=np.uint8)

    @staticmethod
    def _choose_two_distinct(
        episodes: Sequence[VisualEpisode], rng: np.random.Generator
    ) -> tuple[VisualEpisode, VisualEpisode]:
        indices = rng.choice(len(episodes), size=2, replace=False)
        return episodes[int(indices[0])], episodes[int(indices[1])]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        # Index-derived RNG keeps validation reproducible across worker counts.
        rng = np.random.default_rng(self.seed + 1_000_003 * int(index))
        success_a, success_b = self._choose_two_distinct(self.success_episodes, rng)
        failure = self.failure_episodes[int(rng.integers(len(self.failure_episodes)))]

        points_a = self._sample_points(success_a, rng)
        points_b = self._sample_points(success_b, rng)
        points_f = self._sample_points(failure, rng)
        return {
            "success_a_images": self._read_sequence(success_a, points_a),
            "success_b_images": self._read_sequence(success_b, points_b),
            "failure_images": self._read_sequence(failure, points_f),
            # Fractions are diagnostics only; they are never used as progress
            # regression targets. They help detect accidental time shortcuts.
            "success_a_time": (points_a / max(1, success_a.length)).astype(np.float32),
            "success_b_time": (points_b / max(1, success_b.length)).astype(np.float32),
            "failure_time": (points_f / max(1, failure.length)).astype(np.float32),
        }
