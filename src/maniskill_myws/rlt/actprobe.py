"""Action-space rollout protocol for the official ActProbe model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np


ACTPROBE_ROLLOUT_SCHEMA = "maniskill_actprobe_action_rollouts_v1"
ACTPROBE_FEATURE_PROTOCOL = "official_actprobe_acm_tce_raw_action_v1"
ACTPROBE_FEATURE_NAMES = ("action_norm", "chunk_mse")
ACTPROBE_ACTION_SOURCE = "raw_openpi_emitted_prediction_before_env_projection"
ACTPROBE_LABEL_SOURCE = "episode_environment_success_any_step"


def _action_prediction(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [horizon, action_dim]")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def compute_actprobe_features(
    previous_prediction: object | None,
    current_prediction: object,
    *,
    executed_steps: int,
) -> np.ndarray:
    """Compute official ACM and TCE from consecutive emitted action chunks."""

    if executed_steps <= 0:
        raise ValueError("ActProbe executed_steps must be positive")
    current = _action_prediction(current_prediction, name="Current action prediction")
    if current.shape[0] < executed_steps:
        raise ValueError("Current action prediction is shorter than executed_steps")

    executed = current[:executed_steps].astype(np.float64)
    action_norm = float(np.sqrt(np.mean(executed**2)))
    chunk_mse = 0.0
    if previous_prediction is not None:
        previous = _action_prediction(
            previous_prediction, name="Previous action prediction"
        )
        if previous.shape[1] != current.shape[1]:
            raise ValueError("Consecutive ActProbe action dimensions disagree")
        previous_overlap = previous[executed_steps:]
        current_overlap = current[:-executed_steps]
        overlap = min(len(previous_overlap), len(current_overlap))
        if overlap > 0:
            difference = (
                previous_overlap[:overlap].astype(np.float64)
                - current_overlap[:overlap].astype(np.float64)
            )
            chunk_mse = float(np.mean(difference**2))
    features = np.asarray([action_norm, chunk_mse], dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("ActProbe features contain NaN or Inf")
    return features


@dataclass(frozen=True)
class ActProbeEpisode:
    features: np.ndarray
    success: bool
    seed: int
    env_steps: int

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 2 or features.shape[1] != len(ACTPROBE_FEATURE_NAMES):
            raise ValueError("ActProbe episode features must have shape [chunks, 2]")
        if features.shape[0] <= 0:
            raise ValueError("ActProbe episode must contain at least one chunk")
        if features.dtype != np.dtype(np.float32):
            raise TypeError("ActProbe episode features must be float32")
        if not np.all(np.isfinite(features)):
            raise ValueError("ActProbe episode features contain NaN or Inf")
        if int(self.seed) < 0 or int(self.env_steps) <= 0:
            raise ValueError("ActProbe episode seed and env_steps are invalid")


@dataclass(frozen=True)
class ActProbeDataset:
    episodes: tuple[ActProbeEpisode, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("ActProbe rollout dataset is empty")
        expected = {
            "actprobe_feature_protocol": ACTPROBE_FEATURE_PROTOCOL,
            "actprobe_feature_names": list(ACTPROBE_FEATURE_NAMES),
            "actprobe_action_source": ACTPROBE_ACTION_SOURCE,
            "label_source": ACTPROBE_LABEL_SOURCE,
            "base_policy_only": True,
        }
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"ActProbe metadata does not match the protocol: {mismatches}"
            )
        chunk_len = int(self.metadata.get("chunk_len", 0))
        prediction_horizon = int(self.metadata.get("prediction_horizon", 0))
        action_dim = int(self.metadata.get("action_dim", 0))
        if chunk_len <= 0 or action_dim <= 0:
            raise ValueError("ActProbe chunk length or action dimension is invalid")
        if prediction_horizon <= chunk_len:
            raise ValueError(
                "ActProbe prediction horizon must exceed chunk_len for TCE overlap"
            )
        if any(
            len(episode.features)
            != (episode.env_steps + chunk_len - 1) // chunk_len
            for episode in self.episodes
        ):
            raise ValueError("ActProbe feature count does not match executed chunks")
        seeds = [episode.seed for episode in self.episodes]
        if len(seeds) != len(set(seeds)):
            raise ValueError("ActProbe episode seeds must be unique")


def save_actprobe_dataset(path: str | Path, dataset: ActProbeDataset) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    feature_parts = []
    for episode in dataset.episodes:
        feature_parts.append(episode.features)
        offsets.append(offsets[-1] + len(episode.features))
    payload = {
        "schema": np.asarray(ACTPROBE_ROLLOUT_SCHEMA),
        "metadata_json": np.asarray(
            json.dumps(
                dataset.metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        ),
        "features": np.concatenate(feature_parts, axis=0),
        "episode_offsets": np.asarray(offsets, dtype=np.int64),
        "successes": np.asarray(
            [episode.success for episode in dataset.episodes], dtype=np.bool_
        ),
        "seeds": np.asarray(
            [episode.seed for episode in dataset.episodes], dtype=np.int64
        ),
        "env_steps": np.asarray(
            [episode.env_steps for episode in dataset.episodes], dtype=np.int64
        ),
    }
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=destination.parent, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_actprobe_dataset(path: str | Path) -> ActProbeDataset:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        schema = str(np.asarray(payload["schema"]).item())
        if schema != ACTPROBE_ROLLOUT_SCHEMA:
            raise ValueError(f"Unsupported ActProbe rollout schema {schema!r}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        features = np.asarray(payload["features"])
        offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
        successes = np.asarray(payload["successes"], dtype=np.bool_)
        seeds = np.asarray(payload["seeds"], dtype=np.int64)
        env_steps = np.asarray(payload["env_steps"], dtype=np.int64)
    if not isinstance(metadata, dict):
        raise ValueError("ActProbe metadata must be a JSON object")
    episode_count = len(successes)
    if features.ndim != 2 or features.dtype != np.dtype(np.float32):
        raise ValueError("ActProbe feature array is invalid")
    if features.shape[1] != len(ACTPROBE_FEATURE_NAMES):
        raise ValueError("ActProbe feature dimension is invalid")
    if not np.all(np.isfinite(features)):
        raise ValueError("ActProbe feature array contains NaN or Inf")
    if offsets.shape != (episode_count + 1,):
        raise ValueError("ActProbe episode offsets have the wrong shape")
    if seeds.shape != successes.shape or env_steps.shape != successes.shape:
        raise ValueError("ActProbe episode metadata arrays disagree")
    if (
        offsets[0] != 0
        or offsets[-1] != len(features)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("ActProbe episode offsets are not strictly increasing")
    episodes = tuple(
        ActProbeEpisode(
            features=features[offsets[index] : offsets[index + 1]].copy(),
            success=bool(successes[index]),
            seed=int(seeds[index]),
            env_steps=int(env_steps[index]),
        )
        for index in range(episode_count)
    )
    return ActProbeDataset(episodes=episodes, metadata=dict(metadata))


def stratified_actprobe_split_indices(
    successes: Sequence[bool],
    *,
    seen_train_ratio: float = 0.75,
    unseen_episode_ratio: float = 0.2,
    seed: int,
) -> dict[str, list[int]]:
    """Match SAFE's episode-disjoint split for detector comparisons."""

    if not 0.0 < seen_train_ratio < 1.0:
        raise ValueError("ActProbe seen_train_ratio must lie in (0, 1)")
    if not 0.0 < unseen_episode_ratio < 1.0:
        raise ValueError("ActProbe unseen_episode_ratio must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    success_array = np.asarray(successes, dtype=np.bool_)
    splits = {"train": [], "val_seen": [], "val_unseen": []}
    for outcome in (False, True):
        indices = np.flatnonzero(success_array == outcome)
        if len(indices) < 3:
            raise ValueError(
                "ActProbe requires at least three success and three failure episodes"
            )
        rng.shuffle(indices)
        test_count = max(1, int(np.floor(len(indices) * unseen_episode_ratio)))
        seen_count = len(indices) - test_count
        train_count = max(1, int(np.floor(seen_count * seen_train_ratio)))
        calibration_count = seen_count - train_count
        if calibration_count < 1:
            train_count -= 1
            calibration_count = 1
        if train_count < 1:
            raise ValueError("ActProbe split leaves no training episodes")
        splits["train"].extend(int(value) for value in indices[:train_count])
        splits["val_seen"].extend(
            int(value)
            for value in indices[train_count : train_count + calibration_count]
        )
        splits["val_unseen"].extend(
            int(value) for value in indices[train_count + calibration_count :]
        )
    for indices in splits.values():
        rng.shuffle(indices)
    calibration_successes = sum(
        bool(success_array[index]) for index in splits["val_seen"]
    )
    if calibration_successes < 4:
        raise ValueError(
            "ActProbe 5% FPR calibration requires at least four successful "
            "val_seen episodes"
        )
    return splits
