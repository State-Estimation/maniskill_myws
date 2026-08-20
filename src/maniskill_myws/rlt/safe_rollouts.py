"""ManiSkill rollout storage and data adaptation for the official SAFE trainer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Sequence

import numpy as np
import torch

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIFFUSION_SELECTION,
    SAFE_LATENT_DIM,
    SAFE_LATENT_HORIZON_SELECTION,
    SAFE_LATENT_POOLING,
    SAFE_LATENT_PROTOCOL,
    SAFE_LATENT_SOURCE,
)


SAFE_ROLLOUT_SCHEMA = "maniskill_safe_pre_velocity_rollouts_v1"


@dataclass(frozen=True)
class SafeRolloutEpisode:
    latents: np.ndarray
    success: bool
    seed: int
    env_steps: int

    def __post_init__(self) -> None:
        latents = np.asarray(self.latents)
        if latents.ndim != 2 or latents.shape[0] <= 0:
            raise ValueError("SAFE episode latents must have shape [chunks, latent_dim]")
        if latents.shape[1] != SAFE_LATENT_DIM:
            raise ValueError(
                f"SAFE episode latent dimension {latents.shape[1]} != {SAFE_LATENT_DIM}"
            )
        if latents.dtype != np.dtype(np.float32):
            raise TypeError("SAFE episode latents must be float32")
        if not np.all(np.isfinite(latents)):
            raise ValueError("SAFE episode latents contain NaN or Inf")
        if int(self.seed) < 0 or int(self.env_steps) <= 0:
            raise ValueError("SAFE episode seed and env_steps are invalid")


@dataclass(frozen=True)
class SafeRolloutDataset:
    episodes: tuple[SafeRolloutEpisode, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("SAFE rollout dataset is empty")
        expected_metadata = {
            "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
            "safe_latent_dim": SAFE_LATENT_DIM,
            "safe_latent_source": SAFE_LATENT_SOURCE,
            "safe_latent_diffusion_selection": SAFE_LATENT_DIFFUSION_SELECTION,
            "safe_latent_horizon_selection": SAFE_LATENT_HORIZON_SELECTION,
            "safe_latent_pooling": SAFE_LATENT_POOLING,
            "label_source": "episode_environment_success_any_step",
            "base_policy_only": True,
        }
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected_metadata.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"SAFE rollout metadata does not match the protocol: {mismatches}")
        if int(self.metadata.get("chunk_len", 0)) <= 0:
            raise ValueError("SAFE rollout dataset chunk_len is invalid")
        if int(self.metadata.get("action_dim", 0)) <= 0:
            raise ValueError("SAFE rollout dataset action_dim is invalid")
        pred_horizon = self.metadata.get("safe_latent_pred_horizon")
        if (
            isinstance(pred_horizon, bool)
            or not isinstance(pred_horizon, int)
            or pred_horizon < 2
        ):
            raise ValueError("SAFE rollout dataset prediction horizon is invalid")
        chunk_len = int(self.metadata["chunk_len"])
        if any(
            len(episode.latents) != (episode.env_steps + chunk_len - 1) // chunk_len
            for episode in self.episodes
        ):
            raise ValueError("SAFE rollout latent count does not match executed chunks")
        seeds = [episode.seed for episode in self.episodes]
        if len(seeds) != len(set(seeds)):
            raise ValueError("SAFE rollout episode seeds must be unique")


def save_safe_rollout_dataset(
    path: str | Path, dataset: SafeRolloutDataset
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    latent_parts: list[np.ndarray] = []
    for episode in dataset.episodes:
        latent_parts.append(episode.latents)
        offsets.append(offsets[-1] + len(episode.latents))
    payload = {
        "schema": np.asarray(SAFE_ROLLOUT_SCHEMA),
        "metadata_json": np.asarray(
            json.dumps(
                dataset.metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        ),
        "latents": np.concatenate(latent_parts, axis=0),
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


def load_safe_rollout_dataset(path: str | Path) -> SafeRolloutDataset:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        schema = str(np.asarray(payload["schema"]).item())
        if schema != SAFE_ROLLOUT_SCHEMA:
            raise ValueError(f"Unsupported SAFE rollout schema {schema!r}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        if not isinstance(metadata, dict):
            raise ValueError("SAFE rollout metadata must be a JSON object")
        latents = np.asarray(payload["latents"])
        offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
        successes = np.asarray(payload["successes"], dtype=np.bool_)
        seeds = np.asarray(payload["seeds"], dtype=np.int64)
        env_steps = np.asarray(payload["env_steps"], dtype=np.int64)
    episode_count = len(successes)
    if latents.ndim != 2 or latents.dtype != np.dtype(np.float32):
        raise ValueError("SAFE rollout latent array is invalid")
    if not np.all(np.isfinite(latents)):
        raise ValueError("SAFE rollout latent array contains NaN or Inf")
    if offsets.shape != (episode_count + 1,):
        raise ValueError("SAFE rollout episode offsets have the wrong shape")
    if seeds.shape != successes.shape or env_steps.shape != successes.shape:
        raise ValueError("SAFE rollout episode metadata arrays disagree")
    if (
        offsets[0] != 0
        or offsets[-1] != len(latents)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("SAFE rollout episode offsets are not strictly increasing")
    episodes = tuple(
        SafeRolloutEpisode(
            latents=latents[offsets[index] : offsets[index + 1]].copy(),
            success=bool(successes[index]),
            seed=int(seeds[index]),
            env_steps=int(env_steps[index]),
        )
        for index in range(episode_count)
    )
    return SafeRolloutDataset(episodes=episodes, metadata=dict(metadata))


def stratified_safe_split_indices(
    successes: Sequence[bool],
    *,
    seen_train_ratio: float,
    unseen_episode_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    """Create episode-disjoint splits with both outcomes in every split."""

    if not 0.0 < seen_train_ratio < 1.0:
        raise ValueError("SAFE seen_train_ratio must lie in (0, 1)")
    if not 0.0 < unseen_episode_ratio < 1.0:
        raise ValueError("SAFE unseen_episode_ratio must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val_seen": [], "val_unseen": []}
    success_array = np.asarray(successes, dtype=np.bool_)
    for outcome in (False, True):
        indices = np.flatnonzero(success_array == outcome)
        if len(indices) < 3:
            raise ValueError(
                "Official SAFE requires at least three success and three failure "
                "episodes for train/calibration/test"
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
            raise ValueError("SAFE split leaves no training episodes")
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
    calibration_successes = sum(bool(success_array[index]) for index in splits["val_seen"])
    if calibration_successes < 4:
        raise ValueError(
            "Official SAFE functional conformal requires at least four successful "
            "episodes in val_seen; collect more successes or lower seen_train_ratio"
        )
    return splits


class OfficialSafeAdapter:
    """Inject ManiSkill rollouts into the unmodified official SAFE trainer."""

    def __init__(self, rollout_type: Callable[..., Any]) -> None:
        self._rollout_type = rollout_type
        self._dataset: SafeRolloutDataset | None = None
        self._source: Path | None = None

    def prepare_config(self, cfg: Any) -> Any:
        if cfg.dataset.name != "pizero":
            raise ValueError("ManiSkill SAFE adaptation requires dataset=pizero")
        if cfg.dataset.data_path_unseen is not None:
            raise ValueError(
                "Use the adapter's stratified split; dataset.data_path_unseen must be null"
            )
        if cfg.dataset.diff_idx_rel != "concat-2":
            raise ValueError("SAFE diffusion selection must be concat-2")
        if cfg.dataset.horizon_idx_rel != "concat-2":
            raise ValueError("SAFE horizon selection must be concat-2")
        if bool(cfg.dataset.normalize_hidden_states):
            raise ValueError(
                "Official whole-dataset normalization is disabled to avoid split leakage"
            )
        source = Path(str(cfg.dataset.data_path)).resolve()
        if self._dataset is None or source != self._source:
            self._dataset = load_safe_rollout_dataset(source)
            self._source = source
        metadata = self._dataset.metadata
        cfg.dataset.dim_features = SAFE_LATENT_DIM
        cfg.dataset.dim_action = int(metadata["action_dim"])
        cfg.dataset.pred_horizon = int(metadata["safe_latent_pred_horizon"])
        cfg.dataset.exec_horizon = int(metadata["chunk_len"])
        return cfg

    def load_rollouts(self, cfg: Any) -> list[Any]:
        self.prepare_config(cfg)
        assert self._dataset is not None
        minimum_chunks = min(len(episode.latents) for episode in self._dataset.episodes)
        task_description = str(self._dataset.metadata.get("env_id", "ManiSkill"))
        exec_horizon = int(self._dataset.metadata["chunk_len"])
        return [
            self._rollout_type(
                hidden_states=torch.from_numpy(episode.latents.copy()),
                task_suite_name="maniskill",
                task_id=0,
                task_description=task_description,
                episode_idx=episode.seed,
                episode_success=int(episode.success),
                mp4_path="",
                logs=None,
                task_min_step=minimum_chunks,
                exec_horizon=exec_horizon,
                action_vectors=None,
            )
            for episode in self._dataset.episodes
        ]

    def split_rollouts(
        self, cfg: Any, all_rollouts: Sequence[Any]
    ) -> dict[str, list[Any]]:
        indices_by_split = stratified_safe_split_indices(
            [bool(rollout.episode_success) for rollout in all_rollouts],
            seen_train_ratio=float(cfg.dataset.seen_train_ratio),
            unseen_episode_ratio=float(cfg.dataset.unseen_task_ratio),
            seed=int(cfg.train.seed),
        )
        result = {
            split: [all_rollouts[index] for index in indices]
            for split, indices in indices_by_split.items()
        }
        for split, rollouts in result.items():
            successes = sum(int(rollout.episode_success) for rollout in rollouts)
            print(
                f"{split}: {len(rollouts)} rollouts, "
                f"{successes} success, {len(rollouts) - successes} fail"
            )
        return result
