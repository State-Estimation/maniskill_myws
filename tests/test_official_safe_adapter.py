from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIFFUSION_SELECTION,
    SAFE_LATENT_DIM,
    SAFE_LATENT_HORIZON_SELECTION,
    SAFE_LATENT_POOLING,
    SAFE_LATENT_PROTOCOL,
    SAFE_LATENT_SOURCE,
)
from maniskill_myws.rlt.safe_rollouts import (
    OfficialSafeAdapter,
    SafeRolloutDataset,
    SafeRolloutEpisode,
    load_safe_rollout_dataset,
    save_safe_rollout_dataset,
    stratified_safe_split_indices,
)


def _episode(seed: int, success: bool, chunks: int = 2) -> SafeRolloutEpisode:
    return SafeRolloutEpisode(
        latents=np.full((chunks, SAFE_LATENT_DIM), seed / 100.0, dtype=np.float32),
        success=success,
        seed=seed,
        env_steps=chunks * 10,
    )


def _dataset() -> SafeRolloutDataset:
    episodes = tuple(
        _episode(seed, success=seed >= 16, chunks=2 + seed % 2)
        for seed in range(32)
    )
    return SafeRolloutDataset(
        episodes=episodes,
        metadata={
            "env_id": "TakeSafetyHook-v1",
            "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
            "safe_latent_dim": SAFE_LATENT_DIM,
            "safe_latent_source": SAFE_LATENT_SOURCE,
            "safe_latent_diffusion_selection": SAFE_LATENT_DIFFUSION_SELECTION,
            "safe_latent_horizon_selection": SAFE_LATENT_HORIZON_SELECTION,
            "safe_latent_pooling": SAFE_LATENT_POOLING,
            "safe_latent_pred_horizon": 50,
            "label_source": "episode_environment_success_any_step",
            "base_policy_only": True,
            "chunk_len": 10,
            "action_dim": 8,
        },
    )


def _config(path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            name="pizero",
            data_path=str(path),
            data_path_unseen=None,
            diff_idx_rel="concat-2",
            horizon_idx_rel="concat-2",
            normalize_hidden_states=False,
            seen_train_ratio=0.75,
            unseen_task_ratio=0.2,
            dim_features=None,
            dim_action=None,
            pred_horizon=None,
            exec_horizon=None,
        ),
        train=SimpleNamespace(seed=17),
    )


@dataclass
class _OfficialRollout:
    hidden_states: torch.Tensor
    task_suite_name: str
    task_id: int
    task_description: str
    episode_idx: int
    episode_success: int
    mp4_path: str
    logs: object = None
    task_min_step: int | None = None
    exec_horizon: int | None = None
    action_vectors: torch.Tensor | None = None


def test_safe_rollout_npz_roundtrip(tmp_path) -> None:
    path = tmp_path / "rollouts.npz"
    expected = _dataset()

    save_safe_rollout_dataset(path, expected)
    actual = load_safe_rollout_dataset(path)

    assert actual.metadata == expected.metadata
    assert len(actual.episodes) == len(expected.episodes)
    for left, right in zip(actual.episodes, expected.episodes, strict=True):
        assert (left.success, left.seed, left.env_steps) == (
            right.success,
            right.seed,
            right.env_steps,
        )
        np.testing.assert_array_equal(left.latents, right.latents)


def test_official_adapter_populates_config_and_rollouts(tmp_path) -> None:
    path = tmp_path / "rollouts.npz"
    save_safe_rollout_dataset(path, _dataset())
    cfg = _config(path)
    adapter = OfficialSafeAdapter(_OfficialRollout)

    adapter.prepare_config(cfg)
    rollouts = adapter.load_rollouts(cfg)

    assert cfg.dataset.dim_features == SAFE_LATENT_DIM
    assert cfg.dataset.dim_action == 8
    assert cfg.dataset.pred_horizon == 50
    assert cfg.dataset.exec_horizon == 10
    assert len(rollouts) == 32
    assert rollouts[0].hidden_states.shape == (2, SAFE_LATENT_DIM)
    assert rollouts[0].task_min_step == 2
    assert rollouts[0].action_vectors is None


def test_official_adapter_uses_stratified_episode_disjoint_splits(tmp_path) -> None:
    path = tmp_path / "rollouts.npz"
    save_safe_rollout_dataset(path, _dataset())
    cfg = _config(path)
    adapter = OfficialSafeAdapter(_OfficialRollout)
    rollouts = adapter.load_rollouts(cfg)

    first = adapter.split_rollouts(cfg, rollouts)
    second = adapter.split_rollouts(cfg, rollouts)

    assert {
        key: [rollout.episode_idx for rollout in value]
        for key, value in first.items()
    } == {
        key: [rollout.episode_idx for rollout in value]
        for key, value in second.items()
    }
    split_seeds = [
        {rollout.episode_idx for rollout in rollouts_split}
        for rollouts_split in first.values()
    ]
    assert set.union(*split_seeds) == set(range(32))
    assert not (split_seeds[0] & split_seeds[1])
    assert not (split_seeds[0] & split_seeds[2])
    assert not (split_seeds[1] & split_seeds[2])
    for rollouts_split in first.values():
        assert {rollout.episode_success for rollout in rollouts_split} == {0, 1}


def test_safe_split_requires_both_outcomes() -> None:
    with pytest.raises(ValueError, match="three success and three failure"):
        stratified_safe_split_indices(
            [False, False, False, True, True],
            seen_train_ratio=0.75,
            unseen_episode_ratio=0.2,
            seed=0,
        )


def test_safe_split_requires_enough_successful_calibration_episodes() -> None:
    with pytest.raises(ValueError, match="at least four successful"):
        stratified_safe_split_indices(
            [False] * 6 + [True] * 6,
            seen_train_ratio=0.75,
            unseen_episode_ratio=0.2,
            seed=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safe_latent_protocol", "old_protocol"),
        ("safe_latent_dim", 1024),
        ("safe_latent_source", "mean_latent"),
        ("safe_latent_diffusion_selection", "mean"),
        ("safe_latent_horizon_selection", "mean"),
        ("safe_latent_pooling", "mean"),
        ("label_source", "shaped_reward"),
        ("base_policy_only", False),
    ],
)
def test_safe_dataset_rejects_metadata_drift(field: str, value: object) -> None:
    dataset = _dataset()
    metadata = dict(dataset.metadata)
    metadata[field] = value

    with pytest.raises(ValueError, match="does not match the protocol"):
        SafeRolloutDataset(episodes=dataset.episodes, metadata=metadata)


def test_safe_dataset_rejects_invalid_horizon_and_chunk_alignment() -> None:
    dataset = _dataset()
    invalid_horizon = dict(dataset.metadata, safe_latent_pred_horizon=0)
    with pytest.raises(ValueError, match="prediction horizon is invalid"):
        SafeRolloutDataset(episodes=dataset.episodes, metadata=invalid_horizon)

    misaligned = (
        SafeRolloutEpisode(
            latents=np.zeros((2, SAFE_LATENT_DIM), dtype=np.float32),
            success=False,
            seed=100,
            env_steps=21,
        ),
    )
    with pytest.raises(ValueError, match="latent count does not match"):
        SafeRolloutDataset(episodes=misaligned, metadata=dataset.metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diff_idx_rel", "mean", "diffusion selection"),
        ("horizon_idx_rel", "mean", "horizon selection"),
        ("normalize_hidden_states", True, "split leakage"),
    ],
)
def test_official_adapter_rejects_protocol_drift(
    tmp_path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "rollouts.npz"
    save_safe_rollout_dataset(path, _dataset())
    cfg = _config(path)
    setattr(cfg.dataset, field, value)

    with pytest.raises(ValueError, match=message):
        OfficialSafeAdapter(_OfficialRollout).prepare_config(cfg)
