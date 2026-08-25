from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from maniskill_myws.openpi_bridge.remote_policy import SAFE_LATENT_DIM
from maniskill_myws.rlt.value_dataset import (
    VALUE_ROLLOUT_SCHEMA,
    ValueBoundaryDataset,
    scan_value_episodes,
    stratified_value_split,
)
from maniskill_myws.rlt.value_model import (
    DistributionalBaseValueModel,
    DistributionalValueConfig,
    ValueEstimate,
    return_bin_target,
    value_potential_shaping,
    value_progress_estimate,
)


def _config() -> DistributionalValueConfig:
    return DistributionalValueConfig(
        state_dim=7,
        chunk_len=10,
        max_episode_steps=50,
        num_views=2,
        image_height=16,
        image_width=16,
        modality_dim=32,
        feature_dim=24,
        hidden_dim=32,
    )


def test_distributional_value_preserves_failure_and_time_support() -> None:
    config = _config()
    model = DistributionalBaseValueModel(config)
    output = model(
        torch.zeros(3, 2, 16, 16, 3, dtype=torch.uint8),
        torch.zeros(3, 7),
        torch.zeros(3, SAFE_LATENT_DIM),
        torch.zeros(3, 10, 8),
        torch.tensor([0, 10, 20]),
    )
    assert output["logits"].shape == (3, 7)
    assert model.critic_features(output).shape == (3, config.critic_feature_dim)
    torch.testing.assert_close(output["probability"].sum(dim=-1), torch.ones(3))
    assert float(model.return_support[0]) < float(model.return_support[-1])
    assert (
        return_bin_target(success=False, boundary_index=0, boundary_count=5, max_remaining_chunks=5)
        == 0
    )
    assert (
        return_bin_target(success=True, boundary_index=0, boundary_count=5, max_remaining_chunks=5)
        == 6
    )
    assert (
        return_bin_target(success=True, boundary_index=4, boundary_count=5, max_remaining_chunks=5)
        == 2
    )


def test_distributional_value_checkpoint_round_trip(tmp_path) -> None:
    model = DistributionalBaseValueModel(_config())
    path = tmp_path / "value.pt"
    model.save(path, metadata={"purpose": "test"})
    restored, metadata = DistributionalBaseValueModel.load(path)
    assert restored.config == model.config
    assert metadata == {"purpose": "test"}
    for expected, actual in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_macro_potential_shaping_telescopes() -> None:
    gamma = 0.99
    first = value_potential_shaping(
        current_potential=-0.8,
        next_potential=-0.5,
        gamma=gamma,
        duration=10,
        chunk_len=10,
        weight=1.0,
    )
    second = value_potential_shaping(
        current_potential=-0.5,
        next_potential=0.0,
        gamma=gamma,
        duration=10,
        chunk_len=10,
        weight=1.0,
    )
    assert first + gamma * second == pytest.approx(0.8)


def test_value_progress_conditions_on_success_mass() -> None:
    estimate = ValueEstimate(
        critic_features=np.zeros(1, dtype=np.float32),
        potential=-0.4,
        failure_probability=0.2,
        entropy=0.5,
        expected_remaining_chunks=20.0,
    )
    progress = value_progress_estimate(estimate, max_remaining_chunks=50)
    assert progress.success_probability == pytest.approx(0.8)
    assert progress.conditional_remaining_chunks == pytest.approx(25.0)
    assert progress.completion_proximity == pytest.approx(0.5)


def test_value_progress_is_conservative_without_success_mass() -> None:
    estimate = ValueEstimate(
        critic_features=np.zeros(1, dtype=np.float32),
        potential=-1.25,
        failure_probability=1.0,
        entropy=0.0,
        expected_remaining_chunks=0.0,
    )
    progress = value_progress_estimate(estimate, max_remaining_chunks=50)
    assert progress.success_probability == 0.0
    assert progress.conditional_remaining_chunks == 50.0
    assert progress.completion_proximity == 0.0


def _write_episode(file, index: int, *, success: bool) -> None:
    group = file.create_group(f"episode_{index:06d}")
    boundaries = 2
    group.attrs["seed"] = 100 + index
    group.attrs["success"] = success
    group.attrs["env_steps"] = 20
    group.create_dataset("images", data=np.zeros((boundaries, 2, 8, 8, 3), dtype=np.uint8))
    group.create_dataset("states", data=np.zeros((boundaries, 7), dtype=np.float32))
    group.create_dataset("latents", data=np.zeros((boundaries, SAFE_LATENT_DIM), dtype=np.float16))
    group.create_dataset("ref_chunks", data=np.zeros((boundaries, 10, 8), dtype=np.float32))
    group.create_dataset("step_ids", data=np.asarray([0, 10], dtype=np.int32))


def test_value_dataset_split_is_episode_disjoint(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "value.h5"
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = VALUE_ROLLOUT_SCHEMA
        file.attrs["metadata_json"] = json.dumps({"chunk_len": 10})
        for index in range(6):
            _write_episode(file, index, success=index >= 3)
    episodes = scan_value_episodes(path)
    train, validation = stratified_value_split(episodes, validation_fraction=0.34, seed=4)
    assert {episode.seed for episode in train}.isdisjoint(episode.seed for episode in validation)
    assert {episode.success for episode in train} == {False, True}
    assert {episode.success for episode in validation} == {False, True}
    dataset = ValueBoundaryDataset(path, validation, max_remaining_chunks=5)
    try:
        item = dataset[0]
        assert item["latent"].shape == (SAFE_LATENT_DIM,)
        assert item["images"].shape == (2, 8, 8, 3)
    finally:
        dataset.close()
