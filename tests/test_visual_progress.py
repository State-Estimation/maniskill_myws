from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from maniskill_myws.pld.visual_progress import (
    VisualProgressConfig,
    VisualProgressEnsemble,
    VisualTaskProgressModel,
    visual_progress_training_loss,
)
from maniskill_myws.pld.visual_progress_dataset import (
    H5VisualProgressPairDataset,
    scan_visual_episodes,
    split_visual_episodes,
)


def _write_visual_h5(path: Path) -> None:
    with h5py.File(path, "w") as h5_file:
        for episode_index, success in enumerate((True, True, False, False)):
            group = h5_file.create_group(f"traj_{episode_index}")
            group.create_dataset("actions", data=np.zeros((8, 2), dtype=np.float32))
            success_rows = np.zeros(8, dtype=bool)
            success_rows[-1] = success
            group.create_dataset("success", data=success_rows)
            sensor_data = group.create_group("obs").create_group("sensor_data")
            for view_index, camera in enumerate(("base_camera", "hand_camera")):
                camera_group = sensor_data.create_group(camera)
                images = np.zeros((9, 32, 32, 3), dtype=np.uint8)
                for timestep in range(9):
                    images[timestep, :, :, 0] = timestep * 20
                    images[timestep, :, :, 1] = episode_index * 20
                    images[timestep, :, :, 2] = view_index * 50
                camera_group.create_dataset("rgb", data=images)
            # These fields deliberately exist in the source file. The visual
            # dataset must never expose them to the model.
            agent = group["obs"].create_group("agent")
            agent.create_dataset("qpos", data=np.ones((9, 9), dtype=np.float32))
            extra = group["obs"].create_group("extra")
            extra.create_dataset(
                "clean_coverage", data=np.linspace(0, 0.6, 9, dtype=np.float32)
            )


def test_visual_dataset_never_returns_environment_state(tmp_path: Path) -> None:
    h5_path = tmp_path / "rollouts.h5"
    _write_visual_h5(h5_path)
    episodes = scan_visual_episodes([h5_path])
    assert sum(episode.success for episode in episodes) == 2
    dataset = H5VisualProgressPairDataset(
        [h5_path],
        episodes,
        image_keys=[
            "sensor_data/base_camera/rgb",
            "sensor_data/hand_camera/rgb",
        ],
        image_size=32,
        context_frames=2,
        context_stride=1,
        sequence_points=4,
        samples_per_epoch=2,
        seed=3,
    )
    item = dataset[0]
    assert set(item) == {
        "success_a_images",
        "success_b_images",
        "failure_images",
        "success_a_time",
        "success_b_time",
        "failure_time",
    }
    assert item["success_a_images"].shape == (4, 2, 2, 32, 32, 3)
    assert item["success_a_images"].dtype == np.uint8
    assert not any("qpos" in key or "coverage" in key for key in item)


def test_visual_episode_split_has_no_overlap(tmp_path: Path) -> None:
    h5_path = tmp_path / "rollouts.h5"
    _write_visual_h5(h5_path)
    episodes = scan_visual_episodes([h5_path])
    train, validation = split_visual_episodes(
        episodes, validation_fraction=0.5, seed=0
    )
    train_names = {episode.group_name for episode in train}
    validation_names = {episode.group_name for episode in validation}
    assert train_names.isdisjoint(validation_names)
    assert {episode.success for episode in train} == {False, True}
    assert {episode.success for episode in validation} == {False, True}


def test_visual_progress_loss_and_ensemble_roundtrip(tmp_path: Path) -> None:
    config = VisualProgressConfig(
        context_frames=2,
        num_views=2,
        image_height=32,
        image_width=32,
        visual_latent_dim=32,
        embedding_dim=16,
        hidden_dim=32,
    )
    model = VisualTaskProgressModel(config)
    rng = np.random.default_rng(1)
    raw = rng.integers(
        0,
        256,
        size=(2, 4, 2, 2, 32, 32, 3),
        dtype=np.uint8,
    )
    tensor = model.clips_to_tensor(raw, device="cpu")
    output_a = model(tensor)
    output_b = model(torch.roll(tensor, shifts=1, dims=1))
    output_failure = model(torch.flip(tensor, dims=(1,)))
    loss, metrics = visual_progress_training_loss(
        output_a, output_b, output_failure
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(metrics["success_accuracy"]) <= 1.0
    assert output_a["progress"].shape == (2, 4, 1)
    assert output_a["embedding"].shape == (2, 4, 16)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    ensemble = VisualProgressEnsemble(config, ensemble_size=2)
    checkpoint = tmp_path / "visual_progress.pt"
    ensemble.save(checkpoint, metadata={"visual_only": True})
    loaded, metadata = VisualProgressEnsemble.load(checkpoint)
    prediction = loaded.predict(raw[0, 0])
    batch_prediction = loaded.predict_batch(raw[0, :3])
    assert metadata["visual_only"] is True
    assert 0.0 <= prediction["progress"] <= 1.0
    assert prediction["progress_uncertainty"] >= 0.0
    assert prediction["embedding"].shape == (16,)
    assert prediction["embedding_uncertainty"] >= 0.0
    assert batch_prediction["progress"].shape == (3,)
    assert batch_prediction["progress_uncertainty"].shape == (3,)
    assert batch_prediction["success_probability"].shape == (3,)
    assert batch_prediction["success_uncertainty"].shape == (3,)
    assert batch_prediction["embedding"].shape == (3, 16)
    assert batch_prediction["embedding_uncertainty"].shape == (3,)
    assert prediction["progress"] == pytest.approx(
        float(batch_prediction["progress"][0]), abs=1e-6
    )
