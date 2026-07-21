from __future__ import annotations

import json

import numpy as np
import pytest

from maniskill_myws.rlt.dataset import load_rollout_h5_into_replay
from maniskill_myws.rlt.replay import ChunkReplayBuffer, TransitionSource


h5py = pytest.importorskip("h5py")


def _write_rollout(path, *, env_id: str = "TestTask-v1") -> np.ndarray:
    actions = np.arange(5 * 8, dtype=np.float32).reshape(5, 8) / 100.0
    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=actions)
        trajectory.create_dataset("rewards", data=np.arange(5, dtype=np.float32))
        trajectory.create_dataset("terminated", data=[False, False, False, False, True])
        trajectory.create_dataset("truncated", data=np.zeros((5,), dtype=bool))
        trajectory.create_dataset("success", data=[False, False, False, False, True])
        obs = trajectory.create_group("obs")
        agent = obs.create_group("agent")
        extra = obs.create_group("extra")
        sensor_data = obs.create_group("sensor_data")
        agent.create_dataset(
            "qpos", data=np.arange(6 * 9, dtype=np.float32).reshape(6, 9)
        )
        agent.create_dataset(
            "qvel", data=np.arange(6 * 9, dtype=np.float32).reshape(6, 9) + 100
        )
        extra.create_dataset(
            "tcp_pose", data=np.arange(6 * 7, dtype=np.float32).reshape(6, 7) + 200
        )
        base_camera = sensor_data.create_group("base_camera")
        hand_camera = sensor_data.create_group("hand_camera")
        images = np.arange(6, dtype=np.uint8).reshape(6, 1, 1, 1)
        images = np.broadcast_to(images, (6, 2, 2, 3))
        base_camera.create_dataset("rgb", data=images)
        hand_camera.create_dataset("rgb", data=images + 10)

    metadata = {
        "env_info": {
            "env_id": env_id,
            "env_kwargs": {
                "control_mode": "pd_joint_pos",
                "reward_mode": "sparse",
            },
        }
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return actions


def test_rollout_h5_prefills_base_chunks_with_padding_and_images(tmp_path) -> None:
    rollout_path = tmp_path / "rollout.h5"
    actions = _write_rollout(rollout_path)
    buffer = ChunkReplayBuffer(
        4,
        state_dim=25,
        action_dim=8,
        chunk_len=3,
        image_shape=(2, 2, 2, 3),
    )

    stats = load_rollout_h5_into_replay(
        buffer,
        [rollout_path],
        state_keys=["agent/qpos", "agent/qvel", "extra/tcp_pose"],
        image_keys=["sensor_data/base_camera/rgb", "sensor_data/hand_camera/rgb"],
        expected_env_id="TestTask-v1",
        expected_control_mode="pd_joint_pos",
        expected_reward_mode="sparse",
        max_transitions=2,
    )

    assert stats.files == 1
    assert stats.episodes == 1
    assert stats.transitions == 2
    assert stats.env_steps == 5
    assert len(buffer) == 2
    np.testing.assert_allclose(buffer.action_chunks[0], actions[:3])
    np.testing.assert_allclose(buffer.action_chunks[1, :2], actions[3:5])
    np.testing.assert_allclose(buffer.action_chunks[1, 2], actions[4])
    np.testing.assert_allclose(buffer.next_ref_chunks[0, :2], actions[3:5])
    np.testing.assert_allclose(buffer.next_ref_chunks[0, 2], actions[4])
    np.testing.assert_allclose(buffer.next_ref_chunks[1], 0.0)
    np.testing.assert_allclose(buffer.rewards[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(buffer.rewards[1], [3.0, 4.0, 0.0])
    np.testing.assert_allclose(buffer.dones[:2], [0.0, 1.0])
    assert np.all(buffer.source_chunks[:2] == int(TransitionSource.BASE))
    assert np.all(buffer.images[0, 0] == 0)
    assert np.all(buffer.images[0, 1] == 10)
    assert np.all(buffer.next_images[0, 0] == 3)


def test_rollout_h5_rejects_incompatible_metadata(tmp_path) -> None:
    rollout_path = tmp_path / "rollout.h5"
    _write_rollout(rollout_path, env_id="WrongTask-v1")
    buffer = ChunkReplayBuffer(4, state_dim=25, action_dim=8, chunk_len=3)

    with pytest.raises(ValueError, match="env_id"):
        load_rollout_h5_into_replay(
            buffer,
            [rollout_path],
            state_keys=["agent/qpos", "agent/qvel", "extra/tcp_pose"],
            expected_env_id="TestTask-v1",
            max_transitions=2,
        )
