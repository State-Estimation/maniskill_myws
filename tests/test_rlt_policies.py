from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.openpi_bridge import remote_policy as remote_policy_module
from maniskill_myws.rlt.policies import (
    BASE_ACTION_PROJECTION,
    RemoteOpenPIChunkPolicy,
    inference_seed_for_step,
    metadata_sha256,
    openpi_policy_identity_sha256,
    project_action_chunk_to_bounds,
)


def test_inference_seed_is_stable_and_step_specific() -> None:
    assert inference_seed_for_step(7, 16) == inference_seed_for_step(7, 16)
    assert inference_seed_for_step(7, 16) != inference_seed_for_step(7, 24)
    assert inference_seed_for_step(7, 16) != inference_seed_for_step(8, 16)


def test_metadata_hash_is_stable_and_content_sensitive() -> None:
    left = {"checkpoint": "pi0", "nested": {"x": np.asarray([1, 2])}}
    reordered = {"nested": {"x": np.asarray([1, 2])}, "checkpoint": "pi0"}
    changed = {"checkpoint": "pi0", "nested": {"x": np.asarray([1, 3])}}

    assert metadata_sha256(left) == metadata_sha256(reordered)
    assert metadata_sha256(left) != metadata_sha256(changed)


def test_openpi_policy_identity_ignores_path_but_binds_content() -> None:
    def metadata(path: str, content: str) -> dict:
        return {
            "maniskill_policy_identity": {
                "config": "pi0_maniskill",
                "repo_id": "local/task",
                "checkpoint": path,
                "resolved_checkpoint": path,
                "checkpoint_content_sha256": content,
                "default_prompt": None,
            },
            "inference_seed_protocol": "seed-v1",
            "frozen_latent_protocol": "latent-v1",
            "frozen_latent_key": "z",
            "frozen_latent_shape": [1024],
            "frozen_latent_dtype": "float32",
            "frozen_latent_source": "suffix",
            "frozen_latent_pooling": "mean",
        }

    relative = metadata("checkpoints/step", "a" * 64)
    absolute = metadata("/workspace/checkpoints/step", "a" * 64)
    changed = metadata("/workspace/checkpoints/step", "b" * 64)
    assert openpi_policy_identity_sha256(relative) == openpi_policy_identity_sha256(
        absolute
    )
    assert openpi_policy_identity_sha256(relative) != openpi_policy_identity_sha256(
        changed
    )


def test_action_projection_clips_to_per_dimension_environment_bounds() -> None:
    low = np.asarray([-2.0, -1.0, -3.0], dtype=np.float32)
    high = np.asarray([2.0, 1.0, -0.1], dtype=np.float32)
    raw = np.asarray(
        [[-2.2, 0.0, -0.05], [1.5, 1.4, -1.0]], dtype=np.float32
    )

    projected, report = project_action_chunk_to_bounds(
        raw,
        action_low=low,
        action_high=high,
        expected_shape=(2, 3),
    )

    np.testing.assert_allclose(
        projected,
        np.asarray([[-2.0, 0.0, -0.1], [1.5, 1.0, -1.0]], dtype=np.float32),
    )
    assert report["clipped_actions"] == 2
    assert report["clipped_values"] == 3
    assert report["max_lower_violation"] == pytest.approx(0.2)
    assert report["max_upper_violation"] == pytest.approx(0.4)
    assert report["clipped_values_by_dim"] == [1, 1, 1]
    assert report["max_abs_correction_by_dim"] == pytest.approx([0.2, 0.4, 0.05])
    np.testing.assert_array_equal(
        raw,
        np.asarray([[-2.2, 0.0, -0.05], [1.5, 1.4, -1.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.zeros((3, 2), dtype=np.float32), "shape"),
        (np.asarray([[0.0, np.nan]], dtype=np.float32), "NaN or Inf"),
        (np.asarray([["0", "1"]]), "numeric"),
    ],
)
def test_action_projection_rejects_schema_errors(value, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        project_action_chunk_to_bounds(
            value,
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
            expected_shape=(1, 2),
        )


def test_remote_policy_returns_projected_reference_and_tracks_stats(monkeypatch) -> None:
    raw_chunk = np.asarray(
        [
            [1.2, -0.5],
            [0.4, -1.3],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    class FakeRemotePolicy:
        def __init__(self, **_kwargs) -> None:
            self.server_metadata = {"checkpoint": "test"}
            self.last_seed = None

        def reset(self) -> None:
            pass

        def act(self, _obs, *, inference_seed=None) -> np.ndarray:
            self.last_seed = inference_seed
            return raw_chunk[0]

        def planned_chunk(self, *, include_current=True) -> np.ndarray:
            assert include_current
            return raw_chunk

    monkeypatch.setattr(
        remote_policy_module,
        "RemoteWebsocketChunkPolicy",
        FakeRemotePolicy,
    )
    policy = RemoteOpenPIChunkPolicy(
        server="ws://test",
        prompt="test",
        image_key="base",
        wrist_image_key="wrist",
        state_keys=["state"],
        action_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
    )

    planned = policy.plan({}, chunk_len=2, action_dim=2, inference_seed=123)

    np.testing.assert_allclose(
        planned,
        np.asarray([[1.0, -0.5], [0.4, -1.0]], dtype=np.float32),
    )
    assert policy.policy.last_seed == 123
    assert policy.action_projection == BASE_ACTION_PROJECTION
    stats = policy.action_projection_stats
    assert stats["chunks"] == 1
    assert stats["actions"] == 2
    assert stats["clipped_actions"] == 2
    assert stats["clipped_values"] == 2
    assert stats["action_clip_rate"] == 1.0
    assert stats["value_clip_rate"] == 0.5
    assert stats["max_abs_correction"] == pytest.approx(0.3)
    assert stats["clipped_values_by_dim"] == [1, 1]
    assert stats["max_abs_correction_by_dim"] == pytest.approx([0.2, 0.3])


def test_remote_policy_returns_temporal_latent_triple(monkeypatch) -> None:
    chunk = np.zeros((3, 2), dtype=np.float32)
    mean = np.arange(4, dtype=np.float32)
    temporal = np.arange(20, dtype=np.float32).reshape(5, 4)

    class FakeRemotePolicy:
        def __init__(self, **kwargs) -> None:
            assert kwargs["require_frozen_latent"]
            assert kwargs["require_frozen_temporal_latent"]
            self.server_metadata = {"checkpoint": "test"}

        def reset(self) -> None:
            pass

        def act(self, _obs, *, inference_seed=None) -> np.ndarray:
            assert inference_seed == 123
            return chunk[0]

        def planned_chunk(self, *, include_current=True) -> np.ndarray:
            assert include_current
            return chunk

        def planned_latent(self) -> np.ndarray:
            return mean

        def planned_temporal_latent(self) -> np.ndarray:
            return temporal

    monkeypatch.setattr(
        remote_policy_module,
        "RemoteWebsocketChunkPolicy",
        FakeRemotePolicy,
    )
    policy = RemoteOpenPIChunkPolicy(
        server="ws://test",
        prompt="test",
        image_key="base",
        wrist_image_key="wrist",
        state_keys=["state"],
        action_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        require_frozen_latent=True,
        require_frozen_temporal_latent=True,
    )

    planned, planned_mean, planned_temporal = policy.plan_with_temporal_latent(
        {}, chunk_len=2, action_dim=2, inference_seed=123
    )
    np.testing.assert_array_equal(planned, chunk[:2])
    np.testing.assert_array_equal(planned_mean, mean)
    np.testing.assert_array_equal(planned_temporal, temporal)
