from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.rlt.actprobe import (
    ACTPROBE_ACTION_SOURCE,
    ACTPROBE_FEATURE_NAMES,
    ACTPROBE_FEATURE_PROTOCOL,
    ACTPROBE_LABEL_SOURCE,
    ActProbeDataset,
    ActProbeEpisode,
    compute_actprobe_features,
    load_actprobe_dataset,
    save_actprobe_dataset,
    stratified_actprobe_split_indices,
)


def _metadata() -> dict:
    return {
        "env_id": "TakeSafetyHook-v1",
        "actprobe_feature_protocol": ACTPROBE_FEATURE_PROTOCOL,
        "actprobe_feature_names": list(ACTPROBE_FEATURE_NAMES),
        "actprobe_action_source": ACTPROBE_ACTION_SOURCE,
        "label_source": ACTPROBE_LABEL_SOURCE,
        "base_policy_only": True,
        "chunk_len": 2,
        "prediction_horizon": 5,
        "action_dim": 2,
    }


def _episode(seed: int, success: bool, chunks: int = 2) -> ActProbeEpisode:
    return ActProbeEpisode(
        features=np.full((chunks, 2), seed / 100.0, dtype=np.float32),
        success=success,
        seed=seed,
        env_steps=chunks * 2,
    )


def test_actprobe_features_match_official_acm_tce_formula() -> None:
    previous = np.asarray(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        dtype=np.float32,
    )
    current = np.asarray(
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0]],
        dtype=np.float32,
    )

    first = compute_actprobe_features(None, previous, executed_steps=2)
    second = compute_actprobe_features(previous, current, executed_steps=2)

    assert first[0] == pytest.approx(np.sqrt(np.mean(previous[:2] ** 2)))
    assert first[1] == 0.0
    assert second[0] == pytest.approx(np.sqrt(np.mean(current[:2] ** 2)))
    assert second[1] == 0.0
    assert second.dtype == np.float32


def test_actprobe_tce_uses_overlapping_prediction_horizons() -> None:
    previous = np.zeros((5, 2), dtype=np.float32)
    current = np.zeros((5, 2), dtype=np.float32)
    previous[2:] = 3.0
    current[:3] = 1.0

    features = compute_actprobe_features(previous, current, executed_steps=2)

    assert features[1] == pytest.approx(4.0)


def test_actprobe_dataset_roundtrip(tmp_path) -> None:
    expected = ActProbeDataset(
        episodes=tuple(
            _episode(seed, success=seed >= 8, chunks=2 + seed % 2)
            for seed in range(16)
        ),
        metadata=_metadata(),
    )
    path = tmp_path / "rollouts.npz"

    save_actprobe_dataset(path, expected)
    actual = load_actprobe_dataset(path)

    assert actual.metadata == expected.metadata
    assert len(actual.episodes) == len(expected.episodes)
    for left, right in zip(actual.episodes, expected.episodes, strict=True):
        assert (left.success, left.seed, left.env_steps) == (
            right.success,
            right.seed,
            right.env_steps,
        )
        np.testing.assert_array_equal(left.features, right.features)


def test_actprobe_split_is_stratified_disjoint_and_deterministic() -> None:
    successes = [False] * 20 + [True] * 20

    first = stratified_actprobe_split_indices(successes, seed=7)
    second = stratified_actprobe_split_indices(successes, seed=7)

    assert first == second
    split_sets = [set(indices) for indices in first.values()]
    assert set.union(*split_sets) == set(range(40))
    assert not (split_sets[0] & split_sets[1])
    assert not (split_sets[0] & split_sets[2])
    assert not (split_sets[1] & split_sets[2])
    for indices in first.values():
        assert {successes[index] for index in indices} == {False, True}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actprobe_feature_protocol", "custom"),
        ("actprobe_feature_names", ["tce", "acm"]),
        ("actprobe_action_source", "projected_actions"),
        ("label_source", "shaped_reward"),
        ("base_policy_only", False),
    ],
)
def test_actprobe_dataset_rejects_protocol_drift(field: str, value: object) -> None:
    metadata = _metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match="does not match the protocol"):
        ActProbeDataset(
            episodes=(_episode(0, success=False),), metadata=metadata
        )


def test_actprobe_requires_overlap_horizon_and_chunk_alignment() -> None:
    metadata = _metadata()
    metadata["prediction_horizon"] = metadata["chunk_len"]
    with pytest.raises(ValueError, match="must exceed chunk_len"):
        ActProbeDataset(
            episodes=(_episode(0, success=False),), metadata=metadata
        )

    with pytest.raises(ValueError, match="feature count does not match"):
        ActProbeDataset(
            episodes=(
                ActProbeEpisode(
                    features=np.zeros((2, 2), dtype=np.float32),
                    success=False,
                    seed=0,
                    env_steps=5,
                ),
            ),
            metadata=_metadata(),
        )
