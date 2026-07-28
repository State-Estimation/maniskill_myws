from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.openpi_bridge.remote_policy import (
    FROZEN_LATENT_DIM,
    FROZEN_LATENT_KEY,
    FROZEN_LATENT_PROTOCOL,
    FROZEN_TEMPORAL_LATENT_BINS,
    FROZEN_TEMPORAL_LATENT_ACTION_HORIZON,
    FROZEN_TEMPORAL_LATENT_BIN_WIDTH,
    FROZEN_TEMPORAL_LATENT_KEY,
    FROZEN_TEMPORAL_LATENT_PROTOCOL,
    _validate_frozen_latent,
    _validate_frozen_latent_pair,
    _validate_frozen_latent_metadata,
    _validate_frozen_temporal_latent,
    _validate_frozen_temporal_latent_metadata,
)


def _metadata() -> dict:
    return {
        "frozen_latent_protocol": FROZEN_LATENT_PROTOCOL,
        "frozen_latent_key": FROZEN_LATENT_KEY,
        "frozen_latent_shape": [FROZEN_LATENT_DIM],
        "frozen_latent_dtype": "float32",
    }


def _temporal_metadata() -> dict:
    return {
        "frozen_temporal_latent_protocol": FROZEN_TEMPORAL_LATENT_PROTOCOL,
        "frozen_temporal_latent_key": FROZEN_TEMPORAL_LATENT_KEY,
        "frozen_temporal_latent_shape": [
            FROZEN_TEMPORAL_LATENT_BINS,
            FROZEN_LATENT_DIM,
        ],
        "frozen_temporal_latent_dtype": "float32",
        "frozen_temporal_latent_source": (
            "pi0_final_denoise_action_suffix_tokens"
        ),
        "frozen_temporal_latent_pooling": "ordered_equal_contiguous_bins",
        "frozen_temporal_latent_action_horizon": (
            FROZEN_TEMPORAL_LATENT_ACTION_HORIZON
        ),
        "frozen_temporal_latent_bin_width": FROZEN_TEMPORAL_LATENT_BIN_WIDTH,
        "frozen_temporal_latent_parent_protocol": FROZEN_LATENT_PROTOCOL,
        "frozen_temporal_latent_parent_key": FROZEN_LATENT_KEY,
    }


def test_frozen_latent_protocol_accepts_exact_schema() -> None:
    _validate_frozen_latent_metadata(_metadata())
    latent = np.arange(FROZEN_LATENT_DIM, dtype=np.float32)
    restored = _validate_frozen_latent(latent)
    np.testing.assert_array_equal(restored, latent)
    assert restored is not latent


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frozen_latent_protocol", "wrong"),
        ("frozen_latent_key", "wrong"),
        ("frozen_latent_shape", [2048]),
        ("frozen_latent_dtype", "float16"),
    ],
)
def test_frozen_latent_protocol_rejects_metadata_drift(key: str, value) -> None:
    metadata = _metadata()
    metadata[key] = value
    with pytest.raises(RuntimeError, match="frozen Pi0 latent protocol"):
        _validate_frozen_latent_metadata(metadata)


def test_frozen_latent_rejects_shape_dtype_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        _validate_frozen_latent(np.zeros((FROZEN_LATENT_DIM - 1,), dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        _validate_frozen_latent(np.zeros((FROZEN_LATENT_DIM,), dtype=np.float16))
    nonfinite = np.zeros((FROZEN_LATENT_DIM,), dtype=np.float32)
    nonfinite[3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        _validate_frozen_latent(nonfinite)


def test_frozen_temporal_latent_protocol_accepts_exact_schema() -> None:
    _validate_frozen_temporal_latent_metadata(_temporal_metadata())
    latent = np.arange(
        FROZEN_TEMPORAL_LATENT_BINS * FROZEN_LATENT_DIM, dtype=np.float32
    ).reshape(FROZEN_TEMPORAL_LATENT_BINS, FROZEN_LATENT_DIM)
    restored = _validate_frozen_temporal_latent(latent)
    np.testing.assert_array_equal(restored, latent)
    assert restored is not latent


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frozen_temporal_latent_protocol", "wrong"),
        ("frozen_temporal_latent_key", "wrong"),
        ("frozen_temporal_latent_shape", [4, FROZEN_LATENT_DIM]),
        ("frozen_temporal_latent_dtype", "float16"),
        ("frozen_temporal_latent_source", "wrong"),
        ("frozen_temporal_latent_pooling", "wrong"),
        ("frozen_temporal_latent_action_horizon", 40),
        ("frozen_temporal_latent_bin_width", 8),
        ("frozen_temporal_latent_parent_protocol", "wrong"),
        ("frozen_temporal_latent_parent_key", "wrong"),
    ],
)
def test_frozen_temporal_latent_protocol_rejects_metadata_drift(
    key: str, value
) -> None:
    metadata = _temporal_metadata()
    metadata[key] = value
    with pytest.raises(RuntimeError, match="temporal latent protocol"):
        _validate_frozen_temporal_latent_metadata(metadata)


def test_frozen_temporal_latent_rejects_schema_errors() -> None:
    shape = (FROZEN_TEMPORAL_LATENT_BINS, FROZEN_LATENT_DIM)
    with pytest.raises(ValueError, match="shape"):
        _validate_frozen_temporal_latent(np.zeros(shape[1:], dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        _validate_frozen_temporal_latent(np.zeros(shape, dtype=np.float16))
    nonfinite = np.zeros(shape, dtype=np.float32)
    nonfinite[1, 3] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        _validate_frozen_temporal_latent(nonfinite)


def test_frozen_temporal_bins_must_reconstruct_mean_latent() -> None:
    temporal = np.arange(
        FROZEN_TEMPORAL_LATENT_BINS * FROZEN_LATENT_DIM, dtype=np.float32
    ).reshape(FROZEN_TEMPORAL_LATENT_BINS, FROZEN_LATENT_DIM)
    mean = temporal.mean(axis=0, dtype=np.float32)
    _validate_frozen_latent_pair(mean, temporal)

    changed = mean.copy()
    changed[17] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_frozen_latent_pair(changed, temporal)
