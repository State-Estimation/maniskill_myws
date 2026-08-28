from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.openpi_bridge.remote_policy import (
    FROZEN_LATENT_DIM,
    FROZEN_LATENT_KEY,
    FROZEN_LATENT_PROTOCOL,
    SAFE_LATENT_DIM,
    SAFE_LATENT_KEY,
    SAFE_LATENT_PROTOCOL,
    _execution_prefix,
    _validate_frozen_latent,
    _validate_frozen_latent_metadata,
    _validate_safe_latent,
    _validate_safe_latent_metadata,
)


def _metadata() -> dict:
    return {
        "frozen_latent_protocol": FROZEN_LATENT_PROTOCOL,
        "frozen_latent_key": FROZEN_LATENT_KEY,
        "frozen_latent_shape": [FROZEN_LATENT_DIM],
        "frozen_latent_dtype": "float32",
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


def _safe_metadata() -> dict:
    return {
        "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
        "safe_latent_key": SAFE_LATENT_KEY,
        "safe_latent_shape": [SAFE_LATENT_DIM],
        "safe_latent_dtype": "float32",
        "safe_latent_source": "pi0_action_expert_pre_velocity_tokens",
        "safe_latent_diffusion_selection": "concat-2_first_last",
        "safe_latent_horizon_selection": "concat-2_first_last",
        "safe_latent_pooling": "none",
        "safe_latent_pred_horizon": 50,
    }


def test_safe_endpoint_latent_protocol_is_exact_and_unpooled() -> None:
    _validate_safe_latent_metadata(_safe_metadata())
    latent = np.arange(SAFE_LATENT_DIM, dtype=np.float32)
    restored = _validate_safe_latent(latent)
    np.testing.assert_array_equal(restored, latent)
    assert restored is not latent


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("safe_latent_protocol", "wrong"),
        ("safe_latent_shape", [1024]),
        ("safe_latent_pooling", "mean"),
        ("safe_latent_diffusion_selection", "mean"),
    ],
)
def test_safe_endpoint_latent_rejects_protocol_drift(
    key: str, value: object
) -> None:
    metadata = _safe_metadata()
    metadata[key] = value
    with pytest.raises(RuntimeError, match="SAFE Pi0 pre-velocity protocol"):
        _validate_safe_latent_metadata(metadata)


def test_safe_endpoint_latent_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        _validate_safe_latent(np.zeros(1024, dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        _validate_safe_latent(np.zeros(SAFE_LATENT_DIM, dtype=np.float16))
    nonfinite = np.zeros(SAFE_LATENT_DIM, dtype=np.float32)
    nonfinite[3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        _validate_safe_latent(nonfinite)


def test_execution_prefix_replans_after_requested_number_of_actions() -> None:
    chunk = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    prefix = _execution_prefix(chunk, 10)

    assert prefix.shape == (10, 8)
    np.testing.assert_array_equal(prefix, chunk[:10])
    assert _execution_prefix(chunk, None) is chunk


@pytest.mark.parametrize("size", [0, -1, 51])
def test_execution_prefix_rejects_invalid_size(size: int) -> None:
    chunk = np.zeros((50, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="execution_chunk_size"):
        _execution_prefix(chunk, size)
