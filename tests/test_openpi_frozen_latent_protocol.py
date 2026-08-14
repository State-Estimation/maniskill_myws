from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.openpi_bridge.remote_policy import (
    FROZEN_LATENT_DIM,
    FROZEN_LATENT_KEY,
    FROZEN_LATENT_PROTOCOL,
    _validate_frozen_latent,
    _validate_frozen_latent_metadata,
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
