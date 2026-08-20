from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .obs_to_openpi import ObsAdapter, _to_uint8_hwc


INFERENCE_SEED_KEY = "__maniskill_inference_seed__"
INFERENCE_SEED_CAPABILITY = "maniskill_deterministic_inference_seed_v1"
FROZEN_LATENT_PROTOCOL = "maniskill_frozen_pi0_action_suffix_mean_v1"
FROZEN_LATENT_KEY = "frozen_pi0_latent"
FROZEN_LATENT_DIM = 1024
SAFE_LATENT_PROTOCOL = "safe_pi0_pre_velocity_diff2_horizon2_concat_v1"
SAFE_LATENT_KEY = "safe_pi0_pre_velocity"
SAFE_LATENT_DIM = 4 * 1024
SAFE_LATENT_SOURCE = "pi0_action_expert_pre_velocity_tokens"
SAFE_LATENT_DIFFUSION_SELECTION = "concat-2_first_last"
SAFE_LATENT_HORIZON_SELECTION = "concat-2_first_last"
SAFE_LATENT_POOLING = "none"


def _validate_frozen_latent_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "frozen_latent_protocol": FROZEN_LATENT_PROTOCOL,
        "frozen_latent_key": FROZEN_LATENT_KEY,
        "frozen_latent_shape": [FROZEN_LATENT_DIM],
        "frozen_latent_dtype": "float32",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "The OpenPI server does not expose the required frozen Pi0 "
            f"latent protocol: {mismatches}. Restart scripts/pi0/serve.py "
            "with --frozen-action-latent."
        )


def _validate_frozen_latent(value: Any) -> np.ndarray:
    latent = np.asarray(value)
    if latent.shape != (FROZEN_LATENT_DIM,):
        raise ValueError(
            f"OpenPI frozen latent shape {latent.shape} != {(FROZEN_LATENT_DIM,)}"
        )
    if latent.dtype != np.dtype(np.float32):
        raise TypeError(
            f"OpenPI frozen latent must be float32, got {latent.dtype}"
        )
    if not np.all(np.isfinite(latent)):
        raise ValueError("OpenPI frozen latent contains NaN or Inf")
    return latent.copy()


def _validate_safe_latent_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
        "safe_latent_key": SAFE_LATENT_KEY,
        "safe_latent_shape": [SAFE_LATENT_DIM],
        "safe_latent_dtype": "float32",
        "safe_latent_source": SAFE_LATENT_SOURCE,
        "safe_latent_diffusion_selection": SAFE_LATENT_DIFFUSION_SELECTION,
        "safe_latent_horizon_selection": SAFE_LATENT_HORIZON_SELECTION,
        "safe_latent_pooling": SAFE_LATENT_POOLING,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "The OpenPI server does not expose the required SAFE Pi0 "
            f"pre-velocity protocol: {mismatches}. Restart scripts/pi0/serve.py "
            "with --safe-pre-velocity-latent."
        )
    pred_horizon = metadata.get("safe_latent_pred_horizon")
    if (
        isinstance(pred_horizon, bool)
        or not isinstance(pred_horizon, (int, np.integer))
        or int(pred_horizon) < 2
    ):
        raise RuntimeError(
            "The OpenPI server SAFE latent prediction horizon is invalid: "
            f"{pred_horizon!r}"
        )


def _validate_safe_latent(value: Any) -> np.ndarray:
    latent = np.asarray(value)
    if latent.shape != (SAFE_LATENT_DIM,):
        raise ValueError(
            f"OpenPI SAFE latent shape {latent.shape} != {(SAFE_LATENT_DIM,)}"
        )
    if latent.dtype != np.dtype(np.float32):
        raise TypeError(f"OpenPI SAFE latent must be float32, got {latent.dtype}")
    if not np.all(np.isfinite(latent)):
        raise ValueError("OpenPI SAFE latent contains NaN or Inf")
    return latent.copy()


@dataclass
class RemoteWebsocketChunkPolicy:
    """
    Chunked-action policy client for openpi websocket server (serve_policy.py).

    Requires `openpi-client` to be installed in the (robot / ManiSkill) environment:
      pip install -e /path/to/openpi/packages/openpi-client
    """

    server: str  # e.g. "ws://localhost:8000"
    obs_adapter: ObsAdapter
    act_dim: int = 8
    resize: int = 224
    require_frozen_latent: bool = False
    require_safe_latent: bool = False

    def __post_init__(self) -> None:
        try:
            from openpi_client import image_tools
            from openpi_client import websocket_client_policy
        except Exception as e:  # pragma: no cover
            repo_root = Path(__file__).resolve().parents[3]
            client_pkg = repo_root / "third_party" / "openpi" / "packages" / "openpi-client"
            raise RuntimeError(
                "Remote policy requires openpi-client. Install with: "
                f"`pip install -e {client_pkg}`"
            ) from e

        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host=self.server, port=None)
        self._server_metadata = dict(self._client.get_server_metadata())
        if self.require_frozen_latent and self.require_safe_latent:
            raise ValueError("Only one OpenPI latent protocol may be required")
        if self.require_frozen_latent:
            _validate_frozen_latent_metadata(self._server_metadata)
        if self.require_safe_latent:
            _validate_safe_latent_metadata(self._server_metadata)
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None
        self._last_frozen_latent: np.ndarray | None = None

    def reset(self) -> None:
        self._queue.clear()
        self._last_action = None
        self._last_frozen_latent = None

    @property
    def server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def _preprocess_images(self, example: dict) -> dict:
        # Ensure uint8 HWC, then resize_with_pad to reduce bandwidth.
        base = _to_uint8_hwc(example["observation/image"])
        wrist = _to_uint8_hwc(example["observation/wrist_image"])
        base = self._image_tools.resize_with_pad(base, self.resize, self.resize)
        wrist = self._image_tools.resize_with_pad(wrist, self.resize, self.resize)
        base = self._image_tools.convert_to_uint8(base)
        wrist = self._image_tools.convert_to_uint8(wrist)
        example = dict(example)
        example["observation/image"] = base
        example["observation/wrist_image"] = wrist
        return example

    def infer_preprocessed(
        self,
        example: dict,
        *,
        inference_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Infer from an OpenPI-format example without using the ManiSkill adapter."""

        request = self._preprocess_images(example)
        if inference_seed is not None:
            if (
                self._server_metadata.get("inference_seed_protocol")
                != INFERENCE_SEED_CAPABILITY
            ):
                raise RuntimeError(
                    "The OpenPI server does not support deterministic per-request "
                    "inference seeds. Restart it with this repository's "
                    "scripts/pi0/serve.py before deterministic rollout or paired eval."
                )
            request[INFERENCE_SEED_KEY] = int(inference_seed)
        out: dict[str, Any] = self._client.infer(request)
        latent_key = SAFE_LATENT_KEY if self.require_safe_latent else FROZEN_LATENT_KEY
        latent_value = out.get(latent_key)
        latent: np.ndarray | None
        if latent_value is None:
            if self.require_frozen_latent or self.require_safe_latent:
                raise RuntimeError(
                    f"OpenPI response is missing required {latent_key!r}"
                )
            latent = None
        elif self.require_safe_latent:
            latent = _validate_safe_latent(latent_value)
        else:
            latent = _validate_frozen_latent(latent_value)
        chunk = np.asarray(out["actions"])
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk [H, D], got shape={chunk.shape}")
        if chunk.shape[1] != self.act_dim:
            raise ValueError(
                f"Remote policy returned action_dim={chunk.shape[1]}, "
                f"but the current control mode requires exactly {self.act_dim}. "
                "Use a checkpoint trained for this control mode or override --control-mode."
            )
        chunk = chunk.astype(np.float32, copy=False)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Remote policy returned NaN or Inf actions")
        return chunk, latent

    def act(self, obs: dict, *, inference_seed: int | None = None) -> np.ndarray:
        if not self._queue:
            example = self.obs_adapter(obs)
            chunk, self._last_frozen_latent = self.infer_preprocessed(
                example, inference_seed=inference_seed
            )
            for a in chunk:
                self._queue.append(a)
        self._last_action = self._queue.popleft()
        return self._last_action

    def planned_latent(self) -> np.ndarray | None:
        """Return the requested latent paired with the current action chunk."""

        if self._last_frozen_latent is None:
            return None
        return self._last_frozen_latent.copy()

    def planned_chunk(self, *, include_current: bool = True) -> np.ndarray | None:
        """
        Return the currently executing action chunk.

        After `act()` has been called, this is the action just returned plus the
        remaining queued actions. It is useful for visualizing an open-loop
        chunked policy without issuing another inference request.
        """
        parts: list[np.ndarray] = []
        if include_current and self._last_action is not None:
            parts.append(np.asarray(self._last_action, dtype=np.float32))
        parts.extend(np.asarray(a, dtype=np.float32) for a in self._queue)
        if not parts:
            return None
        return np.stack(parts, axis=0)
