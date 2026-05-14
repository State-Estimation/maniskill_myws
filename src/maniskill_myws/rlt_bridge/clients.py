from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np


SOURCE_BASE = 0
SOURCE_RL = 1
SOURCE_HUMAN = 2
SOURCE_MIXED = 3


@dataclass(frozen=True)
class RLTFeatures:
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    raw: dict[str, Any]


@dataclass(frozen=True)
class ActorResult:
    refined_chunk: np.ndarray
    actor_param_version: int
    source: int
    used_fallback: bool = False
    error: str | None = None


def _require_openpi_client():
    try:
        from openpi_client import image_tools
        from openpi_client import msgpack_numpy
        from openpi_client import websocket_client_policy
    except Exception as e:  # pragma: no cover
        repo_root = Path(__file__).resolve().parents[3]
        client_pkg = repo_root / "third_party" / "openpi" / "packages" / "openpi-client"
        rlt_client_pkg = repo_root / "third_party" / "openpi-RLT" / "packages" / "openpi-client"
        raise RuntimeError(
            "RLT bridge requires openpi-client. Install with: "
            f"`pip install -e {client_pkg}` or `pip install -e {rlt_client_pkg}`."
        ) from e
    return image_tools, msgpack_numpy, websocket_client_policy


def _coerce_feature_vector(name: str, value: Any, expected_dim: int | None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1 or [1, D], got shape={arr.shape}")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got shape={arr.shape}")
    return arr


def _coerce_chunk(name: str, value: Any, *, chunk_len: int | None, action_dim: int | None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or [1, T, A], got shape={arr.shape}")
    if chunk_len is not None:
        arr = arr[:chunk_len]
    if action_dim is not None:
        arr = arr[:, :action_dim]
    return arr.astype(np.float32, copy=False)


class FeatureClient:
    """Client for openpi-RLT `scripts/serve_rlt_policy.py`.

    The server returns `z_rl`, `proprio`, and `ref_chunk`. This class also accepts
    the compatibility `actions` key and treats it as `ref_chunk` if needed.
    """

    def __init__(
        self,
        server: str,
        *,
        resize: int = 224,
        z_dim: int | None = 2048,
        proprio_dim: int | None = 7,
        chunk_len: int | None = 10,
        action_dim: int | None = 7,
    ) -> None:
        image_tools, _, websocket_client_policy = _require_openpi_client()
        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host=server, port=None)
        self._resize = int(resize)
        self._z_dim = z_dim
        self._proprio_dim = proprio_dim
        self._chunk_len = chunk_len
        self._action_dim = action_dim

    def infer(self, example: dict[str, Any]) -> RLTFeatures:
        example = self._preprocess_images(example)
        raw = self._client.infer(example)
        return self._features_from_payload(raw)

    def _preprocess_images(self, example: dict[str, Any]) -> dict[str, Any]:
        from maniskill_myws.openpi_bridge.obs_to_openpi import _to_uint8_hwc

        base = _to_uint8_hwc(example["observation/image"])
        wrist = _to_uint8_hwc(example["observation/wrist_image"])
        base = self._image_tools.resize_with_pad(base, self._resize, self._resize)
        wrist = self._image_tools.resize_with_pad(wrist, self._resize, self._resize)
        example = dict(example)
        example["observation/image"] = self._image_tools.convert_to_uint8(base)
        example["observation/wrist_image"] = self._image_tools.convert_to_uint8(wrist)
        return example

    def _features_from_payload(self, payload: dict[str, Any]) -> RLTFeatures:
        if "z_rl" not in payload:
            raise ValueError("RLT feature server response is missing `z_rl`.")
        ref_value = payload.get("ref_chunk", payload.get("actions", payload.get("_raw_actions")))
        if ref_value is None:
            raise ValueError("RLT feature server response is missing `ref_chunk` or compatible action chunk.")

        z_rl = _coerce_feature_vector("z_rl", payload["z_rl"], self._z_dim)
        if "proprio" in payload:
            proprio = _coerce_feature_vector("proprio", payload["proprio"], self._proprio_dim)
        else:
            proprio = _coerce_feature_vector("proprio", payload.get("state"), self._proprio_dim)
        ref_chunk = _coerce_chunk("ref_chunk", ref_value, chunk_len=self._chunk_len, action_dim=self._action_dim)
        return RLTFeatures(z_rl=z_rl, proprio=proprio, ref_chunk=ref_chunk, raw=payload)


class ActorClient:
    """HTTP client for openpi-RLT `ActorService`."""

    def __init__(self, base_url: str, *, timeout_sec: float = 1.0, max_retries: int = 1) -> None:
        _, msgpack_numpy, _ = _require_openpi_client()
        self._packer = msgpack_numpy.Packer()
        self._msgpack_numpy = msgpack_numpy
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = float(timeout_sec)
        self._max_retries = int(max_retries)

    def infer(
        self,
        features: RLTFeatures,
        *,
        request_id: str,
        episode_id: int,
        step_id: int,
        deterministic: bool = True,
    ) -> ActorResult:
        payload = {
            "z_rl": np.asarray(features.z_rl, dtype=np.float32),
            "proprio": np.asarray(features.proprio, dtype=np.float32),
            "ref_chunk": np.asarray(features.ref_chunk, dtype=np.float32),
            "request_id": request_id,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "deterministic": bool(deterministic),
            "timestamp": time.time(),
        }
        body = self._packer.pack(payload)
        request = urllib_request.Request(
            f"{self._base_url}/infer",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                with urllib_request.urlopen(request, timeout=self._timeout_sec) as response:
                    raw = self._msgpack_numpy.unpackb(response.read())
                return ActorResult(
                    refined_chunk=np.asarray(raw["refined_chunk"], dtype=np.float32),
                    actor_param_version=int(raw["actor_param_version"]),
                    source=int(raw.get("source", SOURCE_RL)),
                )
            except (urllib_error.URLError, TimeoutError) as exc:
                last_error = exc
        raise RuntimeError("actor_service inference failed") from last_error

    def version(self) -> int:
        request = urllib_request.Request(f"{self._base_url}/version", method="GET")
        with urllib_request.urlopen(request, timeout=self._timeout_sec) as response:
            raw = self._msgpack_numpy.unpackb(response.read())
        return int(raw["actor_param_version"])


class ReplayClient:
    """HTTP client for openpi-RLT `ReplayManager`."""

    def __init__(self, base_url: str, *, timeout_sec: float = 30.0) -> None:
        _, msgpack_numpy, _ = _require_openpi_client()
        self._packer = msgpack_numpy.Packer()
        self._msgpack_numpy = msgpack_numpy
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = float(timeout_sec)

    def add_transition(self, transition: dict[str, Any]) -> None:
        self._post("/add", transition)

    def add_transitions(self, transitions: list[dict[str, Any]]) -> None:
        self._post("/extend", {"transitions": transitions})

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        body = self._packer.pack(payload)
        request = urllib_request.Request(
            f"{self._base_url}{path}",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib_request.urlopen(request, timeout=self._timeout_sec) as response:
                return self._msgpack_numpy.unpackb(response.read())
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Replay request failed for {path}") from exc
