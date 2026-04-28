from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .obs_to_openpi import ObsAdapter, _to_uint8_hwc


@dataclass
class RemoteWebsocketChunkPolicy:
    """
    Chunked-action policy client for openpi websocket server (serve_policy.py).

    Requires `openpi-client` to be installed in the (robot / ManiSkill) environment:
      pip install -e /path/to/openpi/packages/openpi-client
    """

    server: str  # e.g. "ws://localhost:8000"
    obs_adapter: ObsAdapter
    act_dim: int = 7
    resize: int = 224

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
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None

    def reset(self) -> None:
        self._queue.clear()
        self._last_action = None

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

    def act(self, obs: dict) -> np.ndarray:
        if not self._queue:
            example = self.obs_adapter(obs)
            example = self._preprocess_images(example)
            out: dict[str, Any] = self._client.infer(example)
            chunk = np.asarray(out["actions"])
            if chunk.ndim != 2:
                raise ValueError(f"Expected action chunk [H, D], got shape={chunk.shape}")
            chunk = chunk[:, : self.act_dim].astype(np.float32, copy=False)
            for a in chunk:
                self._queue.append(a)
        self._last_action = self._queue.popleft()
        return self._last_action

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
