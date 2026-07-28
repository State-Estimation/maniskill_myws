from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    """Convert common tensor/array values to a CPU numpy array."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def squeeze_leading_batch(arr: np.ndarray) -> np.ndarray:
    """Squeeze ManiSkill's single-env leading batch dimension."""
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def get_by_path_flexible(d: Mapping[str, Any], path: str, *, sep: str = "/") -> Any:
    """Read nested mappings with slash paths, accepting an optional leading obs/."""
    parts = [p for p in path.strip(sep).split(sep) if p]
    if parts and parts[0] == "obs" and "obs" not in d:
        parts = parts[1:]

    cur: Any = d
    for part in parts:
        if not isinstance(cur, Mapping):
            raise KeyError(f"Path '{path}' failed at '{part}': got {type(cur)}")
        if part not in cur:
            raise KeyError(f"Path '{path}' missing '{part}'. Available keys: {list(cur.keys())}")
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class StateAdapter:
    """Flatten ManiSkill observation entries into the RLT state vector."""

    state_keys: Sequence[str] = ("agent/qpos", "agent/qvel", "extra/tcp_pose")

    def __call__(self, obs: Mapping[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.state_keys:
            arr = squeeze_leading_batch(as_numpy(get_by_path_flexible(obs, key)))
            parts.append(arr.astype(np.float32, copy=False).reshape(-1))
        if not parts:
            raise ValueError("StateAdapter.state_keys is empty")
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
