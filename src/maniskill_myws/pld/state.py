from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    """Convert torch/numpy-like values to a CPU numpy array."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def squeeze_leading_batch(arr: np.ndarray) -> np.ndarray:
    """ManiSkill single-env observations often carry a leading batch dimension."""
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def resize_hwc_nearest(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return image
    size = int(size)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, size).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def prepare_rgb_image(image: np.ndarray, image_size: int | None = None) -> np.ndarray:
    arr = squeeze_leading_batch(as_numpy(image))
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {arr.shape}")
    if arr.shape[-1] not in {1, 3, 4} and arr.shape[0] in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    arr = resize_hwc_nearest(arr, image_size)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def get_by_path_flexible(d: Mapping[str, Any], path: str, *, sep: str = "/") -> Any:
    """
    Read nested values using slash paths.

    The same state keys are convenient for both RecordEpisode H5 groups
    ("obs/agent/qpos") and live ManiSkill observations ("agent/qpos"), so this
    helper strips a leading "obs/" when the live observation is already rooted at
    the observation dict.
    """
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
    """Flatten a small set of ManiSkill observation entries into a 1D state."""

    state_keys: Sequence[str]

    def __call__(self, obs: Mapping[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.state_keys:
            arr = as_numpy(get_by_path_flexible(obs, key))
            arr = squeeze_leading_batch(arr)
            parts.append(arr.astype(np.float32, copy=False).reshape(-1))
        if not parts:
            raise ValueError("StateAdapter.state_keys is empty")
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ImageAdapter:
    """Extract one or more RGB observations as a stacked uint8 image tensor."""

    image_keys: Sequence[str]
    image_size: int | None = None
    image_shape: tuple[int, ...] | None = None

    def __call__(self, obs: Mapping[str, Any]) -> np.ndarray:
        images = [
            prepare_rgb_image(get_by_path_flexible(obs, key), self.image_size)
            for key in self.image_keys
        ]
        if not images:
            raise ValueError("ImageAdapter.image_keys is empty")
        stacked = np.stack(images, axis=0)
        if self.image_shape is not None and tuple(stacked.shape) != tuple(self.image_shape):
            raise ValueError(
                f"Image shape mismatch: got {tuple(stacked.shape)}, expected {self.image_shape}"
            )
        return stacked
