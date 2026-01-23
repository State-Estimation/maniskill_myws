from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .keypath import get_by_path


def _as_numpy(x: Any) -> np.ndarray:
    """Convert input to numpy, supporting torch tensors (including CUDA tensors)."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _squeeze_leading_batch(arr: np.ndarray) -> np.ndarray:
    """
    ManiSkill often returns observations with a leading num_envs dimension, even when num_envs=1.
    For this client we assume num_envs==1 and squeeze that leading dim.
    """
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """
    Best-effort conversion for ManiSkill / LeRobot image conventions.
    Accepts:
      - uint8 HWC
      - float HWC in [0,1]
      - CHW (3,H,W) variants
    Returns uint8 HWC.
    """
    arr = _as_numpy(image)
    arr = _squeeze_leading_batch(arr)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected image with 3 dims (H,W,C) or (1,H,W,C), got shape={arr.shape}. "
            "If you're running multiple envs, select an env index before converting."
        )
    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


@dataclass(frozen=True)
class ObsAdapter:
    """
    Convert ManiSkill observation dict -> openpi policy input example.

    We intentionally produce the same keys as openpi's LIBERO policy expects:
      - observation/image
      - observation/wrist_image
      - observation/state
      - prompt
    """

    image_key: str
    wrist_image_key: str
    state_keys: Sequence[str]
    prompt: str

    def __call__(self, obs: dict) -> dict:
        base_img = _to_uint8_hwc(get_by_path(obs, self.image_key))
        wrist_img = _to_uint8_hwc(get_by_path(obs, self.wrist_image_key))
        parts: list[np.ndarray] = []
        for key in self.state_keys:
            state_arr = _as_numpy(get_by_path(obs, key))
            state_arr = _squeeze_leading_batch(state_arr)
            parts.append(state_arr.astype(np.float32, copy=False).reshape(-1))
        if not parts:
            raise ValueError("ObsAdapter.state_keys is empty")
        state = np.concatenate(parts, axis=0)
        return {
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": self.prompt,
        }


