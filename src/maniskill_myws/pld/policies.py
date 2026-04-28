from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class BasePolicy:
    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ZeroBasePolicy(BasePolicy):
    action_dim: int

    def act(self, obs: dict) -> np.ndarray:
        return np.zeros((self.action_dim,), dtype=np.float32)


@dataclass
class RandomBasePolicy(BasePolicy):
    action_space: object

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.float32).reshape(-1)


class RemoteOpenPIBasePolicy(BasePolicy):
    def __init__(
        self,
        *,
        server: str,
        prompt: str,
        image_key: str,
        wrist_image_key: str,
        state_keys: Sequence[str],
        action_dim: int,
        resize: int = 224,
    ) -> None:
        from maniskill_myws.openpi_bridge.obs_to_openpi import ObsAdapter
        from maniskill_myws.openpi_bridge.remote_policy import RemoteWebsocketChunkPolicy

        adapter = ObsAdapter(
            image_key=image_key,
            wrist_image_key=wrist_image_key,
            state_keys=state_keys,
            prompt=prompt,
        )
        self.policy = RemoteWebsocketChunkPolicy(
            server=server, obs_adapter=adapter, act_dim=action_dim, resize=resize
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(self.policy.act(obs), dtype=np.float32).reshape(-1)


def make_base_policy(
    kind: str,
    *,
    action_space: object,
    action_dim: int,
    server: str | None = None,
    prompt: str = "",
    image_key: str = "sensor_data/base_camera/rgb",
    wrist_image_key: str = "sensor_data/hand_camera/rgb",
    state_keys: Sequence[str] = ("agent/qpos", "agent/qvel", "extra/tcp_pose"),
    resize: int = 224,
) -> BasePolicy:
    if kind == "zero":
        return ZeroBasePolicy(action_dim=action_dim)
    if kind == "random":
        return RandomBasePolicy(action_space=action_space)
    if kind == "remote_openpi":
        if not server:
            raise ValueError("--server is required when --base-policy remote_openpi")
        return RemoteOpenPIBasePolicy(
            server=server,
            prompt=prompt,
            image_key=image_key,
            wrist_image_key=wrist_image_key,
            state_keys=state_keys,
            action_dim=action_dim,
            resize=resize,
        )
    raise ValueError(f"Unknown base policy kind: {kind}")
