from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .replay import pad_or_trim_chunk


class BaseChunkPolicy:
    def reset(self) -> None:
        pass

    def plan(self, obs: dict, *, chunk_len: int, action_dim: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ZeroChunkPolicy(BaseChunkPolicy):
    def plan(self, obs: dict, *, chunk_len: int, action_dim: int) -> np.ndarray:
        return np.zeros((chunk_len, action_dim), dtype=np.float32)


@dataclass
class RandomChunkPolicy(BaseChunkPolicy):
    action_space: object

    def plan(self, obs: dict, *, chunk_len: int, action_dim: int) -> np.ndarray:
        actions = [
            np.asarray(self.action_space.sample(), dtype=np.float32).reshape(-1)[:action_dim]
            for _ in range(chunk_len)
        ]
        return np.stack(actions, axis=0)


class RemoteOpenPIChunkPolicy(BaseChunkPolicy):
    """Query OpenPI for a fresh reference action chunk at each RLT chunk boundary."""

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
            server=server,
            obs_adapter=adapter,
            act_dim=action_dim,
            resize=resize,
        )

    def reset(self) -> None:
        self.policy.reset()

    def plan(self, obs: dict, *, chunk_len: int, action_dim: int) -> np.ndarray:
        self.policy.reset()
        _first_action = self.policy.act(obs)
        planned = self.policy.planned_chunk(include_current=True)
        if planned is None:
            raise RuntimeError("Remote OpenPI policy returned no action chunk")
        return pad_or_trim_chunk(planned, chunk_len=chunk_len, action_dim=action_dim)


def make_base_chunk_policy(
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
) -> BaseChunkPolicy:
    if kind == "zero":
        return ZeroChunkPolicy()
    if kind == "random":
        return RandomChunkPolicy(action_space=action_space)
    if kind == "remote_openpi":
        if not server:
            raise ValueError("--server is required when --base-policy remote_openpi")
        return RemoteOpenPIChunkPolicy(
            server=server,
            prompt=prompt,
            image_key=image_key,
            wrist_image_key=wrist_image_key,
            state_keys=state_keys,
            action_dim=action_dim,
            resize=resize,
        )
    raise ValueError(f"Unknown base chunk policy kind: {kind}")
