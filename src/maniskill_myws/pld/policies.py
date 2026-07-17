from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class BasePolicy:
    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def planned_chunk(self) -> np.ndarray | None:
        return None

    def plan_window(self, obs: dict, *, window_steps: int) -> np.ndarray:
        """Plan one fresh open-loop rollout window from the current observation."""
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        actions = [self.act(obs) for _ in range(window_steps)]
        return np.stack(actions, axis=0).astype(np.float32, copy=False)


@dataclass
class ZeroBasePolicy(BasePolicy):
    action_dim: int
    _last_action: np.ndarray | None = None

    def act(self, obs: dict) -> np.ndarray:
        self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
        return self._last_action

    def reset(self) -> None:
        self._last_action = None

    def planned_chunk(self) -> np.ndarray | None:
        if self._last_action is None:
            return None
        return self._last_action[None, :]

    def plan_window(self, obs: dict, *, window_steps: int) -> np.ndarray:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self._last_action = np.zeros((self.action_dim,), dtype=np.float32)
        return np.repeat(self._last_action[None, :], window_steps, axis=0)


@dataclass
class RandomBasePolicy(BasePolicy):
    action_space: object
    _last_action: np.ndarray | None = None

    def act(self, obs: dict) -> np.ndarray:
        self._last_action = np.asarray(self.action_space.sample(), dtype=np.float32).reshape(-1)
        return self._last_action

    def reset(self) -> None:
        self._last_action = None

    def planned_chunk(self) -> np.ndarray | None:
        if self._last_action is None:
            return None
        return self._last_action[None, :]

    def plan_window(self, obs: dict, *, window_steps: int) -> np.ndarray:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        actions = [
            np.asarray(self.action_space.sample(), dtype=np.float32).reshape(-1)
            for _ in range(window_steps)
        ]
        self._last_action = actions[-1]
        return np.stack(actions, axis=0)


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

    def planned_chunk(self) -> np.ndarray | None:
        return self.policy.planned_chunk(include_current=True)

    def plan_window(self, obs: dict, *, window_steps: int) -> np.ndarray:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        # Discard any stale tail produced from an older observation and request
        # exactly one fresh OpenPI inference at this rollout-window boundary.
        self.policy.reset()
        self.policy.act(obs)
        planned = self.policy.planned_chunk(include_current=True)
        if planned is None:
            raise RuntimeError("Remote OpenPI policy returned no action chunk")
        if planned.shape[0] < window_steps:
            raise ValueError(
                f"OpenPI action chunk has {planned.shape[0]} steps, but rollout window "
                f"requires {window_steps}. Reduce --rollout-window-steps."
            )
        return np.asarray(planned[:window_steps], dtype=np.float32)


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
