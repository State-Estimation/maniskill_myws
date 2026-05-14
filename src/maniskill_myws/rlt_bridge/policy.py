from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from maniskill_myws.openpi_bridge.obs_to_openpi import ObsAdapter

from .clients import ActorClient, ActorResult, FeatureClient, RLTFeatures, SOURCE_BASE


@dataclass(frozen=True)
class RLTPlan:
    features: RLTFeatures
    action_chunk: np.ndarray
    actor_result: ActorResult
    episode_id: int
    step_id: int
    request_id: str


class RLTOnlineChunkPolicy:
    """Minimal ManiSkill policy bridge for openpi-RLT online RL.

    It requests RLT features from `serve_rlt_policy.py`, optionally asks the
    openpi-RLT actor service for a refined chunk, and returns one action per env
    step from that chunk.
    """

    def __init__(
        self,
        *,
        feature_server: str,
        obs_adapter: ObsAdapter,
        actor_url: str | None = None,
        act_dim: int = 7,
        resize: int = 224,
        z_dim: int = 2048,
        proprio_dim: int = 7,
        chunk_len: int = 10,
        actor_timeout_sec: float = 1.0,
        actor_deterministic: bool = True,
        fallback_to_ref: bool = True,
    ) -> None:
        self.obs_adapter = obs_adapter
        self.act_dim = int(act_dim)
        self.chunk_len = int(chunk_len)
        self.actor_deterministic = bool(actor_deterministic)
        self.fallback_to_ref = bool(fallback_to_ref)
        self.feature_client = FeatureClient(
            feature_server,
            resize=resize,
            z_dim=z_dim,
            proprio_dim=proprio_dim,
            chunk_len=chunk_len,
            action_dim=act_dim,
        )
        self.actor_client = None if actor_url is None else ActorClient(actor_url, timeout_sec=actor_timeout_sec)
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None
        self._last_plan: RLTPlan | None = None
        self._episode_id = 0
        self._request_counter = 0

    def reset(self, *, episode_id: int = 0) -> None:
        self._queue.clear()
        self._last_action = None
        self._last_plan = None
        self._episode_id = int(episode_id)
        self._request_counter = 0

    @property
    def last_plan(self) -> RLTPlan | None:
        return self._last_plan

    def has_pending_actions(self) -> bool:
        return bool(self._queue)

    def features_for_obs(self, obs: dict[str, Any]) -> RLTFeatures:
        example = self.obs_adapter(obs)
        return self.feature_client.infer(example)

    def plan(self, obs: dict[str, Any], *, step_id: int) -> RLTPlan:
        features = self.features_for_obs(obs)
        request_id = f"ms-{self._episode_id}-{step_id}-{self._request_counter}"
        self._request_counter += 1

        actor_result = self._refine(features, request_id=request_id, step_id=step_id)
        action_chunk = np.asarray(actor_result.refined_chunk, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected refined chunk [T, A], got shape={action_chunk.shape}")
        action_chunk = action_chunk[: self.chunk_len, : self.act_dim]
        return RLTPlan(
            features=features,
            action_chunk=action_chunk,
            actor_result=actor_result,
            episode_id=self._episode_id,
            step_id=int(step_id),
            request_id=request_id,
        )

    def act(self, obs: dict[str, Any], *, step_id: int) -> np.ndarray:
        if not self._queue:
            plan = self.plan(obs, step_id=step_id)
            self._last_plan = plan
            for action in plan.action_chunk:
                self._queue.append(np.asarray(action, dtype=np.float32))
        self._last_action = self._queue.popleft()
        return self._last_action

    def planned_chunk(self, *, include_current: bool = True) -> np.ndarray | None:
        parts: list[np.ndarray] = []
        if include_current and self._last_action is not None:
            parts.append(np.asarray(self._last_action, dtype=np.float32))
        parts.extend(np.asarray(action, dtype=np.float32) for action in self._queue)
        if not parts:
            return None
        return np.stack(parts, axis=0)

    def _refine(self, features: RLTFeatures, *, request_id: str, step_id: int) -> ActorResult:
        if self.actor_client is None:
            return ActorResult(
                refined_chunk=np.asarray(features.ref_chunk, dtype=np.float32),
                actor_param_version=-1,
                source=SOURCE_BASE,
            )
        try:
            return self.actor_client.infer(
                features,
                request_id=request_id,
                episode_id=self._episode_id,
                step_id=step_id,
                deterministic=self.actor_deterministic,
            )
        except RuntimeError as exc:
            if not self.fallback_to_ref:
                raise
            return ActorResult(
                refined_chunk=np.asarray(features.ref_chunk, dtype=np.float32),
                actor_param_version=-1,
                source=SOURCE_BASE,
                used_fallback=True,
                error=str(exc),
            )

    def actor_version(self) -> int:
        if self.actor_client is None:
            return -1
        try:
            return self.actor_client.version()
        except Exception:
            return -1

    def now(self) -> float:
        return time.time()
