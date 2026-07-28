from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Final, Sequence

import numpy as np


BASE_ACTION_PROJECTION: Final = "clip_to_env_action_space_v1"


def project_action_chunk_to_bounds(
    value: object,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    expected_shape: tuple[int, int],
    name: str = "Base action chunk",
) -> tuple[np.ndarray, dict[str, object]]:
    """Validate and project one OpenPI chunk onto the environment action Box."""

    raw = np.asarray(value)
    if raw.shape != expected_shape:
        raise ValueError(f"{name} shape {raw.shape} != expected {expected_shape}")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype={raw.dtype}")
    action = raw.astype(np.float32, copy=False)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    if low.shape != (expected_shape[1],) or high.shape != (expected_shape[1],):
        raise ValueError(
            f"Action bounds must have shape {(expected_shape[1],)}, got "
            f"{low.shape}/{high.shape}"
        )
    if (
        not np.all(np.isfinite(action))
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
    ):
        raise ValueError(f"{name} or its action bounds contain NaN or Inf")
    if np.any(low >= high):
        raise ValueError("Every action lower bound must be below its upper bound")

    projected = np.clip(action, low, high).astype(np.float32, copy=False)
    correction = np.abs(projected - action)
    clipped = correction > 0.0
    clipped_actions = np.any(clipped, axis=1)
    clipped_correction = correction[clipped]
    report: dict[str, object] = {
        "chunks": 1,
        "actions": int(action.shape[0]),
        "values": int(action.size),
        "clipped_actions": int(np.count_nonzero(clipped_actions)),
        "clipped_values": int(np.count_nonzero(clipped)),
        "correction_sum": float(np.sum(correction, dtype=np.float64)),
        "max_abs_correction": float(np.max(correction, initial=0.0)),
        "max_lower_violation": float(
            np.max(np.maximum(low - action, 0.0), initial=0.0)
        ),
        "max_upper_violation": float(
            np.max(np.maximum(action - high, 0.0), initial=0.0)
        ),
        "mean_clipped_correction": (
            float(np.mean(clipped_correction, dtype=np.float64))
            if clipped_correction.size
            else 0.0
        ),
        "clipped_values_by_dim": np.count_nonzero(clipped, axis=0)
        .astype(np.int64)
        .tolist(),
        "max_abs_correction_by_dim": np.max(correction, axis=0).astype(float).tolist(),
    }
    return projected, report


def inference_seed_for_step(episode_seed: int, step: int, *, stream: int = 0) -> int:
    """Stable uint32 OpenPI diffusion seed for a macro decision."""

    if min(int(episode_seed), int(step), int(stream)) < 0:
        raise ValueError("episode_seed, step, and stream must be non-negative")
    sequence = np.random.SeedSequence([int(episode_seed), int(step), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def metadata_sha256(value: object) -> str:
    """Stable hash for nested OpenPI server metadata, including numpy arrays."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}[".encode("ascii"))
            for entry in item:
                update(entry)
            digest.update(b"]")
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"array:{array.dtype.str}:{array.shape}:".encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, np.generic):
            update(item.item())
        elif isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(f"scalar:{type(item).__name__}:{item!r}".encode("utf-8"))
        else:
            raise TypeError(
                f"Unsupported server metadata value for hashing: {type(item)}"
            )

    update(value)
    return digest.hexdigest()


def openpi_policy_identity_sha256(metadata: Mapping[str, object]) -> str:
    """Hash stable OpenPI policy content while ignoring launch-path spelling."""

    raw_identity = metadata.get("maniskill_policy_identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("OpenPI metadata has no maniskill_policy_identity mapping")
    checkpoint_sha = raw_identity.get("checkpoint_content_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("OpenPI metadata has no valid checkpoint content SHA-256")
    stable_identity = {
        "schema": "openpi_policy_semantic_identity_v1",
        "config": raw_identity.get("config"),
        "repo_id": raw_identity.get("repo_id"),
        "default_prompt": raw_identity.get("default_prompt"),
        "checkpoint_content_sha256": checkpoint_sha,
        "norm_stats_content_sha256": raw_identity.get("norm_stats_content_sha256"),
        "inference_seed_protocol": metadata.get("inference_seed_protocol"),
        "frozen_latent_protocol": metadata.get("frozen_latent_protocol"),
        "frozen_latent_key": metadata.get("frozen_latent_key"),
        "frozen_latent_shape": metadata.get("frozen_latent_shape"),
        "frozen_latent_dtype": metadata.get("frozen_latent_dtype"),
        "frozen_latent_source": metadata.get("frozen_latent_source"),
        "frozen_latent_pooling": metadata.get("frozen_latent_pooling"),
    }
    return metadata_sha256(stable_identity)


class BaseChunkPolicy:
    def reset(self) -> None:
        pass

    @property
    def server_metadata(self) -> dict | None:
        return None

    @property
    def action_projection(self) -> str | None:
        return None

    @property
    def action_projection_stats(self) -> dict[str, object]:
        return {}

    def plan(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def plan_with_latent(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return (
            self.plan(
                obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                inference_seed=inference_seed,
            ),
            None,
        )

    def plan_with_temporal_latent(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        chunk, latent = self.plan_with_latent(
            obs,
            chunk_len=chunk_len,
            action_dim=action_dim,
            inference_seed=inference_seed,
        )
        return chunk, latent, None

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
        action_low: np.ndarray,
        action_high: np.ndarray,
        resize: int = 224,
        require_frozen_latent: bool = False,
        require_frozen_temporal_latent: bool = False,
    ) -> None:
        from maniskill_myws.openpi_bridge.obs_to_openpi import ObsAdapter
        from maniskill_myws.openpi_bridge.remote_policy import RemoteWebsocketChunkPolicy

        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)
        if self._action_low.shape != (action_dim,) or self._action_high.shape != (
            action_dim,
        ):
            raise ValueError(
                f"Remote OpenPI action bounds must have shape {(action_dim,)}, got "
                f"{self._action_low.shape}/{self._action_high.shape}"
            )
        if (
            not np.all(np.isfinite(self._action_low))
            or not np.all(np.isfinite(self._action_high))
            or np.any(self._action_low >= self._action_high)
        ):
            raise ValueError("Remote OpenPI action bounds must be finite with low < high")
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
            require_frozen_latent=require_frozen_latent,
            require_frozen_temporal_latent=require_frozen_temporal_latent,
        )
        self._projection_totals: dict[str, object] = {
            "chunks": 0,
            "actions": 0,
            "values": 0,
            "clipped_actions": 0,
            "clipped_values": 0,
            "correction_sum": 0.0,
            "max_abs_correction": 0.0,
            "max_lower_violation": 0.0,
            "max_upper_violation": 0.0,
            "clipped_values_by_dim": [0] * action_dim,
            "max_abs_correction_by_dim": [0.0] * action_dim,
        }

    def reset(self) -> None:
        self.policy.reset()

    @property
    def server_metadata(self) -> dict:
        return self.policy.server_metadata

    @property
    def action_projection(self) -> str:
        return BASE_ACTION_PROJECTION

    @property
    def action_projection_stats(self) -> dict[str, object]:
        totals = {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._projection_totals.items()
        }
        values = int(totals["values"])
        actions = int(totals["actions"])
        clipped_values = int(totals["clipped_values"])
        clipped_actions = int(totals["clipped_actions"])
        totals.update(
            value_clip_rate=clipped_values / values if values else 0.0,
            action_clip_rate=clipped_actions / actions if actions else 0.0,
            mean_abs_correction=(
                float(totals["correction_sum"]) / values if values else 0.0
            ),
            mean_clipped_correction=(
                float(totals["correction_sum"]) / clipped_values
                if clipped_values
                else 0.0
            ),
        )
        return totals

    def _record_projection(self, report: Mapping[str, object]) -> None:
        for key in (
            "chunks",
            "actions",
            "values",
            "clipped_actions",
            "clipped_values",
        ):
            self._projection_totals[key] = int(self._projection_totals[key]) + int(
                report[key]
            )
        self._projection_totals["correction_sum"] = float(
            self._projection_totals["correction_sum"]
        ) + float(report["correction_sum"])
        for key in (
            "max_abs_correction",
            "max_lower_violation",
            "max_upper_violation",
        ):
            self._projection_totals[key] = max(
                float(self._projection_totals[key]), float(report[key])
            )
        clipped_by_dim = report["clipped_values_by_dim"]
        max_by_dim = report["max_abs_correction_by_dim"]
        if not isinstance(clipped_by_dim, list) or not isinstance(max_by_dim, list):
            raise TypeError("Projection report per-dimension statistics must be lists")
        previous_counts = self._projection_totals["clipped_values_by_dim"]
        previous_max = self._projection_totals["max_abs_correction_by_dim"]
        if not isinstance(previous_counts, list) or not isinstance(previous_max, list):
            raise TypeError("Projection accumulator per-dimension statistics are invalid")
        if not (
            len(clipped_by_dim)
            == len(max_by_dim)
            == len(previous_counts)
            == len(previous_max)
        ):
            raise ValueError("Projection report action dimension changed between chunks")
        self._projection_totals["clipped_values_by_dim"] = [
            int(total) + int(current)
            for total, current in zip(previous_counts, clipped_by_dim, strict=True)
        ]
        self._projection_totals["max_abs_correction_by_dim"] = [
            max(float(total), float(current))
            for total, current in zip(previous_max, max_by_dim, strict=True)
        ]

    def plan(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> np.ndarray:
        self.policy.reset()
        _first_action = self.policy.act(obs, inference_seed=inference_seed)
        planned = self.policy.planned_chunk(include_current=True)
        if planned is None:
            raise RuntimeError("Remote OpenPI policy returned no action chunk")
        planned = np.asarray(planned, dtype=np.float32)
        if planned.ndim != 2 or planned.shape[1] != action_dim:
            raise ValueError(
                f"Remote OpenPI chunk shape {planned.shape} is incompatible with "
                f"the exact [H,{action_dim}] action schema"
            )
        if planned.shape[0] < chunk_len:
            raise ValueError(
                f"Remote OpenPI returned only {planned.shape[0]} actions, but the "
                f"RLT macro action requires {chunk_len}; repeating actions would "
                "change the policy semantics"
            )
        projected, report = project_action_chunk_to_bounds(
            planned[:chunk_len],
            action_low=self._action_low,
            action_high=self._action_high,
            expected_shape=(chunk_len, action_dim),
            name="Remote OpenPI reference chunk",
        )
        self._record_projection(report)
        return projected

    def plan_with_latent(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        chunk = self.plan(
            obs,
            chunk_len=chunk_len,
            action_dim=action_dim,
            inference_seed=inference_seed,
        )
        latent = self.policy.planned_latent()
        if latent is None:
            raise RuntimeError(
                "Remote OpenPI policy did not return a frozen action latent"
            )
        return chunk, latent

    def plan_with_temporal_latent(
        self,
        obs: dict,
        *,
        chunk_len: int,
        action_dim: int,
        inference_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk, latent = self.plan_with_latent(
            obs,
            chunk_len=chunk_len,
            action_dim=action_dim,
            inference_seed=inference_seed,
        )
        temporal_latent = self.policy.planned_temporal_latent()
        if temporal_latent is None:
            raise RuntimeError(
                "Remote OpenPI policy did not return a frozen temporal action latent"
            )
        return chunk, latent, temporal_latent

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
    require_frozen_latent: bool = False,
    require_frozen_temporal_latent: bool = False,
) -> BaseChunkPolicy:
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
            action_low=np.asarray(action_space.low, dtype=np.float32),
            action_high=np.asarray(action_space.high, dtype=np.float32),
            resize=resize,
            require_frozen_latent=require_frozen_latent,
            require_frozen_temporal_latent=require_frozen_temporal_latent,
        )
    raise ValueError(
        f"Unsupported base chunk policy {kind!r}; only 'remote_openpi' is maintained"
    )
