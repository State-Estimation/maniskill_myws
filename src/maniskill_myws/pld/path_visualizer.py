from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .state import as_numpy, get_by_path_flexible, squeeze_leading_batch


def parse_rgba(text: str | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(text, str):
        parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    else:
        parts = [float(x) for x in text]
    if len(parts) == 3:
        parts.append(1.0)
    if len(parts) != 4:
        raise ValueError("Expected color as r,g,b or r,g,b,a")
    return tuple(parts)  # type: ignore[return-value]


def tcp_position_from_obs(obs: dict, tcp_pose_key: str) -> np.ndarray:
    tcp = as_numpy(get_by_path_flexible(obs, tcp_pose_key))
    tcp = squeeze_leading_batch(tcp).reshape(-1)
    if tcp.shape[0] < 3:
        raise ValueError(f"TCP pose key '{tcp_pose_key}' has fewer than 3 values")
    return tcp[:3].astype(np.float32, copy=False)


def predict_tcp_positions_from_action_chunk(
    obs: dict,
    action_chunk: np.ndarray,
    *,
    tcp_pose_key: str,
    position_scale: float,
    max_actions: int,
) -> np.ndarray:
    start = tcp_position_from_obs(obs, tcp_pose_key)
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected action chunk [H, D>=3], got shape={actions.shape}")
    if max_actions > 0:
        actions = actions[:max_actions]
    deltas = actions[:, :3] * float(position_scale)
    return start[None, :] + np.cumsum(deltas, axis=0)


@dataclass
class TCPPathVisualizer:
    """
    Draw TCP path samples as small kinematic visual spheres in the SAPIEN scene.

    The visualizer uses a fixed marker pool so it works in both CPU and GPU
    ManiSkill modes without removing actors during training.
    """

    env: Any
    max_points: int = 500
    radius: float = 0.008
    base_color: tuple[float, float, float, float] = (0.05, 0.35, 1.0, 1.0)
    residual_color: tuple[float, float, float, float] = (1.0, 0.28, 0.02, 1.0)
    tcp_pose_key: str = "extra/tcp_pose"
    _base_markers: list[Any] = field(default_factory=list, init=False)
    _residual_markers: list[Any] = field(default_factory=list, init=False)
    _base_count: int = 0
    _residual_count: int = 0
    _disabled: bool = False

    def clear(self) -> None:
        self._ensure_pool()
        for actor in self._base_markers + self._residual_markers:
            try:
                actor.hide_visual()
            except Exception:
                pass
        self._base_count = 0
        self._residual_count = 0

    def clear_base_prediction(self) -> None:
        self._ensure_pool()
        for actor in self._base_markers[: self._base_count]:
            try:
                actor.hide_visual()
            except Exception:
                pass
        self._base_count = 0

    def add_from_obs(self, obs: dict, source: str) -> None:
        if self._disabled:
            return
        try:
            pos = tcp_position_from_obs(obs, self.tcp_pose_key)
            self.add_point(pos, source)
        except Exception as e:
            self._disabled = True
            print(f"tcp path visualization disabled: {e}")

    def set_base_prediction_from_chunk(
        self,
        obs: dict,
        action_chunk: np.ndarray | None,
        *,
        position_scale: float,
        max_actions: int,
    ) -> None:
        self.clear_base_prediction()
        if self._disabled or action_chunk is None:
            return
        try:
            points = predict_tcp_positions_from_action_chunk(
                obs,
                action_chunk,
                tcp_pose_key=self.tcp_pose_key,
                position_scale=position_scale,
                max_actions=max_actions,
            )
            for xyz in points:
                self.add_point(xyz, "base")
        except Exception as e:
            self._disabled = True
            print(f"base chunk visualization disabled: {e}")

    def add_point(self, xyz: np.ndarray, source: str) -> None:
        self._ensure_pool()
        if source == "base":
            markers = self._base_markers
            idx = self._base_count
            self._base_count += 1
        elif source == "residual":
            markers = self._residual_markers
            idx = self._residual_count
            self._residual_count += 1
        else:
            raise ValueError(f"Unknown path source: {source}")

        if idx >= len(markers):
            return
        marker = markers[idx]
        self._set_marker_pose(marker, xyz)

    def show_used(self) -> None:
        for actor in self._base_markers[: self._base_count]:
            try:
                actor.show_visual()
            except Exception:
                pass
        for actor in self._residual_markers[: self._residual_count]:
            try:
                actor.show_visual()
            except Exception:
                pass

    def hide_used(self) -> None:
        for actor in self._base_markers[: self._base_count]:
            try:
                actor.hide_visual()
            except Exception:
                pass
        for actor in self._residual_markers[: self._residual_count]:
            try:
                actor.hide_visual()
            except Exception:
                pass

    def _ensure_pool(self) -> None:
        if self._disabled or self._base_markers:
            return
        try:
            self._base_markers = self._build_marker_pool("base", self.base_color)
            self._residual_markers = self._build_marker_pool("residual", self.residual_color)
            for actor in self._base_markers + self._residual_markers:
                actor.hide_visual()
        except Exception as e:
            self._disabled = True
            print(f"tcp path visualization disabled: {e}")

    def _build_marker_pool(
        self, source: str, color: tuple[float, float, float, float]
    ) -> list[Any]:
        import sapien

        scene = self.env.unwrapped.scene
        material = sapien.render.RenderMaterial(base_color=list(color))
        markers = []
        for i in range(int(self.max_points)):
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(radius=float(self.radius), material=material)
            actor = builder.build_kinematic(name=f"pld_tcp_path_{source}_{i:04d}")
            markers.append(actor)
        return markers

    def _set_marker_pose(self, actor: Any, xyz: np.ndarray) -> None:
        import sapien

        pose = sapien.Pose(p=np.asarray(xyz, dtype=np.float32).reshape(3).tolist())
        actor.set_pose(pose)
