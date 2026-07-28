"""Pure tensor helpers for the solar-panel brush grasp reward."""

from __future__ import annotations

import torch


def brush_grasp_candidate(
    *,
    left_finger_force: torch.Tensor,
    right_finger_force: torch.Tensor,
    panda_is_grasping: torch.Tensor,
    tcp_in_handle_region: torch.Tensor,
    tcp_brush_distance: torch.Tensor,
    brush_lift: torch.Tensor,
    gripper_aperture: torch.Tensor,
    min_contact_force: float,
    max_tcp_brush_distance: float,
    min_brush_lift: float,
    max_brush_lift: float,
    min_gripper_aperture: float,
    max_gripper_aperture: float,
) -> torch.Tensor:
    """Return the instantaneous, conservative brush-grasp predicate."""

    return (
        panda_is_grasping.to(torch.bool)
        & tcp_in_handle_region.to(torch.bool)
        & (left_finger_force >= min_contact_force)
        & (right_finger_force >= min_contact_force)
        & (tcp_brush_distance <= max_tcp_brush_distance)
        & (brush_lift >= min_brush_lift)
        & (brush_lift <= max_brush_lift)
        & (gripper_aperture >= min_gripper_aperture)
        & (gripper_aperture <= max_gripper_aperture)
    )


def advance_grasp_event(
    candidate: torch.Tensor,
    streak: torch.Tensor,
    already_rewarded: torch.Tensor,
    *,
    confirmation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a stable-grasp counter and emit a one-shot reward event."""

    if confirmation_steps <= 0:
        raise ValueError("confirmation_steps must be positive")
    if candidate.shape != streak.shape or candidate.shape != already_rewarded.shape:
        raise ValueError("candidate, streak, and already_rewarded must have equal shapes")
    if streak.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("streak must use an integer dtype")

    candidate = candidate.to(torch.bool)
    next_streak = torch.where(candidate, streak + 1, torch.zeros_like(streak))
    next_streak = torch.clamp(next_streak, max=confirmation_steps)
    confirmed = next_streak >= confirmation_steps
    event = confirmed & ~already_rewarded
    next_rewarded = already_rewarded | event
    return next_streak, event, next_rewarded
