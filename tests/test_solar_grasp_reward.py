import pytest
import torch

from maniskill_myws.tasks.solar_grasp_reward import (
    advance_grasp_event,
    brush_grasp_candidate,
)


DEFAULT_SIGNALS = {
    "left_finger_force": 1.0,
    "right_finger_force": 1.0,
    "panda_is_grasping": True,
    "tcp_in_handle_region": True,
    "tcp_brush_distance": 0.03,
    "brush_lift": 0.13,
    "gripper_aperture": 0.028,
}

DEFAULT_THRESHOLDS = {
    "min_contact_force": 0.5,
    "max_tcp_brush_distance": 0.06,
    "min_brush_lift": 0.12,
    "max_brush_lift": 0.65,
    "min_gripper_aperture": 0.006,
    "max_gripper_aperture": 0.055,
}


def _candidate(**overrides) -> bool:
    signals = DEFAULT_SIGNALS | overrides
    tensors = {
        key: torch.tensor([value])
        for key, value in signals.items()
    }
    result = brush_grasp_candidate(**tensors, **DEFAULT_THRESHOLDS)
    return bool(result.item())


def test_valid_brush_grasp_is_accepted():
    assert _candidate()


@pytest.mark.parametrize(
    ("signal", "value"),
    [
        ("left_finger_force", 0.0),  # one-finger contact
        ("right_finger_force", 0.0),
        ("panda_is_grasping", False),  # tangential two-finger push
        ("tcp_in_handle_region", False),  # pinching the brush head or side
        ("tcp_brush_distance", 0.10),  # brush moved without the gripper
        ("brush_lift", 0.0),  # pinched or pushed while still on the table
        ("brush_lift", 0.80),  # unstable/flying simulation state
        ("gripper_aperture", 0.0),  # empty fully closed gripper
        ("gripper_aperture", 0.07),  # gripper too wide to clamp the handle
    ],
)
def test_failed_grasp_signals_do_not_trigger_candidate(signal, value):
    assert not _candidate(**{signal: value})


def test_short_collision_does_not_emit_reward_event():
    streak = torch.zeros(1, dtype=torch.int32)
    rewarded = torch.zeros(1, dtype=torch.bool)
    events = []
    for candidate in (True, True, True, True, False):
        streak, event, rewarded = advance_grasp_event(
            torch.tensor([candidate]),
            streak,
            rewarded,
            confirmation_steps=5,
        )
        events.append(bool(event.item()))

    assert events == [False] * 5
    assert int(streak.item()) == 0
    assert not bool(rewarded.item())


def test_stable_grasp_emits_exactly_one_event():
    streak = torch.zeros(1, dtype=torch.int32)
    rewarded = torch.zeros(1, dtype=torch.bool)
    events = []
    for _ in range(9):
        streak, event, rewarded = advance_grasp_event(
            torch.ones(1, dtype=torch.bool),
            streak,
            rewarded,
            confirmation_steps=5,
        )
        events.append(bool(event.item()))

    assert events == [False, False, False, False, True, False, False, False, False]
    assert int(streak.item()) == 5
    assert bool(rewarded.item())


def test_drop_and_regrasp_does_not_pay_twice():
    streak = torch.zeros(1, dtype=torch.int32)
    rewarded = torch.zeros(1, dtype=torch.bool)
    events = []
    for candidate in ([True] * 5 + [False] + [True] * 5):
        streak, event, rewarded = advance_grasp_event(
            torch.tensor([candidate]),
            streak,
            rewarded,
            confirmation_steps=5,
        )
        events.append(bool(event.item()))

    assert sum(events) == 1
