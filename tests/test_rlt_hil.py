from __future__ import annotations

import numpy as np
import pytest

from maniskill_myws.rlt.hil import KeyboardInterventionGate, annotate_chunk_sources
from maniskill_myws.rlt.replay import TransitionSource


class FakeWindow:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.pressed: set[str] = set()

    def key_down(self, key: str) -> bool:
        return key in self.down

    def key_press(self, key: str) -> bool:
        if key not in self.pressed:
            return False
        self.pressed.remove(key)
        return True


def test_hold_gate_is_base_by_default_and_respects_warmup() -> None:
    window = FakeWindow()
    gate = KeyboardInterventionGate(window, mode="hold")

    assert gate.poll(rlt_available=True).use_rlt is False

    window.down.add("r")
    blocked = gate.poll(rlt_available=False)
    assert blocked.requested_rlt is True
    assert blocked.use_rlt is False
    assert blocked.blocked_by_warmup is True
    assert blocked.blocked_changed is True

    enabled = gate.poll(rlt_available=True)
    assert enabled.use_rlt is True
    assert enabled.control_changed is True

    window.down.clear()
    released = gate.poll(rlt_available=True)
    assert released.use_rlt is False
    assert released.control_changed is True


def test_hold_gate_base_key_overrides_rlt_key() -> None:
    window = FakeWindow()
    window.down.update({"r", "b"})
    gate = KeyboardInterventionGate(window, mode="hold")

    assert gate.poll(rlt_available=True).use_rlt is False


def test_latch_gate_switches_between_rlt_and_base_and_can_quit() -> None:
    window = FakeWindow()
    gate = KeyboardInterventionGate(window, mode="latch")

    window.pressed.add("r")
    assert gate.poll(rlt_available=True).use_rlt is True
    assert gate.poll(rlt_available=True).use_rlt is True

    window.pressed.add("b")
    assert gate.poll(rlt_available=True).use_rlt is False

    window.pressed.add("q")
    assert gate.poll(rlt_available=True).quit_requested is True


def test_latch_gate_uses_key_down_edges_when_key_press_is_missing() -> None:
    window = FakeWindow()
    gate = KeyboardInterventionGate(
        window,
        mode="latch",
        rlt_key="R",
        base_key="B",
        quit_key="Q",
    )

    window.down.add("r")
    assert gate.poll(rlt_available=True).use_rlt is True
    assert gate.poll(rlt_available=True).use_rlt is True

    window.down.clear()
    gate.poll(rlt_available=True)
    window.down.add("b")
    assert gate.poll(rlt_available=True).use_rlt is False

    window.down.clear()
    gate.poll(rlt_available=True)
    window.down.add("q")
    assert gate.poll(rlt_available=True).quit_requested is True


def test_keyboard_gate_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="distinct"):
        KeyboardInterventionGate(FakeWindow(), rlt_key="r", base_key="r")


def test_chunk_source_annotation_tracks_mixed_control_and_padding() -> None:
    source, source_chunk = annotate_chunk_sources(
        [
            int(TransitionSource.BASE),
            int(TransitionSource.RLT),
            int(TransitionSource.RLT),
        ],
        chunk_len=5,
    )

    assert source == int(TransitionSource.MIXED)
    np.testing.assert_array_equal(
        source_chunk,
        np.asarray(
            [
                TransitionSource.BASE,
                TransitionSource.RLT,
                TransitionSource.RLT,
                TransitionSource.RLT,
                TransitionSource.RLT,
            ],
            dtype=np.uint8,
        ),
    )


def test_chunk_source_annotation_preserves_single_controller() -> None:
    source, source_chunk = annotate_chunk_sources(
        [int(TransitionSource.BASE)] * 3,
        chunk_len=3,
    )

    assert source == int(TransitionSource.BASE)
    assert np.all(source_chunk == int(TransitionSource.BASE))
