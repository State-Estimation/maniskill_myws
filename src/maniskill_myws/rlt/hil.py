from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .replay import TransitionSource


class KeyboardWindow(Protocol):
    def key_down(self, key: str) -> bool: ...

    def key_press(self, key: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class HILDecision:
    requested_rlt: bool
    use_rlt: bool
    control_changed: bool
    blocked_by_warmup: bool
    blocked_changed: bool
    quit_requested: bool


class KeyboardInterventionGate:
    """Human gate selecting Base or RLT actions through a SAPIEN viewer window."""

    def __init__(
        self,
        window: KeyboardWindow,
        *,
        mode: Literal["hold", "latch"] = "hold",
        rlt_key: str = "r",
        base_key: str = "b",
        quit_key: str = "q",
    ) -> None:
        if mode not in {"hold", "latch"}:
            raise ValueError(f"Unsupported HIL keyboard mode: {mode}")
        keys = [
            str(rlt_key).strip().lower(),
            str(base_key).strip().lower(),
            str(quit_key).strip().lower(),
        ]
        if any(not key for key in keys):
            raise ValueError("HIL keyboard keys must be non-empty")
        if len(set(keys)) != len(keys):
            raise ValueError("HIL RLT, Base, and quit keys must be distinct")

        self.window = window
        self.mode = mode
        self.rlt_key, self.base_key, self.quit_key = keys
        self._latched_rlt = False
        self._last_use_rlt = False
        self._last_blocked = False
        self._last_key_down = {key: False for key in keys}

    def _keyboard_state(self) -> tuple[dict[str, bool], dict[str, bool]]:
        down: dict[str, bool] = {}
        pressed: dict[str, bool] = {}
        for key in (self.rlt_key, self.base_key, self.quit_key):
            key_down = bool(self.window.key_down(key))
            # key_press preserves very short taps reported by SAPIEN. The
            # key_down rising edge is a fallback for backends where key_press
            # is unreliable or has already been observed by a viewer plugin.
            pressed[key] = bool(self.window.key_press(key)) or (
                key_down and not self._last_key_down[key]
            )
            down[key] = key_down
        self._last_key_down = down
        return down, pressed

    def poll(self, *, rlt_available: bool) -> HILDecision:
        down, pressed = self._keyboard_state()
        quit_requested = pressed[self.quit_key]
        if self.mode == "hold":
            requested_rlt = down[self.rlt_key]
            if down[self.base_key]:
                requested_rlt = False
        else:
            if pressed[self.rlt_key]:
                self._latched_rlt = True
            if pressed[self.base_key]:
                self._latched_rlt = False
            requested_rlt = self._latched_rlt

        blocked = bool(requested_rlt and not rlt_available)
        use_rlt = bool(requested_rlt and rlt_available)
        decision = HILDecision(
            requested_rlt=requested_rlt,
            use_rlt=use_rlt,
            control_changed=use_rlt != self._last_use_rlt,
            blocked_by_warmup=blocked,
            blocked_changed=blocked != self._last_blocked,
            quit_requested=quit_requested,
        )
        self._last_use_rlt = use_rlt
        self._last_blocked = blocked
        return decision


def annotate_chunk_sources(
    executed_sources: list[int] | np.ndarray,
    *,
    chunk_len: int,
) -> tuple[int, np.ndarray]:
    """Pad per-step controller sources and return the aggregate chunk source."""

    sources = np.asarray(executed_sources, dtype=np.uint8).reshape(-1)
    if sources.size == 0:
        raise ValueError("Cannot annotate a chunk with no executed sources")
    if chunk_len < 1:
        raise ValueError("chunk_len must be positive")
    if sources.size > chunk_len:
        sources = sources[:chunk_len]

    valid_sources = {int(source) for source in TransitionSource}
    if any(int(source) not in valid_sources for source in sources):
        raise ValueError("Chunk contains an unknown transition source")

    padded = np.empty((chunk_len,), dtype=np.uint8)
    padded[: sources.size] = sources
    padded[sources.size :] = sources[-1]
    aggregate = (
        int(sources[0])
        if np.all(sources == sources[0])
        else int(TransitionSource.MIXED)
    )
    return aggregate, padded
