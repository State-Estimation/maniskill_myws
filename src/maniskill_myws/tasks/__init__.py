"""Lazy task exports so lightweight reward helpers do not require SAPIEN."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BrushSolarPanelEnv": (".brush_solar_panel", "BrushSolarPanelEnv"),
    "OpenSafeDoorEnv": (".open_safe_door", "OpenSafeDoorEnv"),
    "OpenSafeDoor2Env": (".open_safe_door2", "OpenSafeDoor2Env"),
    "OpenSafetyHookEnv": (".open_safety_hook", "OpenSafetyHookEnv"),
    "SolarPanelStaticEnv": (".sweep_solar_panel", "SolarPanelStaticEnv"),
    "SolarPanelStaticEnv2": (".sweep_solar_panel2", "SolarPanelStaticEnv2"),
    "StackCubeV2Env": (".stack_cube_v2", "StackCubeV2Env"),
    "TakeSafetyHookEnv": (".take_safety_hook", "TakeSafetyHookEnv"),
    "TakeSafetyHookEnv2": (".take_safety_hook2", "TakeSafetyHookEnv2"),
    "TurnGlobeValveEnv": (".turn_globe_valve", "TurnGlobeValveEnv"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
