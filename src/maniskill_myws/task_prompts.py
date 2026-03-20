from __future__ import annotations

# Central registry for task prompts.
# Keep this file free of ManiSkill imports so it can be used in conversion scripts.

TASK_PROMPTS: dict[str, str] = {
    "TurnGlobeValve-v1": "turn the globe valve",
    "OpenSafeDoor-v1": "open the safe door",
    "StackCube-v2": "stack the red cube on the green cube",
    "SolarPanelStatic-v1": "sweep the solar panel clean with the brush",
    "OpenSafeDoor-v2": "open the door",
    "TakeSafetyHook-v1": "take the safety hook and hang it on the wall",
}


def get_task_prompt(env_id: str) -> str | None:
    return TASK_PROMPTS.get(env_id)
