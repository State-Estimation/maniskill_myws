from __future__ import annotations

from hashlib import blake2b

# Central registry for task prompts.
# Keep this file free of ManiSkill imports so it can be used in conversion scripts.

TASK_PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "TurnGlobeValve-v1": (
        "grasp the round valve handwheel and rotate it until the valve is turned",
        "hold the valve wheel and turn it through the required rotation",
        "use the gripper to rotate the circular handwheel on the globe valve",
        "grab the handwheel and twist the valve until the task is complete",
        "place the gripper on the valve wheel and rotate the handwheel",
        "turn the circular handwheel of the globe valve with the robot gripper",
        "reach for the valve wheel, grip it, and rotate it far enough",
        "manipulate the round valve handle until the globe valve has been turned",
        "use the end effector to hold the handwheel and spin the valve",
        "rotate the metal valve wheel to complete the valve turning task",
        "grip the rim of the handwheel and twist the globe valve open or closed",
        "move the gripper onto the round handwheel and turn it through a large angle",
        "grab the circular valve control and rotate it until success",
        "use the robot hand to twist the globe valve handwheel",
        "turn the round valve wheel by pushing and pulling on its rim",
        "apply the gripper to the handwheel and rotate the valve mechanism",
        "hold the globe valve wheel securely and twist it around",
        "rotate the handwheel on the valve until the target rotation is reached",
        "operate the globe valve by grasping and turning its circular wheel",
        "complete the valve task by rotating the round handwheel with the gripper",
    ),
    "OpenSafeDoor-v1": (
        "open the safe door",
        "pull the safe door open",
        "grasp the safe handle and open the door",
        "operate the safe door until it is open",
    ),
    "OpenSafeDoor-v2": (
        "press the orange button, then grasp the green door handle and pull the door open",
        "push the orange release button before holding the green handle and opening the door",
        "unlock the safe by pressing the orange button, then pull the green handle to open it",
        "first press the orange button, then grab the green handle and swing the door open",
        "press the orange button on the safe, then use the green handle to pull the door open",
        "activate the orange button, grasp the green handle, and open the safe door",
        "push in the orange release, then hold the green door handle and swing the door outward",
        "use the gripper to press the orange button and then pull on the green handle",
        "first depress the orange button, then grab the green handle to open the door",
        "press the orange latch button, move to the green handle, and pull the safe door open",
        "unlock the door with the orange button and open it using the green handle",
        "push the orange button until released, then pull the green handle to swing the door",
        "operate the safe by pressing the orange button and pulling the green handle",
        "touch the orange button first, then grasp the green handle and open the safe",
        "release the door by pressing the orange button, then pull the green handle outward",
        "press the orange door button, close on the green handle, and open the door",
        "use the end effector to push the orange button before opening the door by the handle",
        "depress the orange button and pull the green handle until the safe door opens",
        "complete the safe task by pressing the orange button and pulling the green handle",
        "unlock with the orange button, then grip and pull the green handle to open the safe",
    ),
    "StackCube-v2": (
        "stack the red cube on the green cube",
        "pick up the red cube and place it on top of the green cube",
        "move the red block onto the green block",
        "build a stack by putting the red cube above the green cube",
    ),
    "BrushSolarPanel-v1": (
        "use the brush to clean the solar panel",
        "pick up the brush and sweep the solar panel surface clean",
        "move the brush head across the solar panel to clean it",
        "grasp the brush and scrub the dirty solar panel",
    ),
    "SolarPanelStatic-v1": (
        "sweep the solar panel clean with the brush",
        "grasp the brush and clean the solar panel surface",
        "use the brush head to wipe across the solar panel until it is clean",
        "pick up the brush and scrub the solar panel area",
        "hold the brush and sweep over the solar panel surface",
        "grab the brush, bring it to the panel, and wipe the panel clean",
        "use the gripper to pick up the brush and clean the solar panel",
        "move the brush head across the solar panel to cover the dirty area",
        "scrub the panel surface with the brush until enough area is clean",
        "pick up the brush from the table and sweep the solar panel face",
        "press the brush head onto the solar panel and wipe across it",
        "clean the solar panel by moving the brush over the panel surface",
        "grasp the brush handle and scrub across the solar panel",
        "use the brush to wipe the solar panel in broad sweeping motions",
        "bring the brush to the solar panel and clean the visible surface",
        "hold the brush steady and sweep it over the panel cells",
        "wipe the solar panel surface with the brush head until the task succeeds",
        "pick up the brush and move it back and forth across the solar panel",
        "clean the panel by sweeping the brush over multiple parts of its surface",
        "use the robot gripper to brush the solar panel clean",
    ),
    "OpenSafetyHook-v1": (
        "open the safety hook and remove it from the rod",
        "push the hook gate open, then slide the hook off the rod",
        "open the clasp on the safety hook and take it off the rod",
        "disengage the safety hook gate and remove the hook from the rod",
    ),
    "TakeSafetyHook-v1": (
        "grasp the safety hook hanging on the blue rod and remove it from the rod",
        "pick up the metal safety hook from the blue horizontal rod",
        "lift the safety hook off the blue rod",
        "take the hanging safety hook down from the blue rod",
        "grab the metal hook on the blue rod and pull it away from the rod",
        "use the gripper to take the safety hook off the blue horizontal rod",
        "reach for the hanging safety hook and remove it from the blue rod",
        "hold the safety hook and lift it clear of the blue rod",
        "pick the safety hook up from where it hangs on the blue rod",
        "grasp the hook body and take it down from the blue rod",
        "remove the metal safety hook hanging from the blue support rod",
        "use the robot hand to lift the safety hook off the blue bar",
        "close the gripper on the safety hook and pull it off the blue rod",
        "take the hook from the blue rod without moving the rod itself",
        "lift the hanging metal hook until it is no longer on the blue rod",
        "move to the safety hook, grasp it, and remove it from the blue rod",
        "pull the safety hook away from the blue horizontal support",
        "pick up the hook that is hanging on the blue rod",
        "use the gripper to hold the safety hook and take it off the rod",
        "complete the hook task by removing the safety hook from the blue rod",
    ),
}

TASK_PROMPTS: dict[str, str] = {
    env_id: prompts[0] for env_id, prompts in TASK_PROMPT_VARIANTS.items()
}


def _stable_index(key: str, size: int) -> int:
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def get_task_prompts(env_id: str) -> tuple[str, ...]:
    return TASK_PROMPT_VARIANTS.get(env_id, ())


def get_task_prompt(env_id: str, variant: int | str | None = None) -> str | None:
    prompts = get_task_prompts(env_id)
    if not prompts:
        return None
    if variant is None:
        return prompts[0]
    if isinstance(variant, int):
        return prompts[variant % len(prompts)]
    return prompts[_stable_index(f"{env_id}:{variant}", len(prompts))]
