#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _as_float(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _entity_set(links_map: dict[str, Any]) -> set[Any]:
    entities = set()
    for link in links_map.values():
        for body in getattr(link, "_bodies", []):
            entities.add(body.entity)
    return entities


def _robot_safe_contacts(env: Any, robot_entities: set[Any], safe_entities: set[Any]):
    hits = []
    for contact in env.unwrapped.scene.get_contacts():
        ent0 = contact.bodies[0].entity
        ent1 = contact.bodies[1].entity
        robot_safe = (ent0 in robot_entities and ent1 in safe_entities) or (
            ent1 in robot_entities and ent0 in safe_entities
        )
        if not robot_safe:
            continue
        impulse = np.zeros(3, dtype=np.float64)
        for point in contact.points:
            impulse += np.asarray(point.impulse, dtype=np.float64)
        hits.append((ent0.name, ent1.name, float(np.linalg.norm(impulse))))
    return hits


def _safe_scalar(env: Any, joint_name: str) -> float:
    joint = env.unwrapped.safe.active_joints_map[joint_name]
    return _as_float(joint.qpos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1000)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=None)
    parser.add_argument("--safe-spawn-center-x", type=float, default=None)
    parser.add_argument("--safe-spawn-center-y", type=float, default=None)
    parser.add_argument("--safe-spawn-half-size-x", type=float, default=None)
    parser.add_argument("--safe-spawn-half-size-y", type=float, default=None)
    parser.add_argument("--safe-yaw-noise", type=float, default=None)
    parser.add_argument("--contact-impulse-threshold", type=float, default=1e-6)
    parser.add_argument("--door-motion-threshold", type=float, default=1e-3)
    parser.add_argument("--button-motion-threshold", type=float, default=1e-3)
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym

    import maniskill_myws

    maniskill_myws.register()

    env_kwargs: dict[str, Any] = {
        "obs_mode": "state_dict",
        "reward_mode": "none",
        "control_mode": "pd_ee_delta_pose",
        "render_mode": None,
    }
    for arg_name, kw_name in [
        ("robot_init_qpos_noise", "robot_init_qpos_noise"),
        ("safe_spawn_center_x", "safe_spawn_center_x"),
        ("safe_spawn_center_y", "safe_spawn_center_y"),
        ("safe_spawn_half_size_x", "safe_spawn_half_size_x"),
        ("safe_spawn_half_size_y", "safe_spawn_half_size_y"),
        ("safe_yaw_noise", "safe_yaw_noise"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            env_kwargs[kw_name] = value

    env = gym.make(args.env_id, **env_kwargs)
    robot_entities = _entity_set(env.unwrapped.agent.robot.links_map)
    safe_entities = _entity_set(env.unwrapped.safe.links_map)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    bad = []
    worst_impulse = 0.0
    worst_door_motion = 0.0
    worst_button_motion = 0.0

    try:
        for i in range(args.num_seeds):
            seed = args.start_seed + i
            env.reset(seed=seed)
            initial_door = _safe_scalar(env, env.unwrapped.DOOR_JOINT_NAME)
            initial_button = _safe_scalar(env, env.unwrapped.BUTTON_JOINT_NAME)
            seed_contacts = []
            max_impulse = 0.0
            max_door_motion = 0.0
            max_button_motion = 0.0

            for step in range(max(0, args.probe_steps) + 1):
                contacts = _robot_safe_contacts(env, robot_entities, safe_entities)
                contacts = [
                    c for c in contacts if c[2] > args.contact_impulse_threshold
                ]
                if contacts:
                    seed_contacts.extend((step, *c) for c in contacts)
                    max_impulse = max(max_impulse, max(c[2] for c in contacts))

                door_motion = abs(
                    _safe_scalar(env, env.unwrapped.DOOR_JOINT_NAME) - initial_door
                )
                button_motion = abs(
                    _safe_scalar(env, env.unwrapped.BUTTON_JOINT_NAME) - initial_button
                )
                max_door_motion = max(max_door_motion, door_motion)
                max_button_motion = max(max_button_motion, button_motion)

                if step < args.probe_steps:
                    env.step(zero_action)

            worst_impulse = max(worst_impulse, max_impulse)
            worst_door_motion = max(worst_door_motion, max_door_motion)
            worst_button_motion = max(worst_button_motion, max_button_motion)

            failed = (
                bool(seed_contacts)
                or max_door_motion > args.door_motion_threshold
                or max_button_motion > args.button_motion_threshold
            )
            if failed:
                bad.append(
                    {
                        "seed": seed,
                        "contacts": seed_contacts[: args.max_report],
                        "max_impulse": max_impulse,
                        "max_door_motion": max_door_motion,
                        "max_button_motion": max_button_motion,
                    }
                )
                print(
                    "bad_seed",
                    bad[-1],
                    flush=True,
                )

            if (i + 1) % 100 == 0:
                print(
                    "progress",
                    {
                        "checked": i + 1,
                        "bad": len(bad),
                        "worst_impulse": worst_impulse,
                        "worst_door_motion": worst_door_motion,
                        "worst_button_motion": worst_button_motion,
                    },
                    flush=True,
                )
    finally:
        env.close()

    print(
        "summary",
        {
            "checked": args.num_seeds,
            "start_seed": args.start_seed,
            "bad": len(bad),
            "worst_impulse": worst_impulse,
            "worst_door_motion": worst_door_motion,
            "worst_button_motion": worst_button_motion,
            "bad_seeds": [x["seed"] for x in bad[: args.max_report]],
        },
    )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
