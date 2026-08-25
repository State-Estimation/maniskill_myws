#!/usr/bin/env python3
"""Collect exact base-policy rollouts for distributional V_base training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maniskill_myws.openpi_bridge.keypath import get_by_path  # noqa: E402
from maniskill_myws.openpi_bridge.remote_policy import (  # noqa: E402
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.value_dataset import VALUE_ROLLOUT_SCHEMA  # noqa: E402
from maniskill_myws.rlt.value_model import resize_value_rgb  # noqa: E402


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _scalar_bool(value: object) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar bool, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-id", default="TakeSafetyHook-v1")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument(
        "--enhanced-determinism", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--policy-resize", type=int, default=224)
    parser.add_argument("--value-image-size", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=61000)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
        args.policy_resize,
        args.value_image_size,
        args.chunk_len,
        args.max_episode_steps,
        args.num_episodes,
    ) <= 0:
        parser.error("image sizes, horizons, and episode count must be positive")
    if args.max_episode_steps % args.chunk_len != 0:
        parser.error("--max-episode-steps must be divisible by --chunk-len")
    if args.start_seed < 0:
        parser.error("--start-seed must be non-negative")
    return args


def _write_episode(
    path: Path,
    *,
    index: int,
    seed: int,
    success: bool,
    env_steps: int,
    images: np.ndarray,
    states: np.ndarray,
    latents: np.ndarray,
    ref_chunks: np.ndarray,
    step_ids: np.ndarray,
) -> None:
    import h5py

    with h5py.File(path, "a") as file:
        name = f"episode_{index:06d}"
        if name in file:
            raise ValueError(f"Value rollout episode already exists: {name}")
        group = file.create_group(name)
        group.attrs["seed"] = int(seed)
        group.attrs["success"] = bool(success)
        group.attrs["env_steps"] = int(env_steps)
        group.create_dataset("images", data=images, compression="lzf", shuffle=True)
        group.create_dataset("states", data=states, compression="lzf", shuffle=True)
        group.create_dataset(
            "latents", data=latents.astype(np.float16), compression="lzf", shuffle=True
        )
        group.create_dataset(
            "ref_chunks", data=ref_chunks, compression="lzf", shuffle=True
        )
        group.create_dataset("step_ids", data=step_ids)
        file.flush()


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite value rollout dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import h5py

    import maniskill_myws
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter

    maniskill_myws.register()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=args.render_mode,
        enhanced_determinism=args.enhanced_determinism,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        backend = require_resolved_backend(
            env,
            expected_sim_backend=args.sim_backend,
            expected_render_backend=args.render_backend,
        )
        base_env = env.unwrapped
        prompt = args.prompt or getattr(base_env, "DEFAULT_TASK_PROMPT", "")
        action_dim = int(np.prod(env.action_space.shape))
        policy = make_base_chunk_policy(
            "remote_openpi",
            action_space=env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.policy_resize,
            require_safe_latent=True,
        )
        state_adapter = StateAdapter(args.state_keys)
        probe_obs, _ = reset_env_fresh_scene(
            env,
            seed=args.start_seed,
            operation="value rollout state shape probe",
        )
        state_dim = int(np.asarray(state_adapter(probe_obs)).size)
        metadata = {
            "env_id": args.env_id,
            "obs_mode": args.obs_mode,
            "reward_mode": args.reward_mode,
            "control_mode": args.control_mode,
            "sim_backend": args.sim_backend,
            "render_backend": args.render_backend,
            "enhanced_determinism": bool(args.enhanced_determinism),
            "backend": backend,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "image_keys": [args.image_key, args.wrist_image_key],
            "state_keys": list(args.state_keys),
            "policy_resize": args.policy_resize,
            "value_image_size": args.value_image_size,
            "chunk_len": args.chunk_len,
            "max_episode_steps": args.max_episode_steps,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
            "safe_latent_dim": SAFE_LATENT_DIM,
            "openpi_policy_identity_sha256": openpi_policy_identity_sha256(
                policy.server_metadata or {}
            ),
            "label_source": "episode_environment_success_any_step",
            "base_policy_only": True,
        }
        if output.exists():
            with h5py.File(output, "r") as file:
                if str(file.attrs.get("schema", "")) != VALUE_ROLLOUT_SCHEMA:
                    raise ValueError("Resume value rollout schema mismatch")
                existing_metadata = json.loads(str(file.attrs["metadata_json"]))
                if existing_metadata != metadata:
                    raise ValueError("Resume value rollout metadata mismatch")
                existing = len(file.keys())
        else:
            with h5py.File(output, "w") as file:
                file.attrs["schema"] = VALUE_ROLLOUT_SCHEMA
                file.attrs["metadata_json"] = json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            existing = 0
        if existing > args.num_episodes:
            raise ValueError("Resume dataset already exceeds --num-episodes")

        successes = 0
        if existing:
            with h5py.File(output, "r") as file:
                for index in range(existing):
                    group = file[f"episode_{index:06d}"]
                    expected_seed = args.start_seed + index
                    if int(group.attrs["seed"]) != expected_seed:
                        raise ValueError("Resume seeds are not a contiguous prefix")
                    successes += int(bool(group.attrs["success"]))

        for episode_index in range(existing, args.num_episodes):
            episode_seed = args.start_seed + episode_index
            obs, _ = reset_env_fresh_scene(
                env,
                seed=episode_seed,
                operation=f"value rollout episode {episode_index}",
            )
            policy.reset()
            episode_steps = 0
            episode_success = False
            image_items: list[np.ndarray] = []
            state_items: list[np.ndarray] = []
            latent_items: list[np.ndarray] = []
            ref_items: list[np.ndarray] = []
            step_items: list[int] = []
            done = False
            while not done and episode_steps < args.max_episode_steps:
                image_items.append(
                    np.stack(
                        [
                            resize_value_rgb(
                                get_by_path(obs, args.image_key),
                                args.value_image_size,
                                args.value_image_size,
                            ),
                            resize_value_rgb(
                                get_by_path(obs, args.wrist_image_key),
                                args.value_image_size,
                                args.value_image_size,
                            ),
                        ],
                        axis=0,
                    )
                )
                state_items.append(
                    np.asarray(state_adapter(obs), dtype=np.float32).copy()
                )
                ref_chunk, latent = policy.plan_with_latent(
                    obs,
                    chunk_len=args.chunk_len,
                    action_dim=action_dim,
                    inference_seed=inference_seed_for_step(
                        episode_seed, episode_steps
                    ),
                )
                latent = np.asarray(latent, dtype=np.float32)
                if latent.shape != (SAFE_LATENT_DIM,) or not np.all(np.isfinite(latent)):
                    raise ValueError("OpenPI returned an invalid SAFE endpoint latent")
                if np.any(np.abs(latent) > np.finfo(np.float16).max):
                    raise ValueError("SAFE latent overflows float16 rollout storage")
                latent_items.append(latent.copy())
                ref_items.append(np.asarray(ref_chunk, dtype=np.float32).copy())
                step_items.append(episode_steps)
                for action in ref_chunk:
                    obs, _, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    if args.render_mode is not None:
                        env.render()
                    if not isinstance(info, dict) or "success" not in info:
                        raise ValueError("Environment info does not expose success")
                    episode_success |= _scalar_bool(info["success"])
                    done = bool(
                        _scalar_bool(terminated)
                        or _scalar_bool(truncated)
                        or episode_steps >= args.max_episode_steps
                    )
                    if done:
                        break
            _write_episode(
                output,
                index=episode_index,
                seed=episode_seed,
                success=episode_success,
                env_steps=episode_steps,
                images=np.stack(image_items).astype(np.uint8, copy=False),
                states=np.stack(state_items).astype(np.float32, copy=False),
                latents=np.stack(latent_items).astype(np.float32, copy=False),
                ref_chunks=np.stack(ref_items).astype(np.float32, copy=False),
                step_ids=np.asarray(step_items, dtype=np.int32),
            )
            successes += int(episode_success)
            print(
                json.dumps(
                    {
                        "stage": "value.collection_episode",
                        "episode": episode_index,
                        "seed": episode_seed,
                        "success": episode_success,
                        "env_steps": episode_steps,
                        "boundaries": len(step_items),
                        "success_rate": successes / float(episode_index + 1),
                        "output": str(output),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
