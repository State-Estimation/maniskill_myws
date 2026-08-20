#!/usr/bin/env python
"""Collect base-policy pre-velocity rollouts for the official SAFE trainer."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _scalar_bool(value: object) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar value, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-key", default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--num-episodes", type=int, default=600)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(
        args.resize, args.chunk_len, args.max_episode_steps, args.num_episodes
    ) <= 0:
        parser.error("resize, chunk length, horizon, and episode count must be positive")
    if args.start_seed < 0:
        parser.error("start seed must be non-negative")
    return args


def _environment_identity(env) -> dict[str, str]:
    task_class = type(env.unwrapped)
    source_path = inspect.getsourcefile(task_class)
    if source_path is None:
        raise RuntimeError("Cannot locate the task environment source")
    source = Path(source_path).read_bytes()
    return {
        "environment_class": f"{task_class.__module__}.{task_class.__qualname__}",
        "environment_module_source_sha256": hashlib.sha256(source).hexdigest(),
    }


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite SAFE rollouts: {output}")
    if args.resume and not output.is_file():
        raise FileNotFoundError(f"Cannot resume missing SAFE rollouts: {output}")

    import gymnasium as gym

    import maniskill_myws
    from maniskill_myws.openpi_bridge.remote_policy import (
        SAFE_LATENT_DIFFUSION_SELECTION,
        SAFE_LATENT_DIM,
        SAFE_LATENT_HORIZON_SELECTION,
        SAFE_LATENT_POOLING,
        SAFE_LATENT_PROTOCOL,
        SAFE_LATENT_SOURCE,
    )
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.safe_rollouts import (
        SafeRolloutDataset,
        SafeRolloutEpisode,
        load_safe_rollout_dataset,
        save_safe_rollout_dataset,
    )

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
        action_dim = int(np.prod(env.action_space.shape))
        prompt = args.prompt or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
        if not prompt:
            raise ValueError("The environment has no default prompt; pass --prompt")
        policy = make_base_chunk_policy(
            "remote_openpi",
            action_space=env.action_space,
            action_dim=action_dim,
            server=args.server,
            prompt=prompt,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            state_keys=args.state_keys,
            resize=args.resize,
            require_safe_latent=True,
        )
        server_metadata = policy.server_metadata or {}
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
            "image_key": args.image_key,
            "wrist_image_key": args.wrist_image_key,
            "state_keys": list(args.state_keys),
            "resize": args.resize,
            "chunk_len": args.chunk_len,
            "max_episode_steps": args.max_episode_steps,
            "action_dim": action_dim,
            "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
            "safe_latent_dim": SAFE_LATENT_DIM,
            "safe_latent_source": SAFE_LATENT_SOURCE,
            "safe_latent_diffusion_selection": SAFE_LATENT_DIFFUSION_SELECTION,
            "safe_latent_horizon_selection": SAFE_LATENT_HORIZON_SELECTION,
            "safe_latent_pooling": SAFE_LATENT_POOLING,
            "safe_latent_pred_horizon": server_metadata.get(
                "safe_latent_pred_horizon"
            ),
            "openpi_policy_identity_sha256": openpi_policy_identity_sha256(
                server_metadata
            ),
            "label_source": "episode_environment_success_any_step",
            "base_policy_only": True,
            **_environment_identity(env),
        }
        episodes: list[SafeRolloutEpisode]
        if args.resume:
            existing = load_safe_rollout_dataset(output)
            if existing.metadata != metadata:
                raise ValueError("SAFE resume identity does not match this runtime")
            expected_seeds = [
                args.start_seed + index for index in range(len(existing.episodes))
            ]
            if [episode.seed for episode in existing.episodes] != expected_seeds:
                raise ValueError("SAFE resume seeds are not a contiguous prefix")
            if len(existing.episodes) > args.num_episodes:
                raise ValueError("SAFE resume data exceeds --num-episodes")
            episodes = list(existing.episodes)
        else:
            episodes = []

        for episode_index in range(len(episodes), args.num_episodes):
            episode_seed = args.start_seed + episode_index
            obs, _ = reset_env_fresh_scene(
                env,
                seed=episode_seed,
                operation=f"SAFE collection episode {episode_index}",
            )
            policy.reset()
            episode_steps = 0
            episode_success = False
            latents: list[np.ndarray] = []
            done = False
            while not done and episode_steps < args.max_episode_steps:
                action_chunk, latent = policy.plan_with_latent(
                    obs,
                    chunk_len=args.chunk_len,
                    action_dim=action_dim,
                    inference_seed=inference_seed_for_step(
                        episode_seed, episode_steps
                    ),
                )
                latent = np.asarray(latent)
                if latent.shape != (SAFE_LATENT_DIM,) or latent.dtype != np.float32:
                    raise ValueError("OpenPI returned an invalid SAFE latent")
                if not np.all(np.isfinite(latent)):
                    raise ValueError("OpenPI SAFE latent contains NaN or Inf")
                latents.append(latent.copy())
                for action in np.asarray(action_chunk, dtype=np.float32):
                    obs, _, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    if args.render_mode is not None:
                        env.render()
                    if not isinstance(info, dict) or "success" not in info:
                        raise ValueError("Environment info does not contain success")
                    episode_success |= _scalar_bool(info["success"])
                    done = bool(
                        _scalar_bool(terminated)
                        or _scalar_bool(truncated)
                        or episode_steps >= args.max_episode_steps
                    )
                    if done:
                        break
            episodes.append(
                SafeRolloutEpisode(
                    latents=np.stack(latents).astype(np.float32, copy=False),
                    success=episode_success,
                    seed=episode_seed,
                    env_steps=episode_steps,
                )
            )
            save_safe_rollout_dataset(
                output,
                SafeRolloutDataset(episodes=tuple(episodes), metadata=metadata),
            )
            print(
                json.dumps(
                    {
                        "stage": "safe.collection_episode",
                        "episode": episode_index,
                        "seed": episode_seed,
                        "success": episode_success,
                        "chunks": len(latents),
                        "env_steps": episode_steps,
                        "success_rate": float(
                            np.mean([episode.success for episode in episodes])
                        ),
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
