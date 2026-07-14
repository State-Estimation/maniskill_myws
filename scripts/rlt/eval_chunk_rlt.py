#!/usr/bin/env python
"""Evaluate a ManiSkill RLT chunk policy on top of a remote OpenPI policy."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _as_done(x) -> bool:
    return bool(_as_numpy(x).reshape(-1)[0])


def _infer_control_dt(env, fallback_hz: float = 20.0) -> tuple[float, float]:
    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = fallback_hz
    control_freq = float(control_freq)
    if control_freq <= 0:
        control_freq = fallback_hz
    return 1.0 / control_freq, control_freq


def _default_max_steps(env, fallback: int = 500) -> int:
    max_steps = getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    return int(max_steps or fallback)


def _seed_result_line(seed: int, success: bool, steps: int, episode_return: float) -> str:
    status = "SUCCESS" if success else "FAIL"
    return f"  Seed {seed}: {status} ({steps} steps, return={episode_return:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/rlt/TurnGlobeValve-v1_pi0_chunk50/maniskill_rlt.pt",
    )
    parser.add_argument("--server", type=str, required=True, help="e.g. ws://127.0.0.1:8000")
    parser.add_argument("--env-id", type=str, default="TurnGlobeValve-v1")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Throttle rendered rollout to env.control_freq.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the RLT actor instead of using deterministic mean actions.",
    )

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--rlt-image-keys", type=str, nargs="+", default=None)
    parser.add_argument(
        "--rlt-image-size",
        type=int,
        default=None,
        help="Optional resize for visual RLT inputs. Defaults to checkpoint image_shape.",
    )

    parser.add_argument("--output-root", type=str, default="outputs/rlt_eval")
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root() / "src"))

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.rlt.policies import make_base_chunk_policy
    from maniskill_myws.rlt.replay import validate_pd_joint_pos_action_dim
    from maniskill_myws.rlt.state import ImageAdapter, StateAdapter
    from maniskill_myws.rlt.trainer import ManiSkillRLTAgent

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maniskill_myws.register()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    checkpoint = Path(args.checkpoint)
    agent = ManiSkillRLTAgent.load(checkpoint, device=device)
    agent.actor.eval()
    cfg = agent.config

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    if args.max_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_steps
    env = gym.make(args.env_id, **env_kwargs)

    action_dim = int(np.prod(env.action_space.shape))
    validate_pd_joint_pos_action_dim(action_dim)
    if action_dim != cfg.action_dim:
        raise ValueError(
            f"Checkpoint action_dim={cfg.action_dim}, but env action_dim={action_dim}."
        )

    prompt = args.prompt
    if prompt is None:
        prompt = getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    base_policy = make_base_chunk_policy(
        "remote_openpi",
        action_space=env.action_space,
        action_dim=action_dim,
        server=args.server,
        prompt=prompt,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=args.state_keys,
        resize=args.resize,
    )
    state_adapter = StateAdapter(args.state_keys)

    image_adapter = None
    if cfg.visual_encoder != "none":
        image_keys = args.rlt_image_keys or [args.image_key, args.wrist_image_key]
        image_size = args.rlt_image_size
        if image_size is None and cfg.image_shape is not None and len(cfg.image_shape) >= 3:
            image_size = int(cfg.image_shape[1])
        image_adapter = ImageAdapter(
            image_keys,
            image_size=image_size,
            image_shape=cfg.image_shape,
        )

    max_steps = args.max_steps or _default_max_steps(env)
    real_time_dt = None
    if args.real_time:
        real_time_dt, control_freq = _infer_control_dt(env)
        print(
            "real-time pacing:",
            dict(control_freq_hz=control_freq, target_dt_s=round(real_time_dt, 4)),
            flush=True,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"{args.env_id}_{timestamp}"
    output_dir = Path(args.output_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_seed = args.start_seed if args.start_seed is not None else args.seed
    results: list[dict[str, object]] = []
    success_count = 0
    total_steps = 0

    print("=== RLT Chunk Evaluation ===", flush=True)
    print(f"Environment: {args.env_id}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Server: {args.server}", flush=True)
    print(f"Seeds: {start_seed} to {start_seed + args.num_seeds - 1}", flush=True)
    print(f"Chunk length: {cfg.chunk_len}", flush=True)
    print(f"Reward mode: {args.reward_mode}", flush=True)
    print(f"Max steps: {max_steps}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print("", flush=True)

    try:
        for i in range(args.num_seeds):
            seed = start_seed + i
            obs, _ = env.reset(seed=seed)
            base_policy.reset()
            state = state_adapter(obs)
            if state.shape[0] != cfg.state_dim:
                raise ValueError(
                    f"State dim mismatch: checkpoint expects {cfg.state_dim}, "
                    f"but state_keys produced {state.shape[0]}."
                )
            image = image_adapter(obs) if image_adapter is not None else None
            steps = 0
            episode_return = 0.0
            success = False
            done = False
            info: dict = {}

            while not done and steps < max_steps:
                ref_chunk = base_policy.plan(
                    obs,
                    chunk_len=cfg.chunk_len,
                    action_dim=cfg.action_dim,
                )
                action_chunk = agent.select_chunk(
                    state,
                    ref_chunk,
                    images=image,
                    deterministic=not args.stochastic,
                )

                for local_action in action_chunk[: cfg.chunk_len]:
                    step_wall_start = time.perf_counter()
                    obs, reward, terminated, truncated, step_info = env.step(local_action)
                    info = step_info if isinstance(step_info, dict) else {}
                    if args.render_mode is not None:
                        env.render()
                    steps += 1
                    total_steps += 1
                    episode_return += float(_as_numpy(reward).reshape(-1)[0])
                    done = (
                        _as_done(terminated)
                        or _as_done(truncated)
                        or steps >= max_steps
                    )
                    if real_time_dt is not None:
                        sleep_s = real_time_dt - (time.perf_counter() - step_wall_start)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                    if done:
                        break

                if not done:
                    state = state_adapter(obs)
                    image = image_adapter(obs) if image_adapter is not None else None

            success = bool(info.get("success", False))
            if success:
                success_count += 1
            result = {
                "seed": seed,
                "success": success,
                "steps": steps,
                "return": episode_return,
            }
            results.append(result)
            print(
                f"[{i + 1}/{args.num_seeds}]"
                + _seed_result_line(seed, success, steps, episode_return),
                flush=True,
            )
    finally:
        env.close()

    success_rate = 100.0 * success_count / max(1, args.num_seeds)
    avg_steps = total_steps / max(1, args.num_seeds)
    avg_return = float(np.mean([float(r["return"]) for r in results])) if results else 0.0

    print("", flush=True)
    print("=== Summary ===", flush=True)
    print(f"Total runs: {args.num_seeds}", flush=True)
    print(f"Success: {success_count}", flush=True)
    print(f"Failure: {args.num_seeds - success_count}", flush=True)
    print(f"Success rate: {success_rate:.1f}%", flush=True)
    print(f"Average steps: {avg_steps:.1f}", flush=True)
    print(f"Average return: {avg_return:.4f}", flush=True)

    results_file = output_dir / "results.txt"
    with results_file.open("w", encoding="utf-8") as f:
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Server: {args.server}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Chunk length: {cfg.chunk_len}\n")
        f.write(f"Deterministic: {not args.stochastic}\n")
        f.write(f"Reward mode: {args.reward_mode}\n")
        f.write(f"Max steps: {max_steps}\n")
        f.write("\nResults:\n")
        for r in results:
            f.write(
                _seed_result_line(
                    int(r["seed"]),
                    bool(r["success"]),
                    int(r["steps"]),
                    float(r["return"]),
                )
                + "\n"
            )
        f.write("\nSummary:\n")
        f.write(f"  Success rate: {success_rate:.1f}%\n")
        f.write(f"  Average steps: {avg_steps:.1f}\n")
        f.write(f"  Average return: {avg_return:.4f}\n")

    print(f"Results saved to: {results_file}", flush=True)


if __name__ == "__main__":
    main()
