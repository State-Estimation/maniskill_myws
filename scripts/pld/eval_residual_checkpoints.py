#!/usr/bin/env python
"""Evaluate deterministic residual SAC checkpoints saved during PLD training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_render_mode(render_mode: str | None) -> str | None:
    if render_mode is None:
        return None
    if render_mode.lower() in {"none", "null", ""}:
        return None
    return render_mode


def _as_done(x) -> bool:
    return bool(np.asarray(x).reshape(-1)[0])


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"residual_sac_step_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def _find_checkpoints(checkpoint_dir: Path, *, include_final: bool) -> list[Path]:
    checkpoints = sorted(
        checkpoint_dir.glob("residual_sac_step_*.pt"),
        key=lambda p: (_checkpoint_step(p), p.name),
    )
    final_ckpt = checkpoint_dir / "residual_sac.pt"
    if include_final and final_ckpt.exists():
        checkpoints.append(final_ckpt)
    return checkpoints


def _init_wandb(args, output_dir: Path):
    if not args.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as e:
        raise SystemExit(
            "wandb is not installed in the current environment. "
            "Install it or run without --wandb-enabled."
        ) from e
    wandb_dir = (
        Path(args.wandb_dir).expanduser().resolve()
        if args.wandb_dir
        else output_dir.resolve()
    )
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=vars(args),
    )
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("eval/*", step_metric="checkpoint_step")
    run_id = getattr(run, "id", None)
    if run_id:
        (output_dir / "wandb_eval_run_id.txt").write_text(f"{run_id}\n")
    return wandb


def _wandb_log(wandb_mod, payload: dict[str, object]) -> None:
    if wandb_mod is not None:
        wandb_mod.log(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/pld/OpenSafeDoor-v2",
        help="Directory containing residual_sac_step_*.pt checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit checkpoint path(s). Overrides --checkpoint-dir scanning.",
    )
    parser.add_argument("--include-final", action="store_true", default=True)
    parser.add_argument("--no-include-final", action="store_false", dest="include_final")

    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--vary-seeds-by-checkpoint",
        action="store_true",
        help="Offset seeds per checkpoint. By default all checkpoints use identical eval seeds.",
    )
    parser.add_argument("--seed-stride", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["residual", "base"],
        default="residual",
        help="'residual' evaluates a_base + deterministic residual; 'base' ignores checkpoints.",
    )
    parser.add_argument(
        "--stochastic-residual",
        action="store_true",
        help="Sample residual actions during eval. Default is deterministic tanh mean.",
    )
    parser.add_argument(
        "--otf-rollout-actions",
        type=int,
        default=0,
        help=(
            "Evaluate the OTF residual policy by sampling this many residual "
            "candidate actions and executing the critic-best candidate. 0 keeps "
            "the deterministic/stochastic residual action path."
        ),
    )
    parser.add_argument(
        "--otf-no-base-candidate",
        action="store_true",
        help="Do not include the unedited base-policy action as an OTF eval candidate.",
    )

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi")
    parser.add_argument("--server", type=str, default=None, help="ws://host:port for remote_openpi")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument(
        "--rl-image-keys",
        type=str,
        nargs="+",
        default=None,
        help="Residual RL image keys. Defaults to --image-key and --wrist-image-key.",
    )
    parser.add_argument("--rl-image-size", type=int, default=None)
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--csv-name", type=str, default="checkpoint_eval.csv")
    parser.add_argument("--json-name", type=str, default="checkpoint_eval.json")

    parser.add_argument("--wandb-enabled", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="maniskill-pld")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.sac import ResidualSAC
    from maniskill_myws.pld.state import ImageAdapter, StateAdapter
    from maniskill_myws.task_prompts import get_task_prompt

    maniskill_myws.register()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if args.checkpoint:
        checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoint]
    elif args.mode == "base":
        checkpoints = []
    else:
        checkpoints = _find_checkpoints(checkpoint_dir, include_final=args.include_final)
    if args.mode == "residual" and not checkpoints:
        raise SystemExit(f"No residual checkpoints found in {checkpoint_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_dir / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args, output_dir)

    render_mode = _normalize_render_mode(args.render_mode)
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
    )
    max_steps = args.max_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_steps is None and getattr(env, "spec", None) is not None:
        max_steps = getattr(env.spec, "max_episode_steps", None)
    max_steps = int(max_steps or 500)

    prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    action_dim = None
    visual_image_shape = None
    if args.mode == "residual":
        first_agent = ResidualSAC.load(checkpoints[0], device=device)
        action_dim = first_agent.config.action_dim
        visual_image_shape = first_agent.config.image_shape
        del first_agent
    else:
        action_dim = int(np.prod(env.action_space.shape))
    base_policy = make_base_policy(
        args.base_policy,
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
    rl_image_keys = args.rl_image_keys or [args.image_key, args.wrist_image_key]
    image_adapter = None
    if visual_image_shape is not None:
        visual_image_shape = tuple(int(x) for x in visual_image_shape)
        image_size = args.rl_image_size or int(visual_image_shape[1])
        image_adapter = ImageAdapter(
            rl_image_keys,
            image_size=image_size,
            image_shape=visual_image_shape,
        )

    eval_items: list[tuple[Path | None, int, str]] = []
    if args.mode == "base":
        eval_items.append((None, -1, "base"))
    else:
        for ckpt in checkpoints:
            step = _checkpoint_step(ckpt)
            name = "final" if step < 0 and ckpt.name == "residual_sac.pt" else ckpt.stem
            eval_items.append((ckpt, step, name))
    max_numeric_step = max((step for _, step, _ in eval_items), default=0)

    rows: list[dict[str, object]] = []
    try:
        for ckpt_path, ckpt_step, ckpt_name in eval_items:
            agent = None
            if ckpt_path is not None:
                agent = ResidualSAC.load(ckpt_path, device=device)
                action_dim = agent.config.action_dim

            successes = 0
            returns: list[float] = []
            lengths: list[int] = []
            t0 = time.time()
            for ep in range(args.num_episodes):
                seed_offset = 0
                if args.vary_seeds_by_checkpoint:
                    seed_offset = max(0, ckpt_step) * args.seed_stride
                seed = args.start_seed + ep + seed_offset
                obs, _ = env.reset(seed=seed)
                base_policy.reset()
                episode_return = 0.0
                success = False
                steps = 0
                for step in range(max_steps):
                    base_action = base_policy.act(obs)
                    if agent is None:
                        action = base_action
                    else:
                        state = state_adapter(obs)
                        image = image_adapter(obs) if image_adapter is not None else None
                        if args.otf_rollout_actions > 0:
                            action = agent.select_action_otf(
                                state,
                                base_action,
                                n_actions=args.otf_rollout_actions,
                                images=image,
                                include_base_action=not args.otf_no_base_candidate,
                            )
                        else:
                            action = agent.select_action(
                                state,
                                base_action,
                                images=image,
                                deterministic=not bool(args.stochastic_residual),
                            )
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(np.asarray(reward).reshape(-1)[0])
                    steps = step + 1
                    success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    if render_mode is not None:
                        env.render()
                    if _as_done(terminated) or _as_done(truncated):
                        break

                successes += int(success)
                returns.append(float(episode_return))
                lengths.append(int(steps))

            success_rate = successes / max(1, args.num_episodes)
            row = {
                "checkpoint": "" if ckpt_path is None else str(ckpt_path),
                "checkpoint_name": ckpt_name,
                "checkpoint_step": ckpt_step,
                "mode": args.mode,
                "deterministic": not bool(args.stochastic_residual),
                "otf_rollout_actions": int(args.otf_rollout_actions),
                "otf_include_base_action": not bool(args.otf_no_base_candidate),
                "num_episodes": args.num_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "mean_return": float(np.mean(returns)) if returns else 0.0,
                "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
                "elapsed_s": round(time.time() - t0, 2),
            }
            rows.append(row)
            print("eval", row)
            wandb_step = ckpt_step if ckpt_step >= 0 else max_numeric_step + 1
            _wandb_log(
                wandb_run,
                {
                    "checkpoint_step": wandb_step,
                    "eval/success_rate": success_rate,
                    "eval/successes": successes,
                    "eval/num_episodes": args.num_episodes,
                    "eval/mean_return": row["mean_return"],
                    "eval/mean_steps": row["mean_steps"],
                    "eval/elapsed_s": row["elapsed_s"],
                    "eval/deterministic": float(not bool(args.stochastic_residual)),
                    "eval/otf_rollout_actions": int(args.otf_rollout_actions),
                },
            )
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print("saved_csv:", csv_path)
    print("saved_json:", json_path)


if __name__ == "__main__":
    main()
