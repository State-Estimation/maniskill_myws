#!/usr/bin/env python
"""Stage 1B: online PLD residual SAC training from a pretrained critic."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from maniskill_myws.pld import train_common as common

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--render-every", type=int, default=1)

    parser.add_argument("--base-policy", choices=["remote_openpi", "zero", "random"], default="remote_openpi")
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--resize", type=int, default=224)

    common.add_offline_replay_args(parser)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env-device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=250_000)
    parser.add_argument("--init-critic-checkpoint", type=str, default=None)
    parser.add_argument("--init-actor-checkpoint", type=str, default=None)
    parser.add_argument(
        "--visual-encoder-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional pretrained ResNetV1-10/visual encoder checkpoint. "
            "Loaded into actor and critics before other checkpoint initialization."
        ),
    )
    parser.add_argument(
        "--no-init-actor-visual-from-critic",
        action="store_true",
        help=(
            "Do not copy the loaded critic visual encoder into the actor after "
            "--init-critic-checkpoint. By default visual actor init follows the critic."
        ),
    )
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=250_000)
    parser.add_argument("--offline-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-episodes", type=int, default=100)
    parser.add_argument("--warmup-buffer-path", type=str, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=25_000)

    common.add_sac_model_args(parser)
    parser.add_argument("--cql-alpha", type=float, default=0.0)
    parser.add_argument("--otf-backup-actions", type=int, default=0)
    parser.add_argument("--otf-rollout-actions", type=int, default=0)
    parser.add_argument("--otf-no-base-candidate", action="store_true")
    parser.add_argument("--otf-backup-entropy", action="store_true")

    parser.add_argument("--visualize-tcp-path", action="store_true")
    parser.add_argument("--path-every", type=int, default=2)
    parser.add_argument("--path-max-points", type=int, default=500)
    parser.add_argument("--path-radius", type=float, default=0.008)
    parser.add_argument("--tcp-pose-key", type=str, default="extra/tcp_pose")
    parser.add_argument("--base-chunk-max-actions", type=int, default=16)
    parser.add_argument("--base-chunk-position-scale", type=float, default=0.1)
    parser.add_argument("--base-path-color", type=str, default="0.05,0.35,1.0,1.0")
    parser.add_argument("--residual-path-color", type=str, default="1.0,0.28,0.02,1.0")

    parser.add_argument("--output-dir", type=str, default="outputs/pld/OpenSafeDoor-v2")
    parser.add_argument("--checkpoint-name", type=str, default="residual_sac.pt")
    parser.add_argument("--actor-checkpoint-name", type=str, default="residual_actor.pt")
    common.add_wandb_args(parser)
    args = parser.parse_args()

    if args.total_env_steps <= 0:
        raise SystemExit("--total-env-steps must be positive for online training")

    common.prepare_runtime(args.seed)
    import gymnasium as gym

    from maniskill_myws.pld.env_device import apply_env_device_kwargs
    from maniskill_myws.pld.policies import make_base_policy
    from maniskill_myws.pld.path_visualizer import TCPPathVisualizer, parse_rgba
    from maniskill_myws.pld.replay_buffer import ReplayBuffer, sample_offline_online
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    from maniskill_myws.pld.state import ImageAdapter, StateAdapter
    from maniskill_myws.task_prompts import get_task_prompt

    offline_data, h5_files, rl_image_keys = common.load_offline_replay(args)
    offline_buffer = common.make_offline_buffer(offline_data, batch_size=args.batch_size)

    device, cuda_info = common.resolve_torch_device(args.device)
    print(
        "devices:",
        dict(
            rl_device=str(device),
            env_device=args.env_device or "maniskill_default",
            **cuda_info,
        ),
        flush=True,
    )

    action_dim = offline_data.action_dim
    render_mode = common.normalize_render_mode(args.render_mode)
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=render_mode,
    )
    apply_env_device_kwargs(env_kwargs, args.env_device)
    env = gym.make(args.env_id, **env_kwargs)
    low, high = common.action_bounds(env.action_space, action_dim)

    image_shape = offline_data.image_shape if args.use_visual_rl else None
    online_buffer = ReplayBuffer(
        args.buffer_capacity, offline_data.state_dim, action_dim, image_shape=image_shape
    )
    warmup_buffer_path = (
        Path(args.warmup_buffer_path).expanduser().resolve()
        if args.warmup_buffer_path
        else None
    )
    loaded_warmup_episode_idx = 0
    loaded_warmup_env_steps = 0
    warmup_buffer_saved = warmup_buffer_path is None
    if warmup_buffer_path is not None and warmup_buffer_path.exists():
        warmup_meta = online_buffer.load(warmup_buffer_path)
        loaded_warmup_episode_idx = int(
            warmup_meta.get("episode_idx", warmup_meta.get("warmup_episodes", args.warmup_episodes))
        )
        loaded_warmup_env_steps = int(warmup_meta.get("env_steps", len(online_buffer)))
        warmup_buffer_saved = True
        print(
            "loaded warmup buffer:",
            dict(
                path=str(warmup_buffer_path),
                transitions=len(online_buffer),
                episode_idx=loaded_warmup_episode_idx,
                env_steps=loaded_warmup_env_steps,
            ),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name

    wandb_run = common.init_wandb(
        args,
        output_dir=output_dir,
        offline_files=len(h5_files),
        offline_transitions=offline_data.size,
        state_dim=offline_data.state_dim,
        action_dim=action_dim,
        action_low=low,
        action_high=high,
    )
    common.wandb_log(
        wandb_run,
        dict(
            offline_update=0,
            **{
                "offline/files": len(h5_files),
                "offline/transitions": offline_data.size,
                "offline/state_dim": offline_data.state_dim,
                "offline/action_dim": action_dim,
                "offline/mc_return_min": float(np.min(offline_data.mc_returns)),
                "offline/mc_return_max": float(np.max(offline_data.mc_returns)),
            },
        ),
    )

    agent = ResidualSAC(
        SACConfig(
            state_dim=offline_data.state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            actor_lr=args.lr,
            critic_lr=args.lr,
            alpha_lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            action_scale=args.action_scale,
            target_entropy=args.target_entropy,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            grad_clip_norm=args.grad_clip_norm,
            cql_alpha=args.cql_alpha,
            otf_backup_actions=args.otf_backup_actions,
            otf_include_base_action=not args.otf_no_base_candidate,
            otf_backup_entropy=args.otf_backup_entropy,
            visual_encoder="resnet10" if args.use_visual_rl else "none",
            image_shape=image_shape,
            visual_latent_dim=args.visual_latent_dim,
            action_low=low,
            action_high=high,
        ),
        device=device,
    )
    if args.visual_encoder_checkpoint:
        loaded = agent.load_visual_encoder(args.visual_encoder_checkpoint)
        print(
            "loaded visual encoder checkpoint:",
            dict(path=args.visual_encoder_checkpoint, tensors=loaded),
            flush=True,
        )
    if args.init_critic_checkpoint:
        agent.load_critics(args.init_critic_checkpoint)
        print("loaded critic checkpoint:", args.init_critic_checkpoint, flush=True)
        if args.use_visual_rl and not args.no_init_actor_visual_from_critic:
            copied = agent.init_actor_visual_from_critic(source="q1")
            print(
                "initialized actor visual encoder from critic:",
                dict(enabled=True, copied=bool(copied), source="q1"),
                flush=True,
            )
    if args.init_actor_checkpoint:
        agent.load_actor(args.init_actor_checkpoint)
        print("loaded actor checkpoint:", args.init_actor_checkpoint, flush=True)

    try:
        if loaded_warmup_env_steps >= args.total_env_steps:
            raise SystemExit(
                f"--total-env-steps ({args.total_env_steps}) must exceed loaded warmup "
                f"env steps ({loaded_warmup_env_steps})."
            )

        prompt = args.prompt or get_task_prompt(args.env_id) or getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
        if wandb_run is not None:
            wandb_run.config.update({"prompt": prompt}, allow_val_change=True)
            if wandb_run.run is not None:
                wandb_run.run.summary["offline/files"] = len(h5_files)
                wandb_run.run.summary["offline/transitions"] = offline_data.size
                wandb_run.run.summary["warmup/buffer_path"] = (
                    str(warmup_buffer_path) if warmup_buffer_path is not None else ""
                )

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
        image_adapter = (
            ImageAdapter(
                rl_image_keys,
                image_size=args.rl_image_size,
                image_shape=image_shape,
            )
            if args.use_visual_rl
            else None
        )
        max_episode_steps = args.max_episode_steps or getattr(env.unwrapped, "max_episode_steps", None)
        if max_episode_steps is None and getattr(env, "spec", None) is not None:
            max_episode_steps = getattr(env.spec, "max_episode_steps", None)
        max_episode_steps = int(max_episode_steps or 500)

        path_visualizer = None
        if args.visualize_tcp_path:
            path_visualizer = TCPPathVisualizer(
                env=env,
                max_points=args.path_max_points,
                radius=args.path_radius,
                base_color=parse_rgba(args.base_path_color),
                residual_color=parse_rgba(args.residual_path_color),
                tcp_pose_key=args.tcp_pose_key,
            )

        obs, _ = env.reset(seed=args.seed + loaded_warmup_episode_idx)
        base_policy.reset()
        state = state_adapter(obs)
        image = image_adapter(obs) if image_adapter is not None else None
        base_action = base_policy.act(obs)
        if path_visualizer is not None:
            path_visualizer.clear()
            path_visualizer.set_base_prediction_from_chunk(
                obs,
                base_policy.planned_chunk(),
                position_scale=args.base_chunk_position_scale,
                max_actions=args.base_chunk_max_actions,
            )
        common.maybe_render(env, render_mode, 0, args.render_every, path_visualizer)

        episode_idx = loaded_warmup_episode_idx
        episode_return = 0.0
        episode_steps = 0
        recent_successes: deque[float] = deque(maxlen=50)
        recent_returns: deque[float] = deque(maxlen=50)
        recent_lengths: deque[float] = deque(maxlen=50)
        metrics: dict[str, float] = {}
        t0 = time.time()

        common.wandb_log(
            wandb_run,
            dict(
                env_step=loaded_warmup_env_steps,
                **{
                    "warmup/loaded": float(loaded_warmup_env_steps > 0),
                    "warmup/transitions": len(online_buffer),
                    "warmup/episode_idx": loaded_warmup_episode_idx,
                },
            ),
        )

        for env_step in range(loaded_warmup_env_steps + 1, args.total_env_steps + 1):
            if path_visualizer is not None:
                path_visualizer.set_base_prediction_from_chunk(
                    obs,
                    base_policy.planned_chunk(),
                    position_scale=args.base_chunk_position_scale,
                    max_actions=args.base_chunk_max_actions,
                )
            common.maybe_render(env, render_mode, env_step, args.render_every, path_visualizer)
            if episode_idx < args.warmup_episodes:
                action_source = "base"
                action = base_action
            else:
                action_source = "residual"
                if args.otf_rollout_actions > 0:
                    action_source = "otf_residual"
                    action = agent.select_action_otf(
                        state,
                        base_action,
                        n_actions=args.otf_rollout_actions,
                        images=image,
                        include_base_action=not args.otf_no_base_candidate,
                    )
                else:
                    action = agent.select_action(
                        state, base_action, images=image, deterministic=False
                    )

            next_obs, reward, terminated, truncated, info = env.step(action)
            if (
                path_visualizer is not None
                and args.path_every > 0
                and env_step % args.path_every == 0
                and action_source in {"residual", "otf_residual"}
            ):
                path_visualizer.add_from_obs(next_obs, "residual")
            done = common.as_done(terminated) or common.as_done(truncated)
            next_state = state_adapter(next_obs)
            next_image = image_adapter(next_obs) if image_adapter is not None else None
            next_base_action = np.zeros_like(base_action) if done else base_policy.act(next_obs)
            online_buffer.add(
                state,
                action,
                base_action,
                common.as_scalar(reward),
                next_state,
                next_base_action,
                done,
                images=image,
                next_images=next_image,
            )
            episode_return += common.as_scalar(reward)
            episode_steps += 1

            for _ in range(max(0, int(args.updates_per_env_step))):
                batch = sample_offline_online(
                    offline_buffer,
                    online_buffer,
                    args.batch_size,
                    offline_fraction=args.offline_fraction,
                )
                metrics = agent.update(batch)

            if done or episode_steps >= max_episode_steps:
                success = bool(common.to_numpy(info.get("success", False)).reshape(-1)[0])
                current_episode_idx = episode_idx
                is_warmup_episode = current_episode_idx < args.warmup_episodes
                recent_successes.append(float(success))
                recent_returns.append(float(episode_return))
                recent_lengths.append(float(episode_steps))
                print(
                    "episode",
                    dict(
                        idx=current_episode_idx,
                        env_step=env_step,
                        steps=episode_steps,
                        ret=round(episode_return, 4),
                        success=success,
                        online_buffer=len(online_buffer),
                    ),
                )
                common.wandb_log(
                    wandb_run,
                    dict(
                        env_step=env_step,
                        **{
                            "episode/index": current_episode_idx,
                            "episode/steps": episode_steps,
                            "episode/return": float(episode_return),
                            "episode/success": float(success),
                            "episode/is_warmup": float(is_warmup_episode),
                            "episode/online_buffer": len(online_buffer),
                            "episode/success_rate_50": float(np.mean(recent_successes)),
                            "episode/return_mean_50": float(np.mean(recent_returns)),
                            "episode/steps_mean_50": float(np.mean(recent_lengths)),
                        },
                    ),
                )
                episode_idx += 1
                if (
                    not warmup_buffer_saved
                    and warmup_buffer_path is not None
                    and episode_idx >= args.warmup_episodes
                ):
                    warmup_buffer_path.parent.mkdir(parents=True, exist_ok=True)
                    online_buffer.save(
                        warmup_buffer_path,
                        env_id=args.env_id,
                        episode_idx=episode_idx,
                        env_steps=env_step,
                        warmup_episodes=args.warmup_episodes,
                        base_policy=args.base_policy,
                        seed=args.seed,
                    )
                    warmup_buffer_saved = True
                    print(
                        "saved warmup buffer:",
                        dict(
                            path=str(warmup_buffer_path),
                            transitions=len(online_buffer),
                            episode_idx=episode_idx,
                            env_steps=env_step,
                        ),
                    )
                    common.wandb_log(
                        wandb_run,
                        dict(
                            env_step=env_step,
                            **{
                                "warmup/saved": 1.0,
                                "warmup/transitions": len(online_buffer),
                                "warmup/episode_idx": episode_idx,
                            },
                        ),
                    )
                obs, _ = env.reset(seed=args.seed + episode_idx)
                base_policy.reset()
                state = state_adapter(obs)
                image = image_adapter(obs) if image_adapter is not None else None
                base_action = base_policy.act(obs)
                if path_visualizer is not None:
                    path_visualizer.clear()
                    path_visualizer.set_base_prediction_from_chunk(
                        obs,
                        base_policy.planned_chunk(),
                        position_scale=args.base_chunk_position_scale,
                        max_actions=args.base_chunk_max_actions,
                    )
                common.maybe_render(env, render_mode, env_step, args.render_every, path_visualizer)
                episode_return = 0.0
                episode_steps = 0
            else:
                obs = next_obs
                state = next_state
                image = next_image
                base_action = next_base_action

            if env_step % max(1, args.log_every) == 0:
                elapsed_s = round(time.time() - t0, 1)
                print(
                    "train",
                    dict(
                        env_step=env_step,
                        updates=agent.total_updates,
                        elapsed_s=elapsed_s,
                        metrics=metrics,
                    ),
                )
                train_payload = dict(
                    env_step=env_step,
                    **{
                        "train/updates": agent.total_updates,
                        "train/elapsed_s": elapsed_s,
                        "train/online_buffer": len(online_buffer),
                        "train/offline_buffer": len(offline_buffer),
                        "train/episodes_finished": episode_idx,
                    },
                )
                if elapsed_s > 0:
                    train_payload["train/env_steps_per_s"] = float(env_step) / float(elapsed_s)
                if recent_successes:
                    train_payload["train/success_rate_50"] = float(np.mean(recent_successes))
                train_payload.update({f"train/{k}": float(v) for k, v in metrics.items()})
                common.wandb_log(wandb_run, train_payload)
            if env_step % max(1, args.save_every) == 0:
                step_ckpt = output_dir / f"residual_sac_step_{env_step}.pt"
                agent.save(step_ckpt)
                agent.save_actor(output_dir / f"residual_actor_step_{env_step}.pt")
                common.wandb_log(
                    wandb_run,
                    dict(
                        env_step=env_step,
                        **{"checkpoint/saved": 1.0, "checkpoint/actor_saved": 1.0},
                    ),
                )

        agent.save(ckpt_path)
        actor_ckpt_path = output_dir / args.actor_checkpoint_name
        agent.save_actor(actor_ckpt_path)
        common.wandb_log(
            wandb_run,
            dict(
                env_step=args.total_env_steps,
                **{"checkpoint/final": 1.0, "checkpoint/actor_saved": 1.0},
            ),
        )
        if wandb_run is not None and wandb_run.run is not None:
            wandb_run.run.summary["checkpoint/final_path"] = str(ckpt_path)
            wandb_run.run.summary["checkpoint/final_env_step"] = args.total_env_steps
        print("saved:", ckpt_path)
        print("saved actor:", actor_ckpt_path)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
