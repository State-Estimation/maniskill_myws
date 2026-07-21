#!/usr/bin/env python
"""Online ManiSkill RLT-style chunk post-training.

This script trains a lightweight PyTorch chunk actor/critic on top of a frozen
reference policy. It does not use RL tokens; the RLT state is ManiSkill
proprioception plus optional RGB features.
"""

from __future__ import annotations

import argparse
from collections import deque
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


def _as_scalar(x) -> float:
    return float(_as_numpy(x).reshape(-1)[0])


def _as_done(x) -> bool:
    return bool(_as_numpy(x).reshape(-1)[0])


def _action_bounds(action_space, action_dim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return (-1.0,) * action_dim, (1.0,) * action_dim
    low_arr = np.asarray(low, dtype=np.float32).reshape(-1)[:action_dim]
    high_arr = np.asarray(high, dtype=np.float32).reshape(-1)[:action_dim]
    if not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(high_arr)):
        return (-1.0,) * action_dim, (1.0,) * action_dim
    return tuple(float(x) for x in low_arr), tuple(float(x) for x in high_arr)


def _infer_control_dt(env, fallback_hz: float = 20.0) -> tuple[float, float]:
    control_freq = getattr(env.unwrapped, "control_freq", None)
    if control_freq is None:
        control_freq = fallback_hz
    control_freq = float(control_freq)
    if control_freq <= 0:
        control_freq = fallback_hz
    return 1.0 / control_freq, control_freq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--reward-mode", type=str, default="sparse")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--base-policy",
        choices=["remote_openpi", "zero", "random"],
        default="remote_openpi",
    )
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    parser.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    parser.add_argument("--state-keys", type=str, nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--resize", type=int, default=224)

    parser.add_argument("--use-visual-rlt", action="store_true")
    parser.add_argument("--rlt-image-keys", type=str, nargs="+", default=None)
    parser.add_argument("--rlt-image-size", type=int, default=128)
    parser.add_argument("--visual-latent-dim", type=int, default=256)

    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--total-env-steps", type=int, default=250_000)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-transitions", type=int, default=600)
    parser.add_argument(
        "--warmup-dataset",
        action="append",
        default=[],
        metavar="ROLLOUT.h5",
        help="Prefill replay from a ManiSkill rollout H5; may be repeated.",
    )
    parser.add_argument(
        "--warmup-dataset-transitions",
        type=int,
        default=None,
        help="Maximum dataset chunks to load (default: --warmup-transitions).",
    )
    parser.add_argument("--warmup-action-key", type=str, default="actions")
    parser.add_argument("--warmup-reward-key", type=str, default="rewards")
    parser.add_argument(
        "--allow-warmup-metadata-mismatch",
        action="store_true",
        help="Allow rollout env/control/reward metadata to differ from online settings.",
    )
    parser.add_argument(
        "--offline-updates",
        type=int,
        default=0,
        help="Gradient updates on prefilled replay before online rollout starts.",
    )
    parser.add_argument("--updates-per-chunk", type=int, default=5)
    parser.add_argument("--log-every-env-steps", type=int, default=1_000)
    parser.add_argument("--save-every-env-steps", type=int, default=25_000)
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pace steps at env.control_freq (enabled by default with keyboard HIL).",
    )

    parser.add_argument(
        "--hil-keyboard",
        action="store_true",
        help="Let a human gate Base/RLT actions from the SAPIEN viewer keyboard.",
    )
    parser.add_argument(
        "--hil-mode",
        choices=["hold", "latch"],
        default="hold",
        help="hold: RLT only while its key is held; latch: RLT/Base keys select control.",
    )
    parser.add_argument("--hil-rlt-key", type=str, default="r")
    parser.add_argument("--hil-base-key", type=str, default="b")
    parser.add_argument("--hil-quit-key", type=str, default="q")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--fixed-std", type=float, default=0.01)
    parser.add_argument("--reference-dropout-prob", type=float, default=0.5)
    parser.add_argument("--action-delta-scale", type=float, default=0.1)
    parser.add_argument("--bc-weight", type=float, default=5.0)
    parser.add_argument("--q-weight", type=float, default=0.1)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--gripper-smoothness-weight", type=float, default=0.1)
    parser.add_argument("--actor-update-period", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--output-dir", type=str, default="outputs/rlt/OpenSafeDoor-v2")
    parser.add_argument("--checkpoint-name", type=str, default="maniskill_rlt.pt")
    args = parser.parse_args()

    if args.hil_keyboard and str(args.render_mode).lower() != "human":
        parser.error("--hil-keyboard requires --render-mode human")
    if args.warmup_transitions < 0:
        parser.error("--warmup-transitions must be non-negative")
    if args.offline_updates < 0:
        parser.error("--offline-updates must be non-negative")
    if args.offline_updates > 0 and not args.warmup_dataset:
        parser.error("--offline-updates requires at least one --warmup-dataset")
    prefill_transitions = (
        args.warmup_transitions
        if args.warmup_dataset_transitions is None
        else args.warmup_dataset_transitions
    )
    if args.warmup_dataset and prefill_transitions <= 0:
        parser.error("dataset prefill requires a positive transition limit")
    if args.warmup_dataset and prefill_transitions > args.buffer_capacity:
        parser.error("dataset prefill transition limit exceeds --buffer-capacity")
    if args.warmup_transitions > args.buffer_capacity:
        parser.error("--warmup-transitions exceeds --buffer-capacity")

    sys.path.insert(0, str(_repo_root() / "src"))

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.rlt.dataset import load_rollout_h5_into_replay
    from maniskill_myws.rlt.hil import KeyboardInterventionGate, annotate_chunk_sources
    from maniskill_myws.rlt.policies import make_base_chunk_policy
    from maniskill_myws.rlt.replay import (
        ChunkReplayBuffer,
        ChunkTransition,
        TransitionSource,
        pad_or_trim_chunk,
        validate_pd_joint_pos_action_dim,
    )
    from maniskill_myws.rlt.state import ImageAdapter, StateAdapter
    from maniskill_myws.rlt.trainer import ManiSkillRLTAgent, RLTTrainConfig

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maniskill_myws.register()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    env = gym.make(args.env_id, **env_kwargs)
    action_dim = int(np.prod(env.action_space.shape))
    validate_pd_joint_pos_action_dim(action_dim)
    action_low, action_high = _action_bounds(env.action_space, action_dim)

    prompt = args.prompt
    if prompt is None:
        prompt = getattr(env.unwrapped, "DEFAULT_TASK_PROMPT", "")
    base_policy = make_base_chunk_policy(
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
    image_keys = args.rlt_image_keys or [args.image_key, args.wrist_image_key]

    obs, _ = env.reset(seed=args.seed)
    state = state_adapter(obs)
    image_adapter = None
    image_shape = None
    image = None
    if args.use_visual_rlt:
        image_adapter = ImageAdapter(image_keys, image_size=args.rlt_image_size)
        image = image_adapter(obs)
        image_shape = tuple(int(x) for x in image.shape)

    buffer = ChunkReplayBuffer(
        args.buffer_capacity,
        state_dim=state.shape[0],
        action_dim=action_dim,
        chunk_len=args.chunk_len,
        image_shape=image_shape,
        seed=args.seed,
    )
    prefill_episode_count = 0
    if args.warmup_dataset:
        load_stats = load_rollout_h5_into_replay(
            buffer,
            args.warmup_dataset,
            state_keys=args.state_keys,
            image_keys=image_keys if args.use_visual_rlt else None,
            image_size=args.rlt_image_size if args.use_visual_rlt else None,
            action_key=args.warmup_action_key,
            reward_key=args.warmup_reward_key,
            max_transitions=prefill_transitions,
            expected_env_id=args.env_id,
            expected_control_mode=args.control_mode,
            expected_reward_mode=args.reward_mode,
            validate_metadata=not args.allow_warmup_metadata_mismatch,
        )
        print(
            "warmup dataset loaded:",
            dict(
                files=load_stats.files,
                episodes=load_stats.episodes,
                transitions=load_stats.transitions,
                env_steps=load_stats.env_steps,
                warmup_required=args.warmup_transitions,
            ),
            flush=True,
        )
        if load_stats.transitions == 0:
            raise ValueError("Warmup datasets produced no replay transitions")
        prefill_episode_count = load_stats.episodes

    agent = ManiSkillRLTAgent(
        RLTTrainConfig(
            state_dim=state.shape[0],
            action_dim=action_dim,
            chunk_len=args.chunk_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            gamma=args.gamma,
            fixed_std=args.fixed_std,
            reference_dropout_prob=args.reference_dropout_prob,
            action_delta_scale=args.action_delta_scale,
            bc_weight=args.bc_weight,
            q_weight=args.q_weight,
            correction_weight=args.correction_weight,
            smoothness_weight=args.smoothness_weight,
            gripper_smoothness_weight=args.gripper_smoothness_weight,
            actor_lr=args.lr,
            critic_lr=args.lr,
            target_tau=args.target_tau,
            actor_update_period=args.actor_update_period,
            grad_clip_norm=args.grad_clip_norm,
            visual_encoder="resnet10" if args.use_visual_rlt else "none",
            image_shape=image_shape,
            visual_latent_dim=args.visual_latent_dim,
            action_low=action_low,
            action_high=action_high,
        ),
        device=device,
    )

    offline_metrics: dict[str, float] = {}
    if args.offline_updates > 0:
        log_interval = max(1, args.offline_updates // 10)
        for update_idx in range(1, args.offline_updates + 1):
            offline_metrics = agent.update(buffer.sample(args.batch_size))
            if update_idx % log_interval == 0 or update_idx == args.offline_updates:
                print(
                    "offline warmup:",
                    dict(
                        update=update_idx,
                        total=args.offline_updates,
                        replay=len(buffer),
                        metrics=offline_metrics,
                    ),
                    flush=True,
                )

    hil_gate = None
    if args.hil_keyboard:
        viewer = env.unwrapped.render_human()
        hil_gate = KeyboardInterventionGate(
            viewer.window,
            mode=args.hil_mode,
            rlt_key=args.hil_rlt_key,
            base_key=args.hil_base_key,
            quit_key=args.hil_quit_key,
        )
        if args.hil_mode == "hold":
            controls = (
                f"hold {args.hil_rlt_key.upper()}=RLT, release=Base, "
                f"{args.hil_base_key.upper()}=force Base"
            )
        else:
            controls = (
                f"{args.hil_rlt_key.upper()}=RLT, "
                f"{args.hil_base_key.upper()}=Base"
            )
        print(
            "HIL keyboard enabled:",
            f"{controls}, {args.hil_quit_key.upper()}=save and quit; "
            "click the SAPIEN viewer to focus it",
            flush=True,
        )

    real_time_enabled = args.hil_keyboard if args.real_time is None else args.real_time
    real_time_dt = None
    if real_time_enabled:
        real_time_dt, control_freq = _infer_control_dt(env)
        print(
            "real-time pacing:",
            dict(control_freq_hz=control_freq, target_dt_s=round(real_time_dt, 4)),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name
    max_episode_steps = args.max_episode_steps or getattr(env.unwrapped, "max_episode_steps", None)
    if max_episode_steps is None and getattr(env, "spec", None) is not None:
        max_episode_steps = getattr(env.spec, "max_episode_steps", None)
    max_episode_steps = int(max_episode_steps or 500)

    env_step = 0
    episode_idx = 0
    episode_steps = 0
    episode_return = 0.0
    episode_base_steps = 0
    episode_rlt_steps = 0
    episode_hil_switches = 0
    total_base_steps = 0
    total_rlt_steps = 0
    total_hil_switches = 0
    recent_successes: deque[float] = deque(maxlen=50)
    metrics: dict[str, float] = offline_metrics
    pending_ref_chunk: np.ndarray | None = None
    stop_requested = False
    t0 = time.time()

    try:
        while env_step < args.total_env_steps and not stop_requested:
            if pending_ref_chunk is None:
                ref_chunk = base_policy.plan(
                    obs, chunk_len=args.chunk_len, action_dim=action_dim
                )
            else:
                ref_chunk = pending_ref_chunk
                pending_ref_chunk = None

            rlt_available = len(buffer) >= args.warmup_transitions
            rlt_chunk = (
                agent.select_chunk(
                    state,
                    ref_chunk,
                    images=image,
                    deterministic=False,
                )
                if rlt_available
                else ref_chunk
            )

            actions_executed: list[np.ndarray] = []
            sources_executed: list[int] = []
            rewards: list[float] = []
            done = False
            info = {}
            next_obs = obs
            for chunk_step in range(args.chunk_len):
                step_wall_start = time.perf_counter()
                if hil_gate is not None:
                    env.unwrapped.render_human()
                    decision = hil_gate.poll(rlt_available=rlt_available)
                    if decision.quit_requested:
                        stop_requested = True
                        break
                    if decision.blocked_changed and decision.blocked_by_warmup:
                        print(
                            "HIL RLT request blocked until warmup completes:",
                            dict(
                                replay=len(buffer),
                                required=args.warmup_transitions,
                            ),
                            flush=True,
                        )
                    if decision.control_changed:
                        episode_hil_switches += 1
                        total_hil_switches += 1
                        print(
                            "HIL control:",
                            dict(
                                controller="RLT" if decision.use_rlt else "Base",
                                env_step=env_step,
                                chunk_step=chunk_step,
                            ),
                            flush=True,
                        )
                    use_rlt = decision.use_rlt
                else:
                    use_rlt = rlt_available

                if use_rlt:
                    local_action = rlt_chunk[chunk_step]
                    step_source = int(TransitionSource.RLT)
                    episode_rlt_steps += 1
                    total_rlt_steps += 1
                else:
                    local_action = ref_chunk[chunk_step]
                    step_source = int(TransitionSource.BASE)
                    episode_base_steps += 1
                    total_base_steps += 1

                next_obs, reward, terminated, truncated, info = env.step(local_action)
                if args.render_mode is not None and hil_gate is None:
                    env.render()
                actions_executed.append(np.asarray(local_action, dtype=np.float32))
                sources_executed.append(step_source)
                reward_f = _as_scalar(reward)
                rewards.append(reward_f)
                env_step += 1
                episode_steps += 1
                episode_return += reward_f
                done = (
                    _as_done(terminated)
                    or _as_done(truncated)
                    or episode_steps >= max_episode_steps
                )
                if real_time_dt is not None:
                    sleep_s = real_time_dt - (time.perf_counter() - step_wall_start)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                if done or env_step >= args.total_env_steps:
                    break

            if stop_requested:
                print(
                    "HIL quit requested; discarding current partial chunk:",
                    dict(executed_steps=len(actions_executed)),
                    flush=True,
                )
                break

            next_state = state_adapter(next_obs)
            next_image = image_adapter(next_obs) if image_adapter is not None else None
            if done:
                next_ref_chunk = np.zeros_like(ref_chunk)
                pending_ref_chunk = None
            else:
                next_ref_chunk = base_policy.plan(
                    next_obs, chunk_len=args.chunk_len, action_dim=action_dim
                )
                pending_ref_chunk = next_ref_chunk

            source, source_chunk = annotate_chunk_sources(
                sources_executed,
                chunk_len=args.chunk_len,
            )

            buffer.add(
                ChunkTransition(
                    obs=state,
                    ref_chunk=ref_chunk,
                    action_chunk=pad_or_trim_chunk(
                        np.stack(actions_executed, axis=0),
                        chunk_len=args.chunk_len,
                        action_dim=action_dim,
                    ),
                    rewards=np.asarray(rewards, dtype=np.float32),
                    done=done,
                    next_obs=next_state,
                    next_ref_chunk=next_ref_chunk,
                    images=image,
                    next_images=next_image,
                    source=source,
                    source_chunk=source_chunk,
                    episode_id=prefill_episode_count + episode_idx,
                    step_id=env_step - len(actions_executed),
                    success=int(_as_done(info.get("success", False)))
                    if isinstance(info, dict)
                    else 0,
                )
            )

            if len(buffer) >= max(args.batch_size, args.warmup_transitions):
                for _ in range(max(0, int(args.updates_per_chunk))):
                    metrics = agent.update(buffer.sample(args.batch_size))

            if done:
                success = _as_done(info.get("success", False)) if isinstance(info, dict) else False
                recent_successes.append(float(success))
                print(
                    "episode",
                    dict(
                        idx=episode_idx,
                        env_step=env_step,
                        steps=episode_steps,
                        ret=round(episode_return, 4),
                        success=success,
                        replay=len(buffer),
                        base_steps=episode_base_steps,
                        rlt_steps=episode_rlt_steps,
                        rlt_intervention_rate=round(
                            episode_rlt_steps
                            / max(1, episode_base_steps + episode_rlt_steps),
                            4,
                        ),
                        hil_switches=episode_hil_switches,
                    ),
                    flush=True,
                )
                episode_idx += 1
                obs, _ = env.reset(seed=args.seed + episode_idx)
                base_policy.reset()
                state = state_adapter(obs)
                image = image_adapter(obs) if image_adapter is not None else None
                episode_steps = 0
                episode_return = 0.0
                episode_base_steps = 0
                episode_rlt_steps = 0
                episode_hil_switches = 0
            else:
                obs = next_obs
                state = next_state
                image = next_image

            if env_step > 0 and env_step % max(1, args.log_every_env_steps) < args.chunk_len:
                elapsed_s = max(time.time() - t0, 1e-6)
                print(
                    "train",
                    dict(
                        env_step=env_step,
                        updates=agent.total_updates,
                        actor_updates=agent.actor_updates,
                        replay=len(buffer),
                        success_rate_50=round(float(np.mean(recent_successes)), 4)
                        if recent_successes
                        else None,
                        rlt_intervention_rate=round(
                            total_rlt_steps / max(1, total_base_steps + total_rlt_steps),
                            4,
                        ),
                        hil_switches=total_hil_switches,
                        env_steps_per_s=round(env_step / elapsed_s, 3),
                        metrics=metrics,
                    ),
                    flush=True,
                )

            if env_step > 0 and env_step % max(1, args.save_every_env_steps) < args.chunk_len:
                step_path = output_dir / f"maniskill_rlt_step_{env_step}.pt"
                agent.save(step_path)
                print("saved:", step_path, flush=True)

        agent.save(ckpt_path)
        print("saved:", ckpt_path, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
