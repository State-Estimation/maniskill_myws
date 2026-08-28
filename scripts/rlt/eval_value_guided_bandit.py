#!/usr/bin/env python
"""Paired deterministic evaluation for the V-gated residual bandit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_STATE_KEYS = ["agent/qpos", "agent/qvel", "extra/tcp_pose"]


def _scalar(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _done(value) -> bool:
    return bool(_scalar(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(10_000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--env-id", default="SolarPanelStatic-v2")
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-seed", type=int, default=51_000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=41_000)
    parser.add_argument("--candidate-noise-std", type=float, default=0.08)
    parser.add_argument(
        "--actor-residual-scale",
        type=float,
        default=1.0,
        help="Scale deterministic actor residuals for a trust-region evaluation audit",
    )
    parser.add_argument(
        "--actor-first-chunk-scale",
        type=float,
        default=1.0,
        help="Scale only the first deterministic actor chunk after each base/cooldown interval",
    )
    parser.add_argument(
        "--success-replay",
        default=None,
        help="Matching replay used to build a diagnostic success-retrieval memory",
    )
    parser.add_argument("--force-success-retrieval", action="store_true")
    parser.add_argument(
        "--force-candidate-search",
        action="store_true",
        help="Evaluate actor-gated chunks with scorer-selected actor/noise candidates even when the checkpoint uses deterministic actor means.",
    )
    parser.add_argument(
        "--allow-confirmed-actor-entry",
        action="store_true",
        help="Allow actor entry after a confirmed VGate event; the configured safety shield remains active.",
    )
    parser.add_argument(
        "--max-intervention-env-steps-override",
        type=int,
        default=None,
        help="Evaluation-only intervention budget override; the checkpoint identity is still verified against its stored gate configuration.",
    )
    parser.add_argument(
        "--max-consecutive-actor-chunks-override",
        type=int,
        default=None,
        help="Evaluation-only cap on consecutive deterministic actor chunks; a short base-policy cooldown follows each cap.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.control_mode != "pd_joint_pos":
        parser.error("This implementation requires --control-mode pd_joint_pos")
    if args.num_seeds <= 0 or args.start_seed < 0:
        parser.error("Evaluation seeds must be non-negative and non-empty")
    if not 0.0 <= args.candidate_noise_std <= 1.0:
        parser.error("--candidate-noise-std must lie in [0, 1]")
    if not 0.0 <= args.actor_residual_scale <= 1.0:
        parser.error("--actor-residual-scale must lie in [0, 1]")
    if not 0.0 <= args.actor_first_chunk_scale <= 1.0:
        parser.error("--actor-first-chunk-scale must lie in [0, 1]")
    if args.force_success_retrieval and not args.success_replay:
        parser.error("--force-success-retrieval requires --success-replay")
    if (
        args.max_intervention_env_steps_override is not None
        and args.max_intervention_env_steps_override <= 0
    ):
        parser.error("--max-intervention-env-steps-override must be positive")
    if (
        args.max_consecutive_actor_chunks_override is not None
        and args.max_consecutive_actor_chunks_override <= 0
    ):
        parser.error("--max-consecutive-actor-chunks-override must be positive")
    return args


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix an evaluation with {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import torch

    import maniskill_myws
    from maniskill_myws.openpi_bridge.remote_policy import (
        SAFE_LATENT_DIM,
        SAFE_LATENT_PROTOCOL,
    )
    from maniskill_myws.rlt.backend import require_resolved_backend
    from maniskill_myws.rlt.latent_actor import make_runtime_identity
    from maniskill_myws.rlt.policies import (
        inference_seed_for_step,
        make_base_chunk_policy,
        openpi_policy_identity_sha256,
    )
    from maniskill_myws.rlt.reset import reset_env_fresh_scene
    from maniskill_myws.rlt.state import StateAdapter
    from maniskill_myws.rlt.value_guided_bandit import (
        ActorChunkThrottle,
        VGate,
        VGateConfig,
        ValueBanditReplayBuffer,
        ValueGuidedBanditAgent,
        update_actor_gate_authorization,
        value_improvement_target,
    )
    from maniskill_myws.rlt.value_model import (
        VALUE_FEATURE_SCHEMA,
        DistributionalBaseValueModel,
        infer_value_estimate,
        value_images_from_observation,
    )

    np.random.seed(args.bootstrap_seed)
    torch.manual_seed(args.bootstrap_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    agent = ValueGuidedBanditAgent.load(args.checkpoint, device=device)
    config = agent.config
    if args.force_success_retrieval:
        with np.load(args.success_replay, allow_pickle=False) as replay_data:
            replay_capacity = int(replay_data["capacity"])
        replay = ValueBanditReplayBuffer(
            replay_capacity,
            state_dim=config.state_dim,
            latent_dim=config.latent_dim,
            chunk_len=config.chunk_len,
            action_dim=config.action_dim,
            seed=0,
        )
        replay.load(args.success_replay)
        if replay.last_loaded_snapshot_id != agent.snapshot_id:
            raise ValueError("Success replay and checkpoint snapshot ids differ")
        agent.refresh_success_memory(replay)
    if config.chunk_len != 10:
        raise ValueError("Bandit checkpoint does not use 10-step chunks")
    value_model, value_metadata = DistributionalBaseValueModel.load(
        args.value_checkpoint, device=device
    )
    value_model.requires_grad_(False)
    value_config = value_model.config
    stored_gate = (agent.runtime_identity or {}).get("gate")
    if not isinstance(stored_gate, dict):
        raise ValueError("Bandit checkpoint has no VGate configuration")
    checkpoint_gate_config = VGateConfig(**stored_gate)

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
        max_episode_steps=config.max_episode_steps,
    )
    backend = require_resolved_backend(
        env,
        expected_sim_backend=args.sim_backend,
        expected_render_backend=args.render_backend,
    )
    base_env = env.unwrapped
    raw_reward_schema = getattr(base_env, "grasp_reward_schema", None)
    reward_schema = (
        dict(raw_reward_schema)
        if raw_reward_schema is not None
        else {"schema": "terminal_success_sparse_reward_v1"}
    )
    action_dim = int(np.prod(env.action_space.shape))
    if action_dim != config.action_dim:
        raise ValueError("Environment action dimension differs from checkpoint")
    prompt = args.prompt or getattr(base_env, "DEFAULT_TASK_PROMPT", "")
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
    metadata = policy.server_metadata or {}
    if metadata.get("safe_latent_protocol") != SAFE_LATENT_PROTOCOL:
        raise RuntimeError("OpenPI server does not expose the required SAFE latent")
    policy_identity = openpi_policy_identity_sha256(metadata)
    state_adapter = StateAdapter(args.state_keys)
    probe_obs, _ = reset_env_fresh_scene(
        env, seed=args.start_seed, operation="bandit evaluation shape probe"
    )
    raw_state = np.asarray(state_adapter(probe_obs), dtype=np.float32)
    if value_config.state_dim != raw_state.size:
        raise ValueError("Value checkpoint state dimension differs from environment")
    dataset_metadata = value_metadata.get("dataset_metadata")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("Value checkpoint has no rollout identity")
    expected_dataset = {
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "control_mode": args.control_mode,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_keys": [args.image_key, args.wrist_image_key],
        "state_keys": list(args.state_keys),
        "chunk_len": config.chunk_len,
        "max_episode_steps": config.max_episode_steps,
        "action_dim": action_dim,
        "safe_latent_protocol": SAFE_LATENT_PROTOCOL,
        "safe_latent_dim": SAFE_LATENT_DIM,
        "openpi_policy_identity_sha256": policy_identity,
        "base_policy_only": True,
    }
    mismatches = {
        key: (expected, dataset_metadata.get(key))
        for key, expected in expected_dataset.items()
        if dataset_metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Value rollout identity mismatch: {mismatches}")
    runtime_identity = make_runtime_identity(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enhanced_determinism=args.enhanced_determinism,
        prompt=prompt,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        state_keys=args.state_keys,
        resize=args.resize,
        chunk_len=config.chunk_len,
        max_episode_steps=config.max_episode_steps,
        openpi_policy_identity_sha256=policy_identity,
        latent_protocol=SAFE_LATENT_PROTOCOL,
        latent_dim=SAFE_LATENT_DIM,
    )
    runtime_identity["environment_reward_schema"] = reward_schema
    runtime_identity["value_model"] = {
        "checkpoint_sha256": _sha256_file(Path(args.value_checkpoint)),
        "feature_schema": VALUE_FEATURE_SCHEMA,
        "training_target": "episode_success_plus_remaining_chunks_no_environment_reward",
    }
    # This is baked into the scorer weights and has no eval-time dependency.
    stored_scorer_target = (agent.runtime_identity or {}).get("scorer_target")
    if stored_scorer_target is not None:
        runtime_identity["scorer_target"] = dict(stored_scorer_target)
    runtime_identity["gate"] = asdict(checkpoint_gate_config)
    stored_behavior = (agent.runtime_identity or {}).get("behavior_policy")
    if stored_behavior is not None:
        runtime_identity["behavior_policy"] = stored_behavior
    stored_curriculum = (agent.runtime_identity or {}).get("exploration_curriculum")
    if stored_curriculum is not None:
        runtime_identity["exploration_curriculum"] = dict(stored_curriculum)
    agent.assert_runtime_identity(runtime_identity)
    gate_config = checkpoint_gate_config
    if args.max_intervention_env_steps_override is not None:
        if args.max_intervention_env_steps_override % config.chunk_len:
            raise ValueError(
                "--max-intervention-env-steps-override must be divisible by the "
                f"{config.chunk_len}-step execution chunk"
            )
        gate_config = replace(
            checkpoint_gate_config,
            max_intervention_env_steps=args.max_intervention_env_steps_override,
        )

    def plan_context(observation: dict, step_id: int, episode_seed: int):
        ref, latent = policy.plan_with_latent(
            observation,
            chunk_len=config.chunk_len,
            action_dim=action_dim,
            inference_seed=inference_seed_for_step(episode_seed, step_id),
        )
        ref = np.asarray(ref, dtype=np.float32)
        latent = np.asarray(latent, dtype=np.float32)
        if ref.shape != (config.chunk_len, action_dim):
            raise ValueError("OpenPI returned an invalid action chunk")
        if latent.shape != (SAFE_LATENT_DIM,):
            raise ValueError("OpenPI returned an invalid SAFE latent")
        raw = np.asarray(state_adapter(observation), dtype=np.float32)
        images = value_images_from_observation(
            observation,
            image_keys=(args.image_key, args.wrist_image_key),
            height=value_config.image_height,
            width=value_config.image_width,
        )
        estimate = infer_value_estimate(
            value_model,
            images=images,
            state=raw,
            latent=latent,
            ref_chunk=ref,
            step_id=step_id,
        )
        state = np.concatenate([raw, estimate.critic_features]).astype(np.float32)
        return ref, latent, state, estimate

    def run_episode(seed: int, *, use_bandit: bool) -> dict:
        obs, _ = reset_env_fresh_scene(
            env,
            seed=seed,
            operation=("V-gated bandit eval" if use_bandit else "paired base eval"),
        )
        policy.reset()
        agent.reset_deployment_state()
        gate = VGate(gate_config)
        steps = 0
        episode_return = 0.0
        success = False
        grasped = False
        interventions: list[int] = []
        gate_entries: list[int] = []
        gate_exits: list[int] = []
        trace: list[dict] = []
        actor_gate_authorized = False
        actor_safety_cooldown = 0
        actor_throttle = ActorChunkThrottle(
            max_consecutive_chunks=(
                args.max_consecutive_actor_chunks_override
                if args.max_consecutive_actor_chunks_override is not None
                else config.actor_max_consecutive_chunks
            ),
            cooldown_chunks=config.actor_throttle_cooldown_chunks,
        )
        pending = None
        done = False
        while not done:
            if use_bandit:
                if pending is None:
                    ref, latent, state, estimate = plan_context(obs, steps, seed)
                else:
                    ref, latent, state, estimate = pending
                    pending = None
                decision = gate.decide(
                    failure_probability=estimate.failure_probability,
                    entropy=estimate.entropy,
                    step_id=steps,
                )
                actor_gate_authorized = update_actor_gate_authorization(
                    actor_gate_authorized,
                    decision,
                    require_immediate_entry=(
                        config.actor_requires_immediate_gate
                        and not args.allow_confirmed_actor_entry
                    ),
                )
                if decision.event.startswith("ENTER"):
                    gate_entries.append(steps)
                if decision.event.startswith("EXIT"):
                    gate_exits.append(steps)
                deterministic_actor_on_gate = (
                    config.deterministic_actor_on_gate
                    and not args.force_candidate_search
                )
                if deterministic_actor_on_gate:
                    actor_requested = bool(
                        decision.active
                        and actor_gate_authorized
                        and actor_safety_cooldown <= 0
                    )
                    actor_throttle_decision = actor_throttle.decide(requested=actor_requested)
                    actor_gate_active = bool(
                        actor_requested and actor_throttle_decision.allowed
                    )
                    actor_throttled = actor_throttle_decision.event == "THROTTLED"
                    if actor_gate_active:
                        if args.force_success_retrieval:
                            residual, deployment = agent.propose_success_retrieval_residual(
                                state, latent, ref, step_id=steps
                            )
                        else:
                            residual, deployment = agent.propose_deployment_residual(
                                state, latent, ref, step_id=steps
                            )
                    else:
                        residual = np.zeros_like(ref)
                        deployment = {
                            "source": "VGATE_BASE",
                            "memory_index": -1,
                            "similarity": float("nan"),
                        }
                    residual *= float(args.actor_residual_scale)
                    if actor_gate_active and actor_throttle_decision.consecutive_chunks == 1:
                        residual *= float(args.actor_first_chunk_scale)
                    selected = int(actor_gate_active)
                    reason = str(deployment["source"])
                    scores = agent.score_candidates(
                        state,
                        latent,
                        ref,
                        np.stack([np.zeros_like(ref), residual]),
                        step_id=steps,
                    )
                    score_index = 1
                else:
                    actor_throttled = False
                    candidates = agent.propose_candidates(
                        state,
                        latent,
                        ref,
                        step_id=steps,
                        noise_std=args.candidate_noise_std,
                        seed=inference_seed_for_step(seed, steps, stream=91),
                    )
                    scores = agent.score_candidates(state, latent, ref, candidates, step_id=steps)
                    selected, reason = (
                        agent.choose_candidate(scores, allow_exploration=False)
                        if decision.active
                        else (0, "VGATE_BASE")
                    )
                    residual = candidates[selected]
                    score_index = selected
                action_chunk = agent.apply_residual(ref, residual)
                intervened = bool(np.any(np.abs(residual) > 1e-6))
                if intervened:
                    interventions.append(steps)
                trace.append(
                    {
                        "step": steps,
                        "failure_probability": estimate.failure_probability,
                        "value_entropy": estimate.entropy,
                        "value_potential": estimate.potential,
                        "gate_active": decision.active,
                        "actor_gate_active": actor_gate_active
                        if deterministic_actor_on_gate
                        else decision.active,
                        "actor_throttled": actor_throttled,
                        "actor_consecutive_chunks": (
                            actor_throttle_decision.consecutive_chunks
                            if deterministic_actor_on_gate
                            else 0
                        ),
                        "actor_throttle_cooldown": (
                            actor_throttle_decision.cooldown_remaining
                            if deterministic_actor_on_gate
                            else 0
                        ),
                        "actor_safety_cooldown": actor_safety_cooldown,
                        "gate_event": decision.event,
                        "smoothed_failure_probability": decision.smoothed_failure_probability,
                        "selected_candidate": selected,
                        "selection_reason": reason,
                        "retrieval_memory_index": int(deployment["memory_index"])
                        if deterministic_actor_on_gate
                        else -1,
                        "retrieval_similarity": (
                            float(deployment["similarity"])
                            if deterministic_actor_on_gate
                            and np.isfinite(float(deployment["similarity"]))
                            else None
                        ),
                        "selected_lcb": float(scores["lcb"][score_index]),
                        "selected_advantage_mean": float(scores["advantage_mean"][score_index]),
                        "selected_advantage_std": float(scores["advantage_std"][score_index]),
                    }
                )
            else:
                action_chunk = policy.plan(
                    obs,
                    chunk_len=config.chunk_len,
                    action_dim=action_dim,
                    inference_seed=inference_seed_for_step(seed, steps),
                )
                selected = 0

            duration = 0
            for action in action_chunk:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += _scalar(reward)
                steps += 1
                duration += 1
                if isinstance(info, dict):
                    success |= _done(info.get("success", False))
                    grasped |= _done(info.get("grasp_reward_event", False))
                if args.render_mode is not None:
                    env.render()
                done = bool(
                    _done(terminated) or _done(truncated) or steps >= config.max_episode_steps
                )
                if done:
                    break
            if use_bandit:
                gate.observe_execution(duration=duration, intervened=intervened)
                if not done:
                    pending = plan_context(obs, steps, seed)
                    improvement = value_improvement_target(
                        current_potential=estimate.potential,
                        next_potential=pending[3].potential,
                        terminal=False,
                        success=False,
                        failure_value=value_config.failure_value,
                    )
                    trace[-1]["observed_value_improvement"] = float(improvement)
                    if (
                        config.actor_safety_enabled
                        and deterministic_actor_on_gate
                    ):
                        if (
                            intervened
                            and improvement < config.actor_safety_min_value_improvement
                        ):
                            actor_safety_cooldown = config.actor_safety_cooldown_chunks
                        elif actor_safety_cooldown > 0:
                            actor_safety_cooldown -= 1
        return {
            "success": success,
            "grasped": grasped,
            "return": episode_return,
            "steps": steps,
            "intervention_chunks": len(interventions),
            "intervention_steps": interventions,
            "gate_entries": gate_entries,
            "gate_exits": gate_exits,
            "boundary_trace": trace,
        }

    rows: list[dict] = []
    results_path = output_dir / "paired_results.jsonl"
    started = time.time()
    try:
        for index in range(args.num_seeds):
            seed = args.start_seed + index
            base = run_episode(seed, use_bandit=False)
            bandit = run_episode(seed, use_bandit=True)
            row = {"index": index, "seed": seed, "base": base, "bandit": bandit}
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, allow_nan=False) + "\n")
            print("pair", row, flush=True)
    finally:
        env.close()

    base_success = np.asarray([row["base"]["success"] for row in rows], np.int8)
    bandit_success = np.asarray([row["bandit"]["success"] for row in rows], np.int8)
    base_grasp = np.asarray([row["base"]["grasped"] for row in rows], np.int8)
    bandit_grasp = np.asarray([row["bandit"]["grasped"] for row in rows], np.int8)
    delta = bandit_success - base_success
    summary = {
        "schema": "safe_value_guided_chunk_bandit_paired_eval_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "value_checkpoint": str(Path(args.value_checkpoint).resolve()),
        "backend": backend,
        "start_seed": args.start_seed,
        "num_seeds": len(rows),
        "base_successes": int(base_success.sum()),
        "base_success_rate": float(base_success.mean()),
        "bandit_successes": int(bandit_success.sum()),
        "bandit_success_rate": float(bandit_success.mean()),
        "paired_delta": float(delta.mean()),
        "paired_delta_bootstrap_95ci": _bootstrap_interval(delta, args.bootstrap_seed),
        "rescues": int(np.count_nonzero((base_success == 0) & (bandit_success == 1))),
        "regressions": int(np.count_nonzero((base_success == 1) & (bandit_success == 0))),
        "base_grasp_rate": float(base_grasp.mean()),
        "bandit_grasp_rate": float(bandit_grasp.mean()),
        "episodes_with_gate_entry": int(sum(bool(row["bandit"]["gate_entries"]) for row in rows)),
        "episodes_with_intervention": int(
            sum(row["bandit"]["intervention_chunks"] > 0 for row in rows)
        ),
        "intervention_chunks": int(sum(row["bandit"]["intervention_chunks"] for row in rows)),
        "checkpoint_gate_config": asdict(checkpoint_gate_config),
        "gate_config": asdict(gate_config),
        "max_intervention_env_steps_override": args.max_intervention_env_steps_override,
        "max_consecutive_actor_chunks_override": args.max_consecutive_actor_chunks_override,
        "effective_actor_max_consecutive_chunks": (
            args.max_consecutive_actor_chunks_override
            if args.max_consecutive_actor_chunks_override is not None
            else config.actor_max_consecutive_chunks
        ),
        "actor_throttle_cooldown_chunks": config.actor_throttle_cooldown_chunks,
        "candidate_noise_std": args.candidate_noise_std,
        "actor_residual_scale": args.actor_residual_scale,
        "actor_first_chunk_scale": args.actor_first_chunk_scale,
        "deterministic_actor_on_gate": config.deterministic_actor_on_gate,
        "force_candidate_search": args.force_candidate_search,
        "allow_confirmed_actor_entry": args.allow_confirmed_actor_entry,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
