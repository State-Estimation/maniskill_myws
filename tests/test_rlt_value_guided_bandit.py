from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.latent_actor import SAFE_ENDPOINT_LATENT_ENCODER
from maniskill_myws.rlt.value_guided_bandit import (
    ActorChunkThrottle,
    ExplorationBurstSchedule,
    PersistentResidualExplorer,
    SmoothKnotResidualExplorer,
    VGate,
    VGateConfig,
    VGateDecision,
    ValueBanditBatch,
    ValueBanditReplayBuffer,
    ValueGuidedBanditAgent,
    ValueGuidedBanditConfig,
    linear_curriculum_value,
    update_actor_gate_authorization,
    value_improvement_target,
)


def _config(**overrides) -> ValueGuidedBanditConfig:
    values = dict(
        state_dim=7,
        latent_dim=SAFE_LATENT_DIM,
        latent_protocol=SAFE_LATENT_PROTOCOL,
        latent_encoder=SAFE_ENDPOINT_LATENT_ENCODER,
        context_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_scorers=3,
        candidate_count=6,
        exploration_knots=3,
        actor_update_period=1,
        action_low=(-1.0,) * 8,
        action_high=(1.0,) * 8,
    )
    values.update(overrides)
    return ValueGuidedBanditConfig(**values)


def _inputs(config: ValueGuidedBanditConfig):
    state = np.zeros(config.state_dim, dtype=np.float32)
    latent = np.zeros(config.latent_dim, dtype=np.float32)
    ref = np.zeros((config.chunk_len, config.action_dim), dtype=np.float32)
    return state, latent, ref


def test_value_improvement_uses_only_frozen_value_utility() -> None:
    assert value_improvement_target(
        current_potential=-0.8,
        next_potential=-0.5,
        terminal=False,
        success=False,
        failure_value=-1.25,
    ) == pytest.approx(0.3)
    assert value_improvement_target(
        current_potential=-0.2,
        next_potential=None,
        terminal=True,
        success=True,
        failure_value=-1.25,
    ) == pytest.approx(0.2)
    assert value_improvement_target(
        current_potential=-0.4,
        next_potential=None,
        terminal=True,
        success=False,
        failure_value=-1.25,
    ) == pytest.approx(-0.85)
    with pytest.raises(ValueError, match="non-terminal"):
        value_improvement_target(
            current_potential=-0.4,
            next_potential=-0.2,
            terminal=False,
            success=True,
            failure_value=-1.25,
        )


def test_linear_curriculum_holds_then_anneals_and_clamps() -> None:
    schedule = dict(
        start_step=20_000,
        anneal_steps=100_000,
        start_value=0.30,
        end_value=0.10,
    )
    assert linear_curriculum_value(0, **schedule) == pytest.approx(0.30)
    assert linear_curriculum_value(20_000, **schedule) == pytest.approx(0.30)
    assert linear_curriculum_value(70_000, **schedule) == pytest.approx(0.20)
    assert linear_curriculum_value(120_000, **schedule) == pytest.approx(0.10)
    assert linear_curriculum_value(200_000, **schedule) == pytest.approx(0.10)


def test_linear_curriculum_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        linear_curriculum_value(
            -1, start_step=0, anneal_steps=1, start_value=0.3, end_value=0.1
        )
    with pytest.raises(ValueError, match="positive"):
        linear_curriculum_value(
            0, start_step=0, anneal_steps=0, start_value=0.3, end_value=0.1
        )


def test_vgate_confirmation_hysteresis_and_budget() -> None:
    config = VGateConfig(
        enter_failure_probability=0.6,
        immediate_failure_probability=0.9,
        exit_failure_probability=0.2,
        enter_confirm_chunks=2,
        exit_confirm_chunks=2,
        min_active_chunks=1,
        ema_alpha=1.0,
        immediate_max_entropy=1.0,
        latest_entry_step=100,
        max_intervention_env_steps=20,
    )
    gate = VGate(config)
    first = gate.decide(failure_probability=0.7, entropy=2.0, step_id=0)
    second = gate.decide(failure_probability=0.7, entropy=2.0, step_id=10)
    assert not first.active and first.event == "IDLE_ENTER_CONFIRM"
    assert second.active and second.event == "ENTER_CONFIRMED"
    assert gate.decide(failure_probability=0.1, entropy=0.0, step_id=20).active
    recovered = gate.decide(failure_probability=0.1, entropy=0.0, step_id=30)
    assert not recovered.active and recovered.event == "EXIT_RECOVERED"

    immediate = VGate(config)
    entered = immediate.decide(failure_probability=0.95, entropy=0.5, step_id=0)
    assert entered.active and entered.event == "ENTER_IMMEDIATE"
    immediate.observe_execution(duration=20, intervened=True)
    exhausted = immediate.decide(failure_probability=0.95, entropy=0.5, step_id=10)
    assert not exhausted.active and exhausted.event == "EXIT_BUDGET"
    still_exhausted = immediate.decide(failure_probability=0.95, entropy=0.5, step_id=20)
    assert not still_exhausted.active
    assert still_exhausted.event == "IDLE_BUDGET_EXHAUSTED"


def test_candidates_are_deterministic_and_scored_relative_to_base() -> None:
    torch.manual_seed(5)
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    state, latent, ref = _inputs(config)
    candidates = agent.propose_candidates(state, latent, ref, step_id=0, noise_std=0.2, seed=17)
    repeated = agent.propose_candidates(state, latent, ref, step_id=0, noise_std=0.2, seed=17)
    assert candidates.shape == (6, 10, 8)
    np.testing.assert_array_equal(candidates, repeated)
    np.testing.assert_array_equal(candidates[0], np.zeros((10, 8), np.float32))
    np.testing.assert_allclose(candidates[2], -candidates[4], atol=1e-6)

    scores = agent.score_candidates(state, latent, ref, candidates, step_id=0)
    np.testing.assert_array_equal(scores["advantage_heads"][0], np.zeros(3, np.float32))
    assert scores["lcb"][0] == pytest.approx(0.0)


def test_alternative_vla_chunk_is_projected_into_residual_trust_region() -> None:
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    _, _, ref = _inputs(config)
    alternative = ref.copy()
    alternative[:, 0] = 0.1
    alternative[:, -1] = -0.1

    residual = agent.normalized_residual(ref, alternative)

    assert residual.dtype == np.float32
    assert np.max(np.abs(residual)) <= 1.0
    np.testing.assert_allclose(agent.apply_residual(ref, residual), alternative, atol=1e-6)


def test_alternative_vla_chunk_residual_clips_large_deviations() -> None:
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    _, _, ref = _inputs(config)
    alternative = np.full_like(ref, 2.0)

    residual = agent.normalized_residual(ref, alternative)

    np.testing.assert_array_equal(residual, np.ones_like(ref))
    projected = agent.apply_residual(ref, residual)
    assert np.all(projected < alternative)


def test_vla_tangent_line_search_scales_and_clips_direction() -> None:
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    _, _, ref = _inputs(config)
    alternative = ref.copy()
    alternative[:, 0] = 0.03

    residuals = agent.vla_tangent_residuals(
        ref, alternative, scales=(1.0, 2.0, 4.0, 8.0)
    )

    assert residuals.shape == (4, config.chunk_len, config.action_dim)
    np.testing.assert_allclose(
        residuals[:, :, 0],
        np.repeat(
            np.asarray((0.25, 0.5, 1.0, 1.0), np.float32)[:, None],
            config.chunk_len,
            axis=1,
        ),
    )
    np.testing.assert_array_equal(residuals[:, :, 1:], 0.0)


def test_vla_tangent_line_search_rejects_invalid_scales() -> None:
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    _, _, ref = _inputs(config)

    with pytest.raises(ValueError, match="finite and positive"):
        agent.vla_tangent_residuals(ref, ref, scales=(1.0, 0.0))


def test_candidate_selection_uses_lcb_then_ucb_exploration() -> None:
    config = _config()
    agent = ValueGuidedBanditAgent(config)
    scores = {
        "lcb": np.asarray([0.0, 0.005, 0.03, -0.1, 0.0, 0.0], np.float32),
        "advantage_mean": np.asarray([0.0, 0.1, 0.02, 0.0, 0.0, 0.0], np.float32),
        "advantage_std": np.asarray([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], np.float32),
        "residual_rms": np.zeros(6, np.float32),
    }
    assert agent.choose_candidate(scores, allow_exploration=False) == (2, "LCB_SELECTED")
    scores["lcb"][2] = 0.005
    assert agent.choose_candidate(scores, allow_exploration=False) == (0, "BASE_FALLBACK")
    assert agent.choose_candidate(scores, allow_exploration=True) == (1, "UCB_EXPLORE")


def test_replay_keeps_environment_reward_separate_from_value_target(tmp_path) -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        4,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=3,
    )
    replay.add(
        state=state,
        latent=latent,
        ref_chunk=ref,
        residual=np.zeros_like(ref),
        value_improvement=0.2,
        environment_return=1.25,
        duration=10,
        step_id=20,
        gate_active=True,
        exploration_active=False,
        terminal_success=True,
        terminal_failure=False,
        episode_id=7,
    )
    assert replay.finalize_episode(7, success=True) == 1
    path = tmp_path / "replay.npz"
    replay.save(path, snapshot_id="snapshot-1")
    restored = ValueBanditReplayBuffer(
        4,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=999,
    )
    assert restored.load(path) == 1
    batch = restored.batch(np.asarray([0]))
    assert batch.value_improvements[0] == pytest.approx(0.2)
    assert batch.environment_returns[0] == pytest.approx(1.25)
    assert batch.outcome_successes[0] == 1
    assert restored.last_loaded_snapshot_id == "snapshot-1"


def test_replay_success_pool_requires_exploratory_nonzero_correction() -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        4,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=4,
    )
    for episode_id, exploration_active in enumerate((False, True)):
        replay.add(
            state=state,
            latent=latent,
            ref_chunk=ref,
            residual=np.full_like(ref, 0.1),
            value_improvement=0.1,
            environment_return=0.0,
            duration=10,
            step_id=0,
            gate_active=False,
            exploration_active=exploration_active,
            terminal_success=False,
            terminal_failure=False,
            episode_id=episode_id,
        )
        replay.finalize_episode(episode_id, success=True)
    assert replay.pool_counts()["successful_nonzero"] == 1


def test_replay_credits_only_best_net_positive_success_burst() -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        8,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=5,
    )
    # The first burst crosses a temporary V decrease but has the largest net
    # recovery.  A later isolated positive chunk must not receive credit.
    for step_id, improvement in zip(
        (0, 10, 20, 40), (0.4, -0.1, 0.2, 0.3), strict=True
    ):
        replay.add(
            state=state,
            latent=latent,
            ref_chunk=ref,
            residual=np.full_like(ref, 0.1),
            value_improvement=improvement,
            environment_return=0.0,
            duration=10,
            step_id=step_id,
            gate_active=True,
            exploration_active=True,
            terminal_success=False,
            terminal_failure=False,
            episode_id=3,
        )
    replay.finalize_episode(
        3,
        success=True,
        success_credit_mode="best_positive_burst",
    )
    np.testing.assert_array_equal(
        replay.batch(np.arange(4)).success_credited,
        np.asarray([True, True, True, False]),
    )
    assert replay.pool_counts()["successful_nonzero"] == 3


def test_replay_rejects_nonpositive_burst_success_credit() -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        2,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=6,
    )
    replay.add(
        state=state,
        latent=latent,
        ref_chunk=ref,
        residual=np.full_like(ref, 0.1),
        value_improvement=0.0,
        environment_return=0.0,
        duration=10,
        step_id=0,
        gate_active=True,
        exploration_active=True,
        terminal_success=True,
        terminal_failure=False,
        episode_id=4,
    )
    replay.finalize_episode(
        4,
        success=True,
        success_credit_mode="best_positive_burst",
    )
    assert not replay.batch(np.asarray([0])).success_credited[0]


def test_success_retrieval_preserves_credited_mode_and_roundtrips(tmp_path) -> None:
    config = _config(
        actor_context_trainable=True,
        actor_deployment_policy="success_retrieval",
        retrieval_max_step_distance=20,
    )
    _, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        4,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=7,
    )
    states = (
        np.full(config.state_dim, -1.0, np.float32),
        np.full(config.state_dim, 1.0, np.float32),
    )
    residuals = (
        np.full_like(ref, -0.2),
        np.full_like(ref, 0.2),
    )
    for episode_id in range(2):
        replay.add(
            state=states[episode_id],
            latent=latent,
            ref_chunk=ref,
            residual=residuals[episode_id],
            value_improvement=0.2,
            environment_return=0.0,
            duration=10,
            step_id=episode_id * 10,
            gate_active=True,
            exploration_active=True,
            terminal_success=False,
            terminal_failure=False,
            episode_id=episode_id,
        )
        if episode_id == 0:
            replay.finalize_episode(episode_id, success=True)
    continuation_residual = np.full_like(ref, 0.3)
    replay.add(
        state=states[1],
        latent=latent,
        ref_chunk=ref,
        residual=continuation_residual,
        value_improvement=0.2,
        environment_return=0.0,
        duration=10,
        step_id=20,
        gate_active=True,
        exploration_active=True,
        terminal_success=False,
        terminal_failure=False,
        episode_id=1,
    )
    replay.finalize_episode(1, success=True)
    agent = ValueGuidedBanditAgent(config)
    assert agent.refresh_success_memory(replay) == 3
    retrieved, metadata = agent.propose_deployment_residual(
        states[1], latent, ref, step_id=10
    )
    np.testing.assert_allclose(retrieved, residuals[1], atol=1e-6)
    assert metadata["source"] == "SUCCESS_RETRIEVAL_SEARCH"
    assert metadata["memory_index"] == 1
    assert metadata["similarity"] == pytest.approx(1.0)
    continued, continued_metadata = agent.propose_deployment_residual(
        states[0], latent, ref, step_id=20
    )
    np.testing.assert_array_equal(continued, continuation_residual)
    assert continued_metadata["source"] == "SUCCESS_RETRIEVAL_CONTINUE"
    assert continued_metadata["memory_index"] == 2

    path = tmp_path / "retrieval-agent.pt"
    agent.save(path, snapshot_id="retrieval")
    restored = ValueGuidedBanditAgent.load(path)
    restored.reset_deployment_state()
    repeated, repeated_metadata = restored.propose_deployment_residual(
        states[1], latent, ref, step_id=10
    )
    np.testing.assert_array_equal(repeated, retrieved)
    assert repeated_metadata == metadata


def test_episode_mc_relabels_ordered_chunk_targets() -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        4,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=3,
    )
    for step, target in ((20, 0.3), (0, 0.1), (10, 0.2)):
        replay.add(
            state=state,
            latent=latent,
            ref_chunk=ref,
            residual=np.zeros_like(ref),
            value_improvement=target,
            environment_return=0.0,
            duration=10,
            step_id=step,
            gate_active=False,
            exploration_active=False,
            terminal_success=False,
            terminal_failure=False,
            episode_id=9,
        )
    replay.finalize_episode(
        9,
        success=False,
        scorer_return_mode="episode_mc",
        scorer_return_gamma=0.9,
    )
    by_step = {
        int(step): float(value)
        for step, value in zip(
            replay.step_ids[:3], replay.value_improvements[:3], strict=True
        )
    }
    assert by_step[20] == pytest.approx(0.3)
    assert by_step[10] == pytest.approx(0.2 + 0.9 * 0.3)
    assert by_step[0] == pytest.approx(0.1 + 0.9 * (0.2 + 0.9 * 0.3))


def test_bounded_reward_trace_only_credits_recent_exploration() -> None:
    config = _config()
    state, latent, ref = _inputs(config)
    replay = ValueBanditReplayBuffer(
        8,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=5,
    )
    for boundary in range(6):
        replay.add(
            state=state,
            latent=latent,
            ref_chunk=ref,
            residual=np.full_like(ref, 0.1) if boundary != 2 else np.zeros_like(ref),
            value_improvement=0.01,
            environment_return=0.25 if boundary == 5 else 0.0,
            duration=10,
            step_id=10 * boundary,
            gate_active=True,
            exploration_active=boundary != 2,
            terminal_success=False,
            terminal_failure=False,
            episode_id=11,
        )
    replay.finalize_episode(
        11,
        success=True,
        scorer_return_mode="bounded_reward_trace",
        scorer_return_gamma=0.9,
        scorer_trace_chunks=3,
    )
    np.testing.assert_allclose(replay.value_improvements[:3], 0.01)
    assert replay.value_improvements[3] == pytest.approx(0.01 + 0.9**2 * 0.25)
    assert replay.value_improvements[4] == pytest.approx(0.01 + 0.9 * 0.25)
    assert replay.value_improvements[5] == pytest.approx(0.01)


def test_persistent_explorer_is_chunk_continuous_and_resettable() -> None:
    explorer = PersistentResidualExplorer(8, correlation=0.99, seed=11, gripper_scale=0.25)
    first = explorer.sample(10, std=0.3)
    second = explorer.sample(10, std=0.3)
    assert first.shape == (10, 8)
    assert np.linalg.norm(second[0] - first[-1]) < 0.5
    assert np.std(first[:, -1]) < np.std(first[:, :-1])

    explorer.reset()
    reset = explorer.sample(10, std=0.3)
    assert not np.array_equal(second, reset)


def test_smooth_knot_explorer_slices_one_macro_burst() -> None:
    explorer = SmoothKnotResidualExplorer(8, knot_count=6, seed=11, gripper_scale=0.25)
    explorer.start_burst(total_steps=50, std=0.3)
    chunks = [explorer.sample(10, std=0.3) for _ in range(5)]
    trajectory = np.concatenate(chunks, axis=0)
    assert trajectory.shape == (50, 8)
    assert np.std(trajectory[:, -1]) < np.std(trajectory[:, :-1])
    curvature = np.max(np.abs(np.diff(trajectory, n=2, axis=0)), axis=1)
    assert np.count_nonzero(curvature > 1e-5) <= 2 * (explorer.knot_count - 2)
    with pytest.raises(RuntimeError, match="exhausted"):
        explorer.sample(10, std=0.3)

    repeated = SmoothKnotResidualExplorer(8, knot_count=6, seed=11, gripper_scale=0.25)
    repeated.start_burst(total_steps=50, std=0.3)
    np.testing.assert_array_equal(trajectory[:10], repeated.sample(10, std=0.3))


def test_exploration_bursts_are_bounded_and_have_cooldown() -> None:
    schedule = ExplorationBurstSchedule(burst_chunks=3, max_bursts=2, cooldown_chunks=2)
    decisions = [schedule.decide(gate_active=True, start_requested=True) for _ in range(9)]
    assert [decision.explore for decision in decisions] == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert decisions[0].event == "BURST_START"
    assert decisions[2].event == "BURST_END"
    assert decisions[-1].event == "BUDGET_EXHAUSTED"


def test_actor_chunk_throttle_matches_two_then_three_base_chunks() -> None:
    throttle = ActorChunkThrottle(max_consecutive_chunks=2, cooldown_chunks=3)
    decisions = [throttle.decide(requested=True) for _ in range(7)]
    assert [decision.allowed for decision in decisions] == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert decisions[2].event == "THROTTLED"
    assert decisions[2].cooldown_remaining == 2
    assert decisions[4].event == "COOLDOWN"
    assert decisions[5].consecutive_chunks == 1


def test_actor_chunk_throttle_resets_after_base_chunk() -> None:
    throttle = ActorChunkThrottle(max_consecutive_chunks=2, cooldown_chunks=3)
    assert throttle.decide(requested=True).allowed
    idle = throttle.decide(requested=False)
    assert idle.event == "IDLE"
    assert throttle.decide(requested=True).consecutive_chunks == 1


def test_actor_gate_authorization_rejects_delayed_entry_when_required() -> None:
    immediate = VGateDecision(
        active=True,
        event="ENTER_IMMEDIATE",
        smoothed_failure_probability=0.9,
        risk_confirmations=0,
        recovery_confirmations=0,
        active_chunks=1,
        intervention_env_steps=0,
    )
    confirmed = VGateDecision(
        active=True,
        event="ENTER_CONFIRMED",
        smoothed_failure_probability=0.7,
        risk_confirmations=0,
        recovery_confirmations=0,
        active_chunks=1,
        intervention_env_steps=0,
    )
    hold = VGateDecision(
        active=True,
        event="HOLD",
        smoothed_failure_probability=0.7,
        risk_confirmations=0,
        recovery_confirmations=0,
        active_chunks=2,
        intervention_env_steps=10,
    )
    exited = VGateDecision(
        active=False,
        event="EXIT_RECOVERED",
        smoothed_failure_probability=0.2,
        risk_confirmations=0,
        recovery_confirmations=0,
        active_chunks=2,
        intervention_env_steps=20,
    )
    assert not update_actor_gate_authorization(False, confirmed, require_immediate_entry=True)
    assert update_actor_gate_authorization(False, immediate, require_immediate_entry=True)
    assert update_actor_gate_authorization(True, hold, require_immediate_entry=True)
    assert not update_actor_gate_authorization(True, exited, require_immediate_entry=True)
    assert update_actor_gate_authorization(False, confirmed, require_immediate_entry=False)


def test_update_and_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(9)
    config = _config()
    batch_size = 3
    zeros = np.zeros((batch_size, config.chunk_len, config.action_dim), np.float32)
    residuals = zeros.copy()
    residuals[1:] = 0.1
    batch = ValueBanditBatch(
        states=np.zeros((batch_size, config.state_dim), np.float32),
        latents=np.zeros((batch_size, config.latent_dim), np.float32),
        ref_chunks=zeros.copy(),
        residuals=residuals,
        value_improvements=np.asarray([0.0, 0.2, -0.1], np.float32),
        environment_returns=np.asarray([0.0, 1.25, 0.0], np.float32),
        durations=np.asarray([10, 7, 10], np.int32),
        step_ids=np.asarray([0, 10, 20], np.int32),
        gate_active=np.asarray([False, True, True]),
        exploration_active=np.asarray([False, True, False]),
        terminal_success=np.asarray([False, True, False]),
        terminal_failure=np.asarray([False, False, True]),
        episode_ids=np.asarray([0, 1, 2], np.int64),
        # Only the sample explicitly marked as exploratory is a success-BC
        # target; gate-active actor continuations are not self-imitation data.
        outcome_successes=np.asarray([0, 1, 1], np.int8),
        success_credited=np.asarray([False, True, False]),
    )
    identity = {"env_id": "unit-test"}
    agent = ValueGuidedBanditAgent(config, runtime_identity=identity)
    metrics = agent.update(batch, update_actor=True)
    assert np.isfinite(metrics["scorer_loss"])
    assert metrics["actor_updated"] == 1.0
    assert metrics["actor_success_bc_samples"] == 1.0
    assert all(parameter.requires_grad for parameter in agent.scorer.parameters())

    path = tmp_path / "agent.pt"
    agent.save(path, snapshot_id="snapshot-2")
    restored = ValueGuidedBanditAgent.load(path)
    assert restored.config == config
    assert restored.total_updates == 1
    assert restored.snapshot_id == "snapshot-2"
    restored.assert_runtime_identity(identity)
    assert {field.name for field in fields(ValueGuidedBanditConfig)} == set(
        restored.config.__dataclass_fields__
    )


def test_trainable_actor_context_receives_actor_gradient_and_roundtrips(tmp_path) -> None:
    torch.manual_seed(12)
    config = _config(
        actor_context_trainable=True,
        actor_value_objective_weight=0.0,
        actor_awr_weight=0.0,
        actor_success_bc_weight=1.0,
    )
    zeros = np.zeros((2, config.chunk_len, config.action_dim), np.float32)
    batch = ValueBanditBatch(
        states=np.ones((2, config.state_dim), np.float32),
        latents=np.ones((2, config.latent_dim), np.float32),
        ref_chunks=zeros.copy(),
        residuals=np.full_like(zeros, 0.1),
        value_improvements=np.asarray([0.2, 0.2], np.float32),
        environment_returns=np.asarray([1.0, 1.0], np.float32),
        durations=np.asarray([10, 10], np.int32),
        step_ids=np.asarray([0, 10], np.int32),
        gate_active=np.asarray([True, False]),
        exploration_active=np.asarray([True, True]),
        terminal_success=np.asarray([False, True]),
        terminal_failure=np.asarray([False, False]),
        episode_ids=np.asarray([0, 0], np.int64),
        outcome_successes=np.asarray([1, 1], np.int8),
        success_credited=np.asarray([True, True]),
    )
    agent = ValueGuidedBanditAgent(config)
    assert agent.actor_context is not None
    assert not agent.actor_context_initialized
    assert agent._initialize_actor_context_from_scorer()
    assert agent.actor_context_initialized
    for expected, actual in zip(
        agent.context.parameters(), agent.actor_context.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)
    assert not agent._initialize_actor_context_from_scorer()
    before = [parameter.detach().clone() for parameter in agent.actor_context.parameters()]

    agent.update(batch, update_actor=True)
    metrics = agent.update(batch, update_actor=True)

    assert metrics["actor_success_bc_samples"] == 2.0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in agent.actor_context.parameters()
    )
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, agent.actor_context.parameters(), strict=True)
    )
    path = tmp_path / "agent-with-actor-context.pt"
    agent.save(path, snapshot_id="actor-context")
    restored = ValueGuidedBanditAgent.load(path)
    assert restored.actor_context is not None
    assert restored.actor_context_initialized
    for expected, actual in zip(
        agent.actor_context.parameters(), restored.actor_context.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)

    gated_agent = ValueGuidedBanditAgent(
        _config(
            actor_context_trainable=True,
            actor_value_objective_weight=0.0,
            actor_awr_weight=0.0,
            actor_success_bc_weight=1.0,
            actor_success_bc_requires_gate=True,
        )
    )
    gated_metrics = gated_agent.update(batch, update_actor=True)
    assert gated_metrics["actor_success_bc_samples"] == 1.0


def test_multi_hypothesis_actor_distributes_tied_anchors_and_roundtrips(tmp_path) -> None:
    torch.manual_seed(18)
    config = _config(
        actor_context_trainable=True,
        actor_hypotheses=3,
        actor_hypothesis_loss_weight=1.0,
        actor_value_objective_weight=0.0,
        actor_awr_weight=0.0,
        actor_success_bc_weight=1.0,
    )
    batch_size = 6
    zeros = np.zeros((batch_size, config.chunk_len, config.action_dim), np.float32)
    residuals = np.stack(
        [
            np.full((config.chunk_len, config.action_dim), value, np.float32)
            for value in (-0.3, 0.3, -0.2, 0.2, -0.1, 0.1)
        ]
    )
    batch = ValueBanditBatch(
        states=np.ones((batch_size, config.state_dim), np.float32),
        latents=np.ones((batch_size, config.latent_dim), np.float32),
        ref_chunks=zeros,
        residuals=residuals,
        value_improvements=np.full(batch_size, 0.2, np.float32),
        environment_returns=np.ones(batch_size, np.float32),
        durations=np.full(batch_size, config.chunk_len, np.int32),
        step_ids=np.arange(batch_size, dtype=np.int32) * config.chunk_len,
        gate_active=np.ones(batch_size, dtype=np.bool_),
        exploration_active=np.ones(batch_size, dtype=np.bool_),
        terminal_success=np.zeros(batch_size, dtype=np.bool_),
        terminal_failure=np.zeros(batch_size, dtype=np.bool_),
        episode_ids=np.arange(batch_size, dtype=np.int64),
        outcome_successes=np.ones(batch_size, dtype=np.int8),
        success_credited=np.ones(batch_size, dtype=np.bool_),
    )
    agent = ValueGuidedBanditAgent(config)
    metrics = agent.update(batch, update_actor=True)

    assert metrics["actor_success_bc_samples"] == float(batch_size)
    assert metrics["actor_hypothesis_active_heads"] == 3.0
    assert metrics["actor_hypothesis_loss"] == pytest.approx(np.log(3.0), rel=1e-5)
    state, latent, ref = _inputs(config)
    residual = agent.propose_actor_residual(state, latent, ref, step_id=0)
    assert residual.shape == (config.chunk_len, config.action_dim)
    assert np.all(np.isfinite(residual))

    path = tmp_path / "multi-hypothesis-agent.pt"
    agent.save(path, snapshot_id="multi-hypothesis")
    restored = ValueGuidedBanditAgent.load(path)
    assert restored.config.actor_hypotheses == 3
    np.testing.assert_allclose(
        restored.propose_actor_residual(state, latent, ref, step_id=0), residual
    )
