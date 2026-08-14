from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from maniskill_myws.rlt.frozen_latent_rl import (
    FROZEN_LATENT_CHECKPOINT_SCHEMA,
    FrozenLatentBatch,
    FrozenLatentReplayBuffer,
    FrozenLatentResidualAgent,
    FrozenLatentRLConfig,
    make_runtime_identity,
    runtime_identity_sha256,
)


def _config(**overrides) -> FrozenLatentRLConfig:
    values = {
        "state_dim": 5,
        "latent_dim": 16,
        "action_dim": 8,
        "chunk_len": 4,
        "max_episode_steps": 20,
        "context_dim": 16,
        "hidden_dim": 32,
        "num_layers": 1,
        "num_critics": 2,
        "fixed_std": 0.0,
        "target_noise": 0.0,
        "actor_update_period": 1,
        "exploration_knots": 2,
        "action_low": (-1.0,) * 8,
        "action_high": (1.0,) * 8,
    }
    values.update(overrides)
    return FrozenLatentRLConfig(**values)


def _latent_shape(config: FrozenLatentRLConfig) -> tuple[int, ...]:
    if config.latent_storage_rows == 1:
        return (config.latent_dim,)
    return (config.latent_storage_rows, config.latent_dim)


def _batch(config: FrozenLatentRLConfig, size: int = 8) -> FrozenLatentBatch:
    states = np.zeros((size, config.state_dim), dtype=np.float32)
    latents = np.zeros((size, *_latent_shape(config)), dtype=np.float32)
    refs = np.zeros((size, config.chunk_len, config.action_dim), dtype=np.float32)
    actions = refs.copy()
    actions[:, 1:, 0] = -0.08
    actions[:, 1:, 3] = 0.05
    rewards = np.zeros((size, config.chunk_len), dtype=np.float32)
    rewards[:, -1] = 1.0
    return FrozenLatentBatch(
        states=states,
        latents=latents,
        ref_chunks=refs,
        action_chunks=actions,
        rewards=rewards,
        dones=np.ones((size,), dtype=np.float32),
        next_states=states.copy(),
        next_latents=latents.copy(),
        next_ref_chunks=refs.copy(),
        durations=np.full((size,), config.chunk_len, dtype=np.int32),
        step_ids=np.zeros((size,), dtype=np.int32),
        mc_returns=np.ones((size,), dtype=np.float32),
    )


def _buffer(config: FrozenLatentRLConfig, capacity: int = 8, seed: int = 7):
    return FrozenLatentReplayBuffer(
        capacity,
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        latent_bins=config.latent_storage_rows,
        chunk_len=config.chunk_len,
        action_dim=config.action_dim,
        seed=seed,
    )


def _add(
    replay: FrozenLatentReplayBuffer,
    config: FrozenLatentRLConfig,
    *,
    residual: float,
    mc_return: float,
    step_id: int,
) -> None:
    latent = np.full(_latent_shape(config), step_id / 10.0, dtype=np.float32)
    ref = np.zeros((config.chunk_len, config.action_dim), dtype=np.float32)
    action = ref.copy()
    action[:, 0] = residual
    rewards = np.zeros(config.chunk_len, dtype=np.float32)
    rewards[-1] = float(mc_return > 0.5)
    state = np.full(config.state_dim, step_id, dtype=np.float32)
    replay.add(
        state=state,
        latent=latent,
        ref_chunk=ref,
        action_chunk=action,
        rewards=rewards,
        done=True,
        next_state=state + 1,
        next_latent=latent + 0.1,
        next_ref_chunk=ref,
        duration=config.chunk_len,
        step_id=step_id,
        mc_return=mc_return,
    )


def _identity(*, temporal_latent_bins: int = 1) -> dict:
    return make_runtime_identity(
        env_id="Test-v0",
        obs_mode="rgb",
        reward_mode="sparse",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        render_backend="sapien_cuda:0",
        enhanced_determinism=False,
        prompt="test",
        image_key="sensor_data/base_camera/rgb",
        wrist_image_key="sensor_data/hand_camera/rgb",
        state_keys=["agent/qpos"],
        resize=224,
        chunk_len=4,
        max_episode_steps=20,
        openpi_policy_identity_sha256="a" * 64,
        temporal_latent_bins=temporal_latent_bins,
    )


def test_config_supports_only_mean_or_five_temporal_bins() -> None:
    assert _config().latent_storage_rows == 1
    assert _config(temporal_latent_bins=5).latent_storage_rows == 6
    with pytest.raises(ValueError, match="temporal_latent_bins"):
        _config(temporal_latent_bins=3)
    with pytest.raises(ValueError, match="action_dim=8"):
        _config(action_dim=7, action_low=(-1.0,) * 7, action_high=(1.0,) * 7)


def test_zero_initialized_actor_is_exact_base_policy() -> None:
    config = _config()
    agent = FrozenLatentResidualAgent(config)
    rng = np.random.default_rng(4)
    ref = rng.uniform(
        -0.5, 0.5, (config.chunk_len, config.action_dim)
    ).astype(np.float32)
    action, residual = agent.select_chunk(
        np.zeros(config.state_dim, dtype=np.float32),
        np.zeros(config.latent_dim, dtype=np.float32),
        ref,
        step_id=0,
        deterministic=True,
    )
    np.testing.assert_array_equal(residual, np.zeros_like(residual))
    np.testing.assert_array_equal(action, ref)


def test_actor_expands_smooth_knots_across_full_chunk() -> None:
    config = _config()
    agent = FrozenLatentResidualAgent(config)
    output = agent.actor.trunk[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.bias[0] = -0.4
        output.bias[config.action_dim + 3] = 0.25
    residual = agent.select_residual(
        np.zeros(config.state_dim, dtype=np.float32),
        np.zeros(config.latent_dim, dtype=np.float32),
        np.zeros((config.chunk_len, config.action_dim), dtype=np.float32),
        step_id=0,
        deterministic=True,
    )
    assert residual[0, 0] < 0.0
    assert residual[-1, 3] > 0.0
    assert np.max(np.abs(residual)) <= config.actor_residual_limit


@pytest.mark.parametrize("temporal_bins", [1, 5])
def test_replay_exact_round_trip_preserves_executed_actions(
    tmp_path, temporal_bins: int
) -> None:
    config = _config(temporal_latent_bins=temporal_bins)
    replay = _buffer(config, capacity=4)
    _add(replay, config, residual=0.07, mc_return=1.0, step_id=0)
    _add(replay, config, residual=-0.03, mc_return=0.0, step_id=4)
    path = tmp_path / "replay.npz"
    replay.save(path, snapshot_id="snapshot")
    expected_sample = replay.sample(5)

    restored = _buffer(config, capacity=4, seed=999)
    assert restored.load(path) == 2
    assert restored.last_load_was_exact
    assert restored.last_loaded_snapshot_id == "snapshot"
    actual_sample = restored.sample(5)
    for field in FrozenLatentBatch.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(actual_sample, field), getattr(expected_sample, field)
        )


def test_compact_v2_replay_migrates_without_claiming_exact_resume(tmp_path) -> None:
    config = _config()
    source = _buffer(config, capacity=4)
    _add(source, config, residual=0.07, mc_return=1.0, step_id=0)
    _add(source, config, residual=-0.03, mc_return=0.0, step_id=4)
    source_batch = source.batch(np.arange(len(source), dtype=np.int64))
    path = tmp_path / "legacy_v2.npz"
    np.savez_compressed(
        path,
        schema_version=np.int32(2),
        state_dim=np.int32(config.state_dim),
        latent_dim=np.int32(config.latent_dim),
        chunk_len=np.int32(config.chunk_len),
        action_dim=np.int32(config.action_dim),
        states=source_batch.states,
        latents=source.latents[: len(source)],
        ref_chunks=source_batch.ref_chunks,
        action_chunks=source_batch.action_chunks,
        rewards=source_batch.rewards,
        dones=source_batch.dones,
        next_states=source_batch.next_states,
        next_latents=source.next_latents[: len(source)],
        next_ref_chunks=source_batch.next_ref_chunks,
        durations=source_batch.durations,
        step_ids=source_batch.step_ids,
        mc_returns=source_batch.mc_returns,
    )

    restored = _buffer(config, capacity=8, seed=999)
    expected_rng_state = restored._rng.bit_generator.state
    assert restored.load(path) == 2
    assert not restored.last_load_was_exact
    assert restored.last_loaded_snapshot_id is None
    assert restored.pos == 2
    assert not restored.full
    assert restored._rng.bit_generator.state == expected_rng_state
    assert restored.last_migration_stats == {
        "source_schema": 2,
        "source_latent_bins": 1,
        "target_latent_bins": 1,
        "raw_rows": 2,
        "migrated": False,
        "layout_migrated": True,
    }
    restored_batch = restored.batch(np.arange(2, dtype=np.int64))
    for field in FrozenLatentBatch.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(restored_batch, field), getattr(source_batch, field)
        )

    too_small = _buffer(config, capacity=1)
    with pytest.raises(ValueError, match="does not fit"):
        too_small.load(path)

    nonempty = _buffer(config, capacity=8)
    _add(nonempty, config, residual=0.0, mc_return=0.0, step_id=0)
    with pytest.raises(ValueError, match="requires an empty buffer"):
        nonempty.load(path)


def test_replay_stratification_balances_success_and_failure() -> None:
    config = _config()
    replay = _buffer(config)
    _add(replay, config, residual=0.0, mc_return=0.0, step_id=0)
    _add(replay, config, residual=0.08, mc_return=1.0, step_id=4)
    _add(replay, config, residual=-0.08, mc_return=0.0, step_id=8)
    assert replay.has_stratified_support(
        8,
        nonzero_fraction=0.5,
        nonzero_success_fraction=0.5,
        success_threshold=0.5,
    )
    batch = replay.sample_stratified(
        8,
        nonzero_fraction=0.5,
        nonzero_success_fraction=0.5,
        success_threshold=0.5,
    )
    nonzero = np.any(
        np.abs(batch.action_chunks - batch.ref_chunks) > 1e-6, axis=(1, 2)
    )
    successful = batch.mc_returns > 0.5
    assert np.count_nonzero(~nonzero) == 4
    assert np.count_nonzero(nonzero & successful) == 2
    assert np.count_nonzero(nonzero & ~successful) == 2


def test_mean_replay_migrates_to_zero_information_temporal_rows(tmp_path) -> None:
    mean_config = _config()
    source = _buffer(mean_config, capacity=4)
    _add(source, mean_config, residual=0.04, mc_return=1.0, step_id=0)
    path = tmp_path / "mean.npz"
    source.save(path)

    temporal_config = _config(temporal_latent_bins=5)
    target = _buffer(temporal_config, capacity=4)
    assert target.load(path) == 1
    assert not target.last_load_was_exact
    assert target.last_migration_stats == {
        "source_schema": 3,
        "source_latent_bins": 1,
        "target_latent_bins": 6,
        "raw_rows": 1,
        "migrated": True,
    }
    np.testing.assert_array_equal(
        target.latents[0], np.repeat(target.latents[0, :1], 6, axis=0)
    )


def test_temporal_upgrade_is_exact_at_zero_gate(tmp_path) -> None:
    torch.manual_seed(3)
    source = FrozenLatentResidualAgent(
        _config(), runtime_identity=_identity()
    )
    with torch.no_grad():
        output = source.actor.trunk[-1]
        assert isinstance(output, torch.nn.Linear)
        output.bias[0] = 0.2
        source.sync_target_actor()
    path = tmp_path / "mean.pt"
    source.save(path)
    temporal = FrozenLatentResidualAgent.upgrade_from_mean_checkpoint(
        path,
        runtime_identity=_identity(temporal_latent_bins=5),
        temporal_latent_bins=5,
        temporal_adapter_dim=8,
    )

    rng = np.random.default_rng(8)
    state = rng.normal(size=source.config.state_dim).astype(np.float32)
    mean = rng.normal(size=source.config.latent_dim).astype(np.float32)
    ref = rng.uniform(
        -0.5, 0.5, (source.config.chunk_len, source.config.action_dim)
    ).astype(np.float32)
    repeated = np.repeat(mean[None], 6, axis=0)
    source_residual = source.select_residual(
        state, mean, ref, step_id=4, deterministic=True
    )
    temporal_residual = temporal.select_residual(
        state, repeated, ref, step_id=4, deterministic=True
    )
    np.testing.assert_array_equal(temporal_residual, source_residual)
    assert float(temporal.context.temporal_gate.detach()) == 0.0


def test_temporal_gate_can_learn_ordered_token_signal() -> None:
    config = _config(temporal_latent_bins=5, temporal_adapter_dim=8)
    agent = FrozenLatentResidualAgent(config)
    batch = _batch(config)
    batch.latents[:, 1:] = np.linspace(
        -1.0, 1.0, 5, dtype=np.float32
    )[None, :, None]
    metrics = agent.update(batch, update_actor=False)
    assert np.isfinite(metrics["critic_loss"])
    assert agent.context.temporal_gate.grad is not None
    assert torch.isfinite(agent.context.temporal_gate.grad)


def test_critic_warmup_does_not_update_actor_then_td3_does() -> None:
    config = _config()
    agent = FrozenLatentResidualAgent(config)
    batch = _batch(config)
    actor_before = {
        name: value.detach().clone() for name, value in agent.actor.state_dict().items()
    }
    warmup = agent.update(batch, update_actor=False)
    assert warmup["actor_updated"] == 0.0
    for name, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)

    trained = agent.update(batch, update_actor=True)
    assert trained["actor_updated"] == 1.0
    assert trained["actor_success_bc_samples"] == len(batch.states)
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in agent.actor.state_dict().items()
    )


def test_zero_success_bc_weight_uses_legacy_frozen_actor_objective() -> None:
    config = _config(actor_success_bc_weight=0.0)
    agent = FrozenLatentResidualAgent(config)
    metrics = agent.update(_batch(config), update_actor=True)
    assert metrics["actor_updated"] == 1.0
    assert metrics["actor_success_bc_loss"] == 0.0
    assert metrics["actor_success_bc_samples"] == 0.0


def test_success_bc_can_filter_non_advantageous_residuals() -> None:
    config = _config(actor_success_bc_min_q_advantage=1e6)
    agent = FrozenLatentResidualAgent(config)
    metrics = agent.update(_batch(config), update_actor=True)
    assert metrics["actor_updated"] == 1.0
    assert metrics["actor_success_bc_samples"] == 0.0
    assert metrics["actor_success_bc_loss"] == 0.0
    assert metrics["actor_success_bc_advantage"] == 0.0


def test_action_clipping_defines_effective_residual() -> None:
    config = _config()
    agent = FrozenLatentResidualAgent(config)
    ref = np.full((config.chunk_len, config.action_dim), 0.99, dtype=np.float32)
    requested = np.ones_like(ref)
    action = agent.apply_residual(ref, requested)
    assert np.all(action <= 1.0)
    recovered = agent.normalized_residual(ref, action)
    assert np.all(recovered < 1.0)
    np.testing.assert_allclose(agent.apply_residual(ref, recovered), action, atol=1e-7)


def test_dueling_critic_gives_exact_zero_base_advantage() -> None:
    config = _config()
    agent = FrozenLatentResidualAgent(config)
    advantage = agent.conservative_advantage(
        np.zeros(config.state_dim, dtype=np.float32),
        np.zeros(config.latent_dim, dtype=np.float32),
        np.zeros((config.chunk_len, config.action_dim), dtype=np.float32),
        np.zeros((config.chunk_len, config.action_dim), dtype=np.float32),
        step_id=0,
    )
    assert advantage == 0.0


def test_checkpoint_round_trip_binds_runtime_identity(tmp_path) -> None:
    identity = _identity()
    agent = FrozenLatentResidualAgent(_config(), runtime_identity=identity)
    agent.update(_batch(agent.config), update_actor=True)
    path = tmp_path / "agent.pt"
    agent.save(path, snapshot_id="generation")

    restored = FrozenLatentResidualAgent.load(path)
    assert restored.runtime_identity == identity
    assert restored.snapshot_id == "generation"
    restored.assert_runtime_identity(identity)
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        restored.assert_runtime_identity({**identity, "resize": 96})
    assert len(runtime_identity_sha256(identity)) == 64


def test_legacy_v3_checkpoint_restores_pre_causal_actor_objective(tmp_path) -> None:
    agent = FrozenLatentResidualAgent(_config(), runtime_identity=_identity())
    path = tmp_path / "legacy_v3.pt"
    agent.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema"] = "lightweight_rlt_frozen_pi0_continuous_residual_v3"
    payload["checkpoint_version"] = 3
    payload["config"].pop("actor_success_bc_weight")
    payload["config"].pop("actor_success_bc_min_residual_rms")
    payload["config"].pop("outcome_success_threshold")
    torch.save(payload, path)

    restored = FrozenLatentResidualAgent.load(path)
    assert restored.config.actor_success_bc_weight == 0.0
    causal_continuation = FrozenLatentResidualAgent.load(
        path, legacy_actor_success_bc_weight=2.0
    )
    assert causal_continuation.config.actor_success_bc_weight == 2.0


def test_checkpoint_rejects_removed_policy_branch(tmp_path) -> None:
    agent = FrozenLatentResidualAgent(_config(), runtime_identity=_identity())
    path = tmp_path / "agent.pt"
    agent.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == FROZEN_LATENT_CHECKPOINT_SCHEMA
    payload["config"] = {**asdict(agent.config), "latent_mode_count": 3}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="removed policy branch"):
        FrozenLatentResidualAgent.load(path)


def test_replay_rejects_float16_latent_overflow() -> None:
    config = _config()
    replay = _buffer(config)
    latent = np.full(config.latent_dim, 70_000.0, dtype=np.float32)
    ref = np.zeros((config.chunk_len, config.action_dim), dtype=np.float32)
    with pytest.raises(ValueError, match="overflows"):
        replay.add(
            state=np.zeros(config.state_dim, dtype=np.float32),
            latent=latent,
            ref_chunk=ref,
            action_chunk=ref,
            rewards=np.zeros(config.chunk_len, dtype=np.float32),
            done=False,
            next_state=np.zeros(config.state_dim, dtype=np.float32),
            next_latent=latent,
            next_ref_chunk=ref,
            duration=config.chunk_len,
            step_id=0,
        )
