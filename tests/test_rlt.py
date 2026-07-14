from __future__ import annotations

import numpy as np
import pytest
import torch

from maniskill_myws.rlt.networks import ChunkActor
from maniskill_myws.rlt.replay import ChunkReplayBatch, validate_pd_joint_pos_action_dim
from maniskill_myws.rlt.trainer import (
    ManiSkillRLTAgent,
    RLTTrainConfig,
    pd_joint_pos_regularizers,
)


def _batch(
    *,
    batch_size: int = 6,
    state_dim: int = 25,
    action_dim: int = 8,
    chunk_len: int = 4,
    image_shape: tuple[int, ...] | None = None,
) -> ChunkReplayBatch:
    rng = np.random.default_rng(0)
    images = None
    next_images = None
    if image_shape is not None:
        images = rng.integers(0, 255, size=(batch_size, *image_shape), dtype=np.uint8)
        next_images = rng.integers(0, 255, size=(batch_size, *image_shape), dtype=np.uint8)
    ref = rng.normal(0.0, 0.1, size=(batch_size, chunk_len, action_dim)).astype(np.float32)
    actions = ref + rng.normal(0.0, 0.01, size=ref.shape).astype(np.float32)
    return ChunkReplayBatch(
        obs=rng.normal(size=(batch_size, state_dim)).astype(np.float32),
        ref_chunks=ref,
        action_chunks=actions,
        rewards=rng.normal(size=(batch_size, chunk_len)).astype(np.float32),
        dones=np.zeros((batch_size,), dtype=np.float32),
        next_obs=rng.normal(size=(batch_size, state_dim)).astype(np.float32),
        next_ref_chunks=ref.copy(),
        source_chunks=np.zeros((batch_size, chunk_len), dtype=np.uint8),
        images=images,
        next_images=next_images,
    )


def test_pd_joint_pos_action_dim_is_enforced() -> None:
    validate_pd_joint_pos_action_dim(8)
    with pytest.raises(ValueError):
        validate_pd_joint_pos_action_dim(7)


def test_chunk_actor_outputs_pd_joint_pos_chunk_shape() -> None:
    actor = ChunkActor(state_dim=25, action_dim=8, chunk_len=4, hidden_dim=32)
    state = torch.zeros((3, 25), dtype=torch.float32)
    ref = torch.zeros((3, 4, 8), dtype=torch.float32)
    action = actor.sample(state, ref, deterministic=True)
    assert action.shape == (3, 4, 8)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_pd_joint_pos_regularizer_separates_arm_and_gripper() -> None:
    pred = torch.zeros((2, 4, 8), dtype=torch.float32)
    target = torch.zeros_like(pred)
    ref = torch.zeros_like(pred)
    pred[:, :, 7] = torch.arange(4, dtype=torch.float32)
    correction, smoothness, gripper_smoothness, metrics = pd_joint_pos_regularizers(
        pred,
        target,
        ref,
        gripper_smoothness_weight=0.1,
    )
    assert torch.isclose(correction, torch.tensor(0.0))
    assert torch.isclose(metrics["arm_smoothness_penalty"], torch.tensor(0.0))
    assert gripper_smoothness > 0
    assert torch.isclose(smoothness, gripper_smoothness * 0.1)


def test_maniskill_rlt_update_runs() -> None:
    cfg = RLTTrainConfig(
        state_dim=25,
        action_dim=8,
        chunk_len=4,
        hidden_dim=32,
        num_layers=1,
        actor_update_period=2,
    )
    agent = ManiSkillRLTAgent(cfg)
    batch = _batch(chunk_len=cfg.chunk_len)
    metrics1 = agent.update(batch)
    metrics2 = agent.update(batch)
    assert metrics1["did_actor_update"] == 0.0
    assert metrics2["did_actor_update"] == 1.0
    assert agent.total_updates == 2
    assert agent.actor_updates == 1


def test_visual_rlt_update_runs() -> None:
    image_shape = (2, 32, 32, 3)
    cfg = RLTTrainConfig(
        state_dim=25,
        action_dim=8,
        chunk_len=3,
        hidden_dim=32,
        num_layers=1,
        actor_update_period=1,
        visual_encoder="resnet10",
        image_shape=image_shape,
        visual_latent_dim=16,
    )
    agent = ManiSkillRLTAgent(cfg)
    metrics = agent.update(_batch(chunk_len=cfg.chunk_len, image_shape=image_shape))
    assert metrics["did_actor_update"] == 1.0
