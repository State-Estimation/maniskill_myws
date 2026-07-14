from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .networks import ChunkActor, TwinCritic
from .replay import ChunkReplayBatch, TransitionSource, validate_pd_joint_pos_action_dim


@dataclass
class RLTTrainConfig:
    state_dim: int
    action_dim: int = 8
    chunk_len: int = 10
    hidden_dim: int = 256
    num_layers: int = 2

    gamma: float = 0.99
    fixed_std: float = 0.01
    reference_dropout_prob: float = 0.5
    actor_output_mode: Literal["delta", "absolute"] = "delta"
    action_delta_scale: float = 0.1

    bc_weight: float = 5.0
    q_weight: float = 0.1
    correction_weight: float = 1.0
    smoothness_weight: float = 1.0
    gripper_smoothness_weight: float = 0.1
    smoothness_target: Literal["bc_target", "zero"] = "bc_target"

    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    target_tau: float = 0.005
    actor_update_period: int = 2
    grad_clip_norm: float = 1.0

    visual_encoder: Literal["none", "resnet10"] = "none"
    image_shape: tuple[int, ...] | None = None
    visual_latent_dim: int = 256
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        validate_pd_joint_pos_action_dim(self.action_dim)
        if self.chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        if self.reference_dropout_prob < 0.0 or self.reference_dropout_prob > 1.0:
            raise ValueError("reference_dropout_prob must be in [0, 1]")
        if self.image_shape is None and self.visual_encoder != "none":
            raise ValueError("visual_encoder requires image_shape")


def _discounted_chunk_rewards(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    discounts = torch.pow(
        torch.as_tensor(float(gamma), dtype=rewards.dtype, device=rewards.device),
        torch.arange(rewards.shape[-1], dtype=rewards.dtype, device=rewards.device),
    )
    return (rewards * discounts.view(1, -1)).sum(dim=-1)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _reference_dropout(
    ref_chunks: torch.Tensor,
    dropout_prob: float,
) -> torch.Tensor:
    if dropout_prob <= 0.0:
        return ref_chunks
    keep = torch.rand((ref_chunks.shape[0], 1, 1), dtype=ref_chunks.dtype, device=ref_chunks.device)
    keep = (keep >= float(dropout_prob)).to(ref_chunks.dtype)
    return ref_chunks * keep


def pd_joint_pos_regularizers(
    pred_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    ref_chunk: torch.Tensor,
    *,
    gripper_smoothness_weight: float = 0.1,
    smoothness_target: Literal["bc_target", "zero"] = "bc_target",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Regularizers for ManiSkill Panda pd_joint_pos chunks.

    Action layout is [7 arm joint position targets, 1 gripper target]. The
    correction penalty limits arm-joint edits from the OpenPI reference chunk.
    The smoothness penalty compares step-to-step joint target deltas; gripper
    smoothness is tracked separately and down-weighted by default because it is
    often intentionally discontinuous.
    """

    arm = slice(0, 7)
    gripper = slice(7, 8)
    correction = torch.mean(torch.square(pred_chunk[..., arm] - ref_chunk[..., arm]))

    if pred_chunk.shape[1] <= 1:
        zero = torch.zeros((), dtype=pred_chunk.dtype, device=pred_chunk.device)
        metrics = {
            "arm_smoothness_penalty": zero,
            "gripper_smoothness_penalty": zero,
            "correction_penalty": correction,
        }
        return correction, zero, zero, metrics

    pred_arm_delta = pred_chunk[:, 1:, arm] - pred_chunk[:, :-1, arm]
    pred_gripper_delta = pred_chunk[:, 1:, gripper] - pred_chunk[:, :-1, gripper]
    if smoothness_target == "bc_target":
        target_arm_delta = target_chunk[:, 1:, arm] - target_chunk[:, :-1, arm]
        target_gripper_delta = target_chunk[:, 1:, gripper] - target_chunk[:, :-1, gripper]
    elif smoothness_target == "zero":
        target_arm_delta = torch.zeros_like(pred_arm_delta)
        target_gripper_delta = torch.zeros_like(pred_gripper_delta)
    else:
        raise ValueError(f"Unsupported smoothness_target: {smoothness_target}")

    arm_smoothness = torch.mean(torch.square(pred_arm_delta - target_arm_delta))
    gripper_smoothness = torch.mean(torch.square(pred_gripper_delta - target_gripper_delta))
    smoothness = arm_smoothness + float(gripper_smoothness_weight) * gripper_smoothness
    metrics = {
        "arm_smoothness_penalty": arm_smoothness,
        "gripper_smoothness_penalty": gripper_smoothness,
        "correction_penalty": correction,
    }
    return correction, smoothness, gripper_smoothness, metrics


class ManiSkillRLTAgent:
    """PyTorch RLT-style chunk learner for ManiSkill pd_joint_pos tasks."""

    def __init__(self, config: RLTTrainConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        common_kwargs = dict(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            chunk_len=config.chunk_len,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            image_shape=config.image_shape,
            visual_encoder=config.visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        )
        self.actor = ChunkActor(
            **common_kwargs,
            fixed_std=config.fixed_std,
            action_delta_scale=config.action_delta_scale,
            output_mode=config.actor_output_mode,
            action_low=config.action_low,
            action_high=config.action_high,
        ).to(self.device)
        self.target_actor = ChunkActor(
            **common_kwargs,
            fixed_std=config.fixed_std,
            action_delta_scale=config.action_delta_scale,
            output_mode=config.actor_output_mode,
            action_low=config.action_low,
            action_high=config.action_high,
        ).to(self.device)
        self.critic = TwinCritic(**common_kwargs).to(self.device)
        self.target_critic = TwinCritic(**common_kwargs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.total_updates = 0
        self.actor_updates = 0

    def _images_to_tensor(self, images: np.ndarray | None) -> torch.Tensor | None:
        if images is None:
            return None
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device) / 255.0
        if tensor.ndim != 5:
            raise ValueError(f"Expected images with shape (B,V,H,W,C), got {tuple(tensor.shape)}")
        return tensor.permute(0, 1, 4, 2, 3).reshape(
            tensor.shape[0],
            tensor.shape[1] * tensor.shape[4],
            tensor.shape[2],
            tensor.shape[3],
        )

    def _to_tensor_batch(self, batch: ChunkReplayBatch) -> dict[str, torch.Tensor | None]:
        return {
            "obs": torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device),
            "ref_chunks": torch.as_tensor(
                batch.ref_chunks, dtype=torch.float32, device=self.device
            ),
            "action_chunks": torch.as_tensor(
                batch.action_chunks, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device),
            "next_ref_chunks": torch.as_tensor(
                batch.next_ref_chunks, dtype=torch.float32, device=self.device
            ),
            "source_chunks": torch.as_tensor(
                batch.source_chunks, dtype=torch.long, device=self.device
            ),
            "images": self._images_to_tensor(batch.images),
            "next_images": self._images_to_tensor(batch.next_images),
        }

    def _bc_target(self, b: dict[str, torch.Tensor | None]) -> torch.Tensor:
        source_chunks = b["source_chunks"]
        assert isinstance(source_chunks, torch.Tensor)
        action_chunks = b["action_chunks"]
        ref_chunks = b["ref_chunks"]
        assert isinstance(action_chunks, torch.Tensor)
        assert isinstance(ref_chunks, torch.Tensor)
        human_mask = (source_chunks == int(TransitionSource.HUMAN)) | (
            source_chunks == int(TransitionSource.MIXED)
        )
        return torch.where(human_mask[..., None], action_chunks, ref_chunks)

    @torch.no_grad()
    def select_chunk(
        self,
        obs: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        images: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        self.actor.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
            1, self.config.state_dim
        )
        ref_t = torch.as_tensor(ref_chunk, dtype=torch.float32, device=self.device).reshape(
            1, self.config.chunk_len, self.config.action_dim
        )
        images_t = self._images_to_tensor(images[None] if images is not None else None)
        action = self.actor.sample(obs_t, ref_t, images_t, deterministic=deterministic)
        self.actor.train()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: ChunkReplayBatch) -> dict[str, float]:
        b = self._to_tensor_batch(batch)
        obs = b["obs"]
        ref_chunks = b["ref_chunks"]
        action_chunks = b["action_chunks"]
        rewards = b["rewards"]
        dones = b["dones"]
        next_obs = b["next_obs"]
        next_ref_chunks = b["next_ref_chunks"]
        images = b["images"]
        next_images = b["next_images"]
        assert isinstance(obs, torch.Tensor)
        assert isinstance(ref_chunks, torch.Tensor)
        assert isinstance(action_chunks, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert isinstance(next_obs, torch.Tensor)
        assert isinstance(next_ref_chunks, torch.Tensor)

        with torch.no_grad():
            next_action = self.target_actor.sample(
                next_obs, next_ref_chunks, next_images, deterministic=False
            )
            next_q1, next_q2 = self.target_critic.q_values(next_obs, next_action, next_images)
            bootstrap = (1.0 - dones) * (self.config.gamma ** self.config.chunk_len) * torch.min(
                next_q1, next_q2
            )
            target_q = _discounted_chunk_rewards(rewards, self.config.gamma) + bootstrap

        q1, q2 = self.critic.q_values(obs, action_chunks, images)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_opt.step()

        metrics: dict[str, float] = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "q1": float(q1.mean().detach().cpu()),
            "q2": float(q2.mean().detach().cpu()),
            "target_q": float(target_q.mean().detach().cpu()),
        }

        self.total_updates += 1
        if self.total_updates % max(1, self.config.actor_update_period) == 0:
            dropped_ref = _reference_dropout(ref_chunks, self.config.reference_dropout_prob)
            pred_chunk = self.actor.sample(obs, dropped_ref, images, deterministic=False)
            actor_q1, _ = self.critic.q_values(obs, pred_chunk, images)
            bc_target = self._bc_target(b)
            bc_penalty = torch.mean(torch.square(pred_chunk - bc_target))
            correction, smoothness, _, reg_metrics = pd_joint_pos_regularizers(
                pred_chunk,
                bc_target,
                ref_chunks,
                gripper_smoothness_weight=self.config.gripper_smoothness_weight,
                smoothness_target=self.config.smoothness_target,
            )
            actor_q = actor_q1.mean()
            actor_loss = (
                self.config.bc_weight * bc_penalty
                - self.config.q_weight * actor_q
                + self.config.correction_weight * correction
                + self.config.smoothness_weight * smoothness
            )

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()
            _soft_update(self.target_actor, self.actor, self.config.target_tau)
            _soft_update(self.target_critic, self.critic, self.config.target_tau)
            self.actor_updates += 1

            metrics.update(
                actor_loss=float(actor_loss.detach().cpu()),
                actor_q=float(actor_q.detach().cpu()),
                bc_penalty=float(bc_penalty.detach().cpu()),
                correction_penalty=float(correction.detach().cpu()),
                smoothness_penalty=float(smoothness.detach().cpu()),
                did_actor_update=1.0,
            )
            metrics.update({k: float(v.detach().cpu()) for k, v in reg_metrics.items()})
        else:
            metrics["did_actor_update"] = 0.0

        metrics["total_updates"] = float(self.total_updates)
        metrics["actor_updates"] = float(self.actor_updates)
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "total_updates": self.total_updates,
                "actor_updates": self.actor_updates,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> ManiSkillRLTAgent:
        payload = torch.load(path, map_location=device)
        agent = cls(RLTTrainConfig(**payload["config"]), device=device)
        agent.actor.load_state_dict(payload["actor"])
        agent.target_actor.load_state_dict(payload["target_actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        agent.actor_opt.load_state_dict(payload["actor_opt"])
        agent.critic_opt.load_state_dict(payload["critic_opt"])
        agent.total_updates = int(payload.get("total_updates", 0))
        agent.actor_updates = int(payload.get("actor_updates", 0))
        return agent
