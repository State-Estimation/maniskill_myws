from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from .replay_buffer import ReplayBatch


@dataclass
class SACConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    action_scale: float = 0.1
    target_entropy: float | None = None
    alpha_min: float = 1e-4
    alpha_max: float = 10.0
    actor_update_interval: int = 2
    grad_clip_norm: float = 1.0
    cql_alpha: float = 0.0
    cql_random_actions: int = 10
    calql_alpha: float = 5.0
    calql_n_actions: int = 10
    calql_temp: float = 1.0
    calql_importance_sample: bool = True
    calql_max_target_backup: bool = True
    calql_backup_entropy: bool = False
    otf_backup_actions: int = 0
    otf_include_base_action: bool = True
    otf_backup_entropy: bool = False
    visual_encoder: str = "none"
    image_shape: tuple[int, ...] | None = None
    visual_latent_dim: int = 256
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResNetV1Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_num_groups(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetV1_10Encoder(nn.Module):
    """Compact ResNetV1-10 style visual encoder used by the residual RL policy."""

    def __init__(self, image_shape: tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        if len(image_shape) != 4:
            raise ValueError(f"Expected image_shape=(views,H,W,C), got {image_shape}")
        views, _, _, channels = image_shape
        in_channels = int(views) * int(channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_num_groups(32), 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            ResNetV1Block(32, 32, stride=1),
            ResNetV1Block(32, 64, stride=2),
            ResNetV1Block(64, 128, stride=2),
            ResNetV1Block(128, 256, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(images)))


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        *,
        image_shape: tuple[int, ...] | None,
        visual_encoder: str,
        visual_latent_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_shape = image_shape
        self.visual_encoder_name = visual_encoder
        if visual_encoder == "none" or image_shape is None:
            self.visual: nn.Module | None = None
            self.out_dim = self.state_dim
        elif visual_encoder == "resnet10":
            self.visual = ResNetV1_10Encoder(image_shape, int(visual_latent_dim))
            self.out_dim = self.state_dim + int(visual_latent_dim)
        else:
            raise ValueError(f"Unsupported visual_encoder: {visual_encoder}")

    def forward(self, obs: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        if self.visual is None:
            return obs
        if images is None:
            raise ValueError("Visual observation encoder requires images")
        visual_features = self.visual(images)
        return torch.cat([obs, visual_features], dim=-1)


class GaussianResidualActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_scale: float,
        *,
        image_shape: tuple[int, ...] | None,
        visual_encoder: str,
        visual_latent_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_encoder = ObservationEncoder(
            state_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=visual_latent_dim,
        )
        self.body = MLP(self.obs_encoder.out_dim + action_dim, 2 * action_dim, hidden_dim)
        self.register_buffer("scale", torch.full((action_dim,), float(action_scale)))

    def forward(
        self,
        obs: torch.Tensor,
        base_action: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.obs_encoder(obs, images)
        out = self.body(torch.cat([features, base_action], dim=-1))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        base_action: torch.Tensor,
        images: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs, base_action, images)
        if deterministic:
            raw = mean
        else:
            raw = Normal(mean, log_std.exp()).rsample()
        squashed = torch.tanh(raw)
        delta = squashed * self.scale
        if deterministic:
            log_prob = torch.zeros((obs.shape[0], 1), device=obs.device)
        else:
            normal = Normal(mean, log_std.exp())
            correction = torch.log(self.scale * (1.0 - squashed.pow(2)) + 1e-6)
            log_prob = (normal.log_prob(raw) - correction).sum(dim=-1, keepdim=True)
        return delta, log_prob


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        image_shape: tuple[int, ...] | None,
        visual_encoder: str,
        visual_latent_dim: int,
    ) -> None:
        super().__init__()
        self.obs_encoder = ObservationEncoder(
            state_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=visual_latent_dim,
        )
        self.q = MLP(self.obs_encoder.out_dim + action_dim, 1, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, images: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.obs_encoder(obs, images)
        return self.q(torch.cat([features, action], dim=-1))


class ResidualSAC:
    def __init__(self, config: SACConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        image_shape = (
            tuple(int(x) for x in config.image_shape)
            if config.image_shape is not None
            else None
        )
        visual_encoder = "none" if image_shape is None else config.visual_encoder
        self.actor = GaussianResidualActor(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            config.action_scale,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        ).to(self.device)
        self.q1 = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        ).to(self.device)
        self.q2 = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        ).to(self.device)
        self.q1_target = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        ).to(self.device)
        self.q2_target = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=config.visual_latent_dim,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.q_opt = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr
        )
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = self._default_target_entropy()
        low = config.action_low if config.action_low is not None else (-1.0,) * config.action_dim
        high = config.action_high if config.action_high is not None else (1.0,) * config.action_dim
        self.action_low = torch.as_tensor(low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(high, dtype=torch.float32, device=self.device)
        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _default_target_entropy(self) -> float:
        if self.config.target_entropy is not None:
            return float(self.config.target_entropy)
        # The residual actor samples delta = action_scale * tanh(raw). Because
        # log_prob includes that scale correction, the usual -|A| SAC target is
        # shifted by log(action_scale) per action dimension.
        scale = max(float(self.config.action_scale), 1e-6)
        return -float(self.config.action_dim) + float(self.config.action_dim) * np.log(scale)

    def _images_to_tensor(self, images: np.ndarray | None) -> torch.Tensor | None:
        if images is None:
            return None
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device) / 255.0
        if tensor.ndim != 5:
            raise ValueError(f"Expected images with shape (B,V,H,W,C), got {tuple(tensor.shape)}")
        return tensor.permute(0, 1, 4, 2, 3).reshape(
            tensor.shape[0], tensor.shape[1] * tensor.shape[4], tensor.shape[2], tensor.shape[3]
        )

    def _repeat_images(self, images: torch.Tensor | None, n_actions: int) -> torch.Tensor | None:
        if images is None:
            return None
        batch_size = images.shape[0]
        return images[:, None].expand(batch_size, n_actions, *images.shape[1:]).reshape(
            batch_size * n_actions, *images.shape[1:]
        )

    def _to_tensor_batch(self, batch: ReplayBatch) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device),
            "base_actions": torch.as_tensor(
                batch.base_actions, dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "next_obs": torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device),
            "next_base_actions": torch.as_tensor(
                batch.next_base_actions, dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "mc_returns": torch.as_tensor(
                batch.mc_returns, dtype=torch.float32, device=self.device
            ).unsqueeze(-1),
            "images": self._images_to_tensor(batch.images),
            "next_images": self._images_to_tensor(batch.next_images),
        }

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _sample_policy_actions(
        self,
        obs: torch.Tensor,
        base_actions: torch.Tensor,
        n_actions: int,
        images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        n = max(1, int(n_actions))
        obs_rep = obs[:, None, :].expand(batch_size, n, self.config.state_dim)
        base_rep = base_actions[:, None, :].expand(batch_size, n, self.config.action_dim)
        flat_obs = obs_rep.reshape(batch_size * n, self.config.state_dim)
        flat_base = base_rep.reshape(batch_size * n, self.config.action_dim)
        flat_images = self._repeat_images(images, n)
        delta, logp = self.actor.sample(flat_obs, flat_base, flat_images)
        actions = self._clip_action(flat_base + delta).reshape(
            batch_size, n, self.config.action_dim
        )
        return actions, logp.reshape(batch_size, n)

    def _q_for_action_set(
        self,
        q_net: QNetwork,
        obs: torch.Tensor,
        actions: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_actions, _ = actions.shape
        obs_rep = obs[:, None, :].expand(batch_size, n_actions, self.config.state_dim)
        image_rep = self._repeat_images(images, n_actions)
        return q_net(
            obs_rep.reshape(batch_size * n_actions, self.config.state_dim),
            actions.reshape(batch_size * n_actions, self.config.action_dim),
            image_rep,
        ).reshape(batch_size, n_actions)

    def _otf_candidate_actions(
        self,
        obs: torch.Tensor,
        base_actions: torch.Tensor,
        n_actions: int,
        *,
        images: torch.Tensor | None = None,
        include_base_action: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions, logp = self._sample_policy_actions(obs, base_actions, n_actions, images)
        if not include_base_action:
            return actions, logp
        batch_size = obs.shape[0]
        base_candidate = self._clip_action(base_actions).reshape(
            batch_size, 1, self.config.action_dim
        )
        base_logp = torch.zeros((batch_size, 1), dtype=logp.dtype, device=self.device)
        return torch.cat([base_candidate, actions], dim=1), torch.cat([base_logp, logp], dim=1)

    def _select_otf_action_tensors(
        self,
        obs: torch.Tensor,
        base_actions: torch.Tensor,
        n_actions: int,
        *,
        images: torch.Tensor | None = None,
        include_base_action: bool,
        use_target_critic: bool,
        soft_value: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, logp = self._otf_candidate_actions(
            obs,
            base_actions,
            n_actions,
            images=images,
            include_base_action=include_base_action,
        )
        q1_net = self.q1_target if use_target_critic else self.q1
        q2_net = self.q2_target if use_target_critic else self.q2
        q_values = torch.min(
            self._q_for_action_set(q1_net, obs, actions, images),
            self._q_for_action_set(q2_net, obs, actions, images),
        )
        values = q_values - self.alpha.detach() * logp if soft_value else q_values
        selected_idx = values.argmax(dim=1, keepdim=True)
        gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, self.config.action_dim)
        selected_action = actions.gather(1, gather_idx).squeeze(1)
        selected_logp = logp.gather(1, selected_idx)
        selected_q = q_values.gather(1, selected_idx)
        selected_value = values.gather(1, selected_idx)
        if include_base_action:
            selected_base = (selected_idx == 0).float()
        else:
            selected_base = torch.zeros_like(selected_idx, dtype=torch.float32)
        return selected_action, selected_logp, selected_q, selected_value, selected_base

    def select_delta(
        self,
        obs: np.ndarray,
        base_action: np.ndarray,
        *,
        images: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
                1, self.config.state_dim
            )
            base_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device).reshape(
                1, self.config.action_dim
            )
            images_t = self._images_to_tensor(images[None] if images is not None else None)
            delta, _ = self.actor.sample(obs_t, base_t, images_t, deterministic=deterministic)
        self.actor.train()
        return delta.squeeze(0).cpu().numpy().astype(np.float32)

    def select_action(
        self,
        obs: np.ndarray,
        base_action: np.ndarray,
        *,
        images: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        delta = self.select_delta(obs, base_action, images=images, deterministic=deterministic)
        action = np.asarray(base_action, dtype=np.float32) + delta
        low = np.asarray(self.action_low.cpu(), dtype=np.float32)
        high = np.asarray(self.action_high.cpu(), dtype=np.float32)
        return np.clip(action, low, high).astype(np.float32)

    def select_action_otf(
        self,
        obs: np.ndarray,
        base_action: np.ndarray,
        *,
        n_actions: int,
        images: np.ndarray | None = None,
        include_base_action: bool | None = None,
    ) -> np.ndarray:
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        if include_base_action is None:
            include_base_action = self.config.otf_include_base_action
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
                1, self.config.state_dim
            )
            base_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device).reshape(
                1, self.config.action_dim
            )
            images_t = self._images_to_tensor(images[None] if images is not None else None)
            action, _, _, _, _ = self._select_otf_action_tensors(
                obs_t,
                base_t,
                n_actions,
                images=images_t,
                include_base_action=include_base_action,
                use_target_critic=False,
                soft_value=False,
            )
        self.actor.train()
        self.q1.train()
        self.q2.train()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def pretrain_critic_calql(self, batch: ReplayBatch) -> dict[str, float]:
        b = self._to_tensor_batch(batch)
        with torch.no_grad():
            n = int(self.config.calql_n_actions)
            if self.config.calql_max_target_backup:
                next_actions, next_logp = self._sample_policy_actions(
                    b["next_obs"], b["next_base_actions"], n, b["next_images"]
                )
                q_next = torch.min(
                    self._q_for_action_set(
                        self.q1_target, b["next_obs"], next_actions, b["next_images"]
                    ),
                    self._q_for_action_set(
                        self.q2_target, b["next_obs"], next_actions, b["next_images"]
                    ),
                )
                if self.config.calql_backup_entropy:
                    q_next = q_next - self.alpha.detach() * next_logp
                target_q_next = q_next.gather(1, q_next.argmax(dim=1, keepdim=True))
            else:
                next_actions, next_logp = self._sample_policy_actions(
                    b["next_obs"], b["next_base_actions"], 1, b["next_images"]
                )
                target_q_next = torch.min(
                    self._q_for_action_set(
                        self.q1_target, b["next_obs"], next_actions, b["next_images"]
                    ),
                    self._q_for_action_set(
                        self.q2_target, b["next_obs"], next_actions, b["next_images"]
                    ),
                )
                if self.config.calql_backup_entropy:
                    target_q_next = target_q_next - self.alpha.detach() * next_logp
            target_q = b["rewards"] + self.config.gamma * (1.0 - b["dones"]) * target_q_next

        q1 = self.q1(b["obs"], b["actions"], b["images"])
        q2 = self.q2(b["obs"], b["actions"], b["images"])
        q1_bellman = F.mse_loss(q1, target_q)
        q2_bellman = F.mse_loss(q2, target_q)
        q1_cql, q1_calql_metrics = self._calql_penalty(self.q1, b, q1)
        q2_cql, q2_calql_metrics = self._calql_penalty(self.q2, b, q2)
        q1_loss = q1_bellman + self.config.calql_alpha * q1_cql
        q2_loss = q2_bellman + self.config.calql_alpha * q2_cql
        q_loss = q1_loss + q2_loss

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.config.grad_clip_norm,
            )
        self.q_opt.step()
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self.total_updates += 1

        metrics: dict[str, float] = {
            "q_loss": float(q_loss.detach().cpu()),
            "q1": float(q1.mean().detach().cpu()),
            "q2": float(q2.mean().detach().cpu()),
            "target_q": float(target_q.mean().detach().cpu()),
            "q1_bellman_loss": float(q1_bellman.detach().cpu()),
            "q2_bellman_loss": float(q2_bellman.detach().cpu()),
            "q1_calql_loss": float(q1_cql.detach().cpu()),
            "q2_calql_loss": float(q2_cql.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "calql_alpha": float(self.config.calql_alpha),
            "mc_return": float(b["mc_returns"].mean().detach().cpu()),
        }
        metrics.update({f"q1_{k}": v for k, v in q1_calql_metrics.items()})
        metrics.update({f"q2_{k}": v for k, v in q2_calql_metrics.items()})
        return metrics

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        b = self._to_tensor_batch(batch)
        otf_metrics: dict[str, float] = {}
        with torch.no_grad():
            if self.config.otf_backup_actions > 0:
                _, _, q_next, backup_value, selected_base = self._select_otf_action_tensors(
                    b["next_obs"],
                    b["next_base_actions"],
                    self.config.otf_backup_actions,
                    images=b["next_images"],
                    include_base_action=self.config.otf_include_base_action,
                    use_target_critic=True,
                    soft_value=self.config.otf_backup_entropy,
                )
                target_next = backup_value if self.config.otf_backup_entropy else q_next
                target_q = b["rewards"] + self.config.gamma * (1.0 - b["dones"]) * target_next
                otf_metrics = {
                    "otf_backup_q": float(q_next.mean().detach().cpu()),
                    "otf_backup_value": float(backup_value.mean().detach().cpu()),
                    "otf_backup_base_rate": float(selected_base.mean().detach().cpu()),
                    "otf_backup_candidates": float(
                        int(self.config.otf_backup_actions)
                        + int(bool(self.config.otf_include_base_action))
                    ),
                }
            else:
                next_delta, next_logp = self.actor.sample(
                    b["next_obs"], b["next_base_actions"], b["next_images"]
                )
                next_action = self._clip_action(b["next_base_actions"] + next_delta)
                q_next = torch.min(
                    self.q1_target(b["next_obs"], next_action, b["next_images"]),
                    self.q2_target(b["next_obs"], next_action, b["next_images"]),
                )
                target_q = b["rewards"] + self.config.gamma * (1.0 - b["dones"]) * (
                    q_next - self.alpha.detach() * next_logp
                )

        q1 = self.q1(b["obs"], b["actions"], b["images"])
        q2 = self.q2(b["obs"], b["actions"], b["images"])
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        if self.config.cql_alpha > 0.0:
            q1_loss = q1_loss + self.config.cql_alpha * self._cql_penalty(
                self.q1, b["obs"], q1, b["images"]
            )
            q2_loss = q2_loss + self.config.cql_alpha * self._cql_penalty(
                self.q2, b["obs"], q2, b["images"]
            )

        q_loss = q1_loss + q2_loss
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.config.grad_clip_norm,
            )
        self.q_opt.step()

        metrics: dict[str, float] = {
            "q_loss": float(q_loss.detach().cpu()),
            "q1": float(q1.mean().detach().cpu()),
            "q2": float(q2.mean().detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }
        metrics.update(otf_metrics)

        self.total_updates += 1
        if self.total_updates % max(1, self.config.actor_update_interval) == 0:
            delta, logp = self.actor.sample(b["obs"], b["base_actions"], b["images"])
            policy_action = self._clip_action(b["base_actions"] + delta)
            q_pi = torch.min(
                self.q1(b["obs"], policy_action, b["images"]),
                self.q2(b["obs"], policy_action, b["images"]),
            )
            actor_loss = (self.alpha.detach() * logp - q_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(
                    np.log(max(float(self.config.alpha_min), 1e-12)),
                    np.log(max(float(self.config.alpha_max), 1e-12)),
                )

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            metrics.update(
                actor_loss=float(actor_loss.detach().cpu()),
                alpha_loss=float(alpha_loss.detach().cpu()),
                entropy=float((-logp).mean().detach().cpu()),
            )
        return metrics

    def _cql_penalty(
        self,
        q_net: QNetwork,
        obs: torch.Tensor,
        q_data: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        n = int(self.config.cql_random_actions)
        low = self.action_low.view(1, 1, -1)
        high = self.action_high.view(1, 1, -1)
        random_actions = low + torch.rand(
            (batch_size, n, self.config.action_dim), device=self.device
        ) * (high - low)
        obs_rep = obs[:, None, :].expand(batch_size, n, self.config.state_dim)
        q_rand = q_net(
            obs_rep.reshape(batch_size * n, self.config.state_dim),
            random_actions.reshape(batch_size * n, self.config.action_dim),
            self._repeat_images(images, n),
        ).reshape(batch_size, n)
        return (torch.logsumexp(q_rand, dim=1, keepdim=True) - np.log(n) - q_data).mean()

    def _calql_penalty(
        self, q_net: QNetwork, b: dict[str, torch.Tensor], q_data: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = b["obs"].shape[0]
        n = max(1, int(self.config.calql_n_actions))
        low = self.action_low.view(1, 1, -1)
        high = self.action_high.view(1, 1, -1)
        random_actions = low + torch.rand(
            (batch_size, n, self.config.action_dim), device=self.device
        ) * (high - low)
        q_rand = self._q_for_action_set(q_net, b["obs"], random_actions, b["images"])

        current_actions, current_logp = self._sample_policy_actions(
            b["obs"], b["base_actions"], n, b["images"]
        )
        next_actions, next_logp = self._sample_policy_actions(
            b["next_obs"], b["next_base_actions"], n, b["next_images"]
        )
        q_current = self._q_for_action_set(q_net, b["obs"], current_actions, b["images"])
        # The reference Cal-QL implementation samples next-state actions but
        # evaluates them under the current state for the conservative term.
        q_next = self._q_for_action_set(q_net, b["obs"], next_actions, b["images"])

        lower_bounds = b["mc_returns"].expand_as(q_current)
        bound_rate_current = (q_current < lower_bounds).float().mean()
        bound_rate_next = (q_next < lower_bounds).float().mean()
        q_current = torch.maximum(q_current, lower_bounds)
        q_next = torch.maximum(q_next, lower_bounds)

        if self.config.calql_importance_sample:
            random_log_density = -torch.log(
                torch.clamp(self.action_high - self.action_low, min=1e-6)
            ).sum()
            cql_values = torch.cat(
                [
                    q_rand - random_log_density,
                    q_next - next_logp,
                    q_current - current_logp,
                ],
                dim=1,
            )
        else:
            cql_values = torch.cat([q_rand, q_data, q_next, q_current], dim=1)

        temp = max(float(self.config.calql_temp), 1e-6)
        cql_ood = torch.logsumexp(cql_values / temp, dim=1, keepdim=True) * temp
        cql_diff = (cql_ood - q_data).mean()
        metrics = {
            "calql_diff": float(cql_diff.detach().cpu()),
            "calql_bound_rate_current": float(bound_rate_current.detach().cpu()),
            "calql_bound_rate_next": float(bound_rate_next.detach().cpu()),
            "calql_q_rand": float(q_rand.mean().detach().cpu()),
            "calql_q_current": float(q_current.mean().detach().cpu()),
            "calql_q_next": float(q_next.mean().detach().cpu()),
        }
        return cql_diff, metrics

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for p, p_targ in zip(src.parameters(), dst.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "total_updates": self.total_updates,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "ResidualSAC":
        try:
            payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # Older torch versions do not expose weights_only.
            payload = torch.load(path, map_location=device)
        cfg_dict = dict(payload["config"])
        if isinstance(cfg_dict.get("action_low"), list):
            cfg_dict["action_low"] = tuple(cfg_dict["action_low"])
        if isinstance(cfg_dict.get("action_high"), list):
            cfg_dict["action_high"] = tuple(cfg_dict["action_high"])
        if isinstance(cfg_dict.get("image_shape"), list):
            cfg_dict["image_shape"] = tuple(cfg_dict["image_shape"])
        agent = cls(SACConfig(**cfg_dict), device=device)
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.q1_target.load_state_dict(payload["q1_target"])
        agent.q2_target.load_state_dict(payload["q2_target"])
        agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        agent.total_updates = int(payload.get("total_updates", 0))
        return agent
