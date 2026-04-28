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
    action_scale: float = 0.5
    target_entropy: float | None = None
    actor_update_interval: int = 2
    grad_clip_norm: float = 1.0
    cql_alpha: float = 0.0
    cql_random_actions: int = 10
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


class GaussianResidualActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_scale: float) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.body = MLP(state_dim + action_dim, 2 * action_dim, hidden_dim)
        self.register_buffer("scale", torch.full((action_dim,), float(action_scale)))

    def forward(self, obs: torch.Tensor, base_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.body(torch.cat([obs, base_action], dim=-1))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor, base_action: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs, base_action)
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
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class ResidualSAC:
    def __init__(self, config: SACConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor = GaussianResidualActor(
            config.state_dim, config.action_dim, config.hidden_dim, config.action_scale
        ).to(self.device)
        self.q1 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.q_opt = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr
        )
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = (
            float(config.target_entropy)
            if config.target_entropy is not None
            else -float(config.action_dim)
        )
        low = config.action_low if config.action_low is not None else (-1.0,) * config.action_dim
        high = config.action_high if config.action_high is not None else (1.0,) * config.action_dim
        self.action_low = torch.as_tensor(low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(high, dtype=torch.float32, device=self.device)
        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

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
        }

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def select_delta(
        self, obs: np.ndarray, base_action: np.ndarray, *, deterministic: bool = True
    ) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
                1, self.config.state_dim
            )
            base_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device).reshape(
                1, self.config.action_dim
            )
            delta, _ = self.actor.sample(obs_t, base_t, deterministic=deterministic)
        self.actor.train()
        return delta.squeeze(0).cpu().numpy().astype(np.float32)

    def select_action(
        self, obs: np.ndarray, base_action: np.ndarray, *, deterministic: bool = True
    ) -> np.ndarray:
        delta = self.select_delta(obs, base_action, deterministic=deterministic)
        action = np.asarray(base_action, dtype=np.float32) + delta
        low = np.asarray(self.action_low.cpu(), dtype=np.float32)
        high = np.asarray(self.action_high.cpu(), dtype=np.float32)
        return np.clip(action, low, high).astype(np.float32)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        b = self._to_tensor_batch(batch)
        with torch.no_grad():
            next_delta, next_logp = self.actor.sample(b["next_obs"], b["next_base_actions"])
            next_action = self._clip_action(b["next_base_actions"] + next_delta)
            q_next = torch.min(
                self.q1_target(b["next_obs"], next_action),
                self.q2_target(b["next_obs"], next_action),
            )
            target_q = b["rewards"] + self.config.gamma * (1.0 - b["dones"]) * (
                q_next - self.alpha.detach() * next_logp
            )

        q1 = self.q1(b["obs"], b["actions"])
        q2 = self.q2(b["obs"], b["actions"])
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        if self.config.cql_alpha > 0.0:
            q1_loss = q1_loss + self.config.cql_alpha * self._cql_penalty(self.q1, b["obs"], q1)
            q2_loss = q2_loss + self.config.cql_alpha * self._cql_penalty(self.q2, b["obs"], q2)

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

        self.total_updates += 1
        if self.total_updates % max(1, self.config.actor_update_interval) == 0:
            delta, logp = self.actor.sample(b["obs"], b["base_actions"])
            policy_action = self._clip_action(b["base_actions"] + delta)
            q_pi = torch.min(self.q1(b["obs"], policy_action), self.q2(b["obs"], policy_action))
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

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            metrics.update(
                actor_loss=float(actor_loss.detach().cpu()),
                alpha_loss=float(alpha_loss.detach().cpu()),
                entropy=float((-logp).mean().detach().cpu()),
            )
        return metrics

    def _cql_penalty(self, q_net: QNetwork, obs: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
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
        ).reshape(batch_size, n)
        return (torch.logsumexp(q_rand, dim=1, keepdim=True) - np.log(n) - q_data).mean()

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
        agent = cls(SACConfig(**cfg_dict), device=device)
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.q1_target.load_state_dict(payload["q1_target"])
        agent.q2_target.load_state_dict(payload["q2_target"])
        agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        agent.total_updates = int(payload.get("total_updates", 0))
        return agent
