"""SAFE frozen-latent context and residual actor used by the V-gated bandit."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)
from maniskill_myws.rlt.value_model import SafeEndpointTokenEncoder


SAFE_ENDPOINT_LATENT_ENCODER = "safe_endpoint_attention"


def make_runtime_identity(
    *,
    env_id: str,
    obs_mode: str,
    reward_mode: str,
    control_mode: str,
    sim_backend: str,
    render_backend: str,
    enhanced_determinism: bool,
    prompt: str,
    image_key: str,
    wrist_image_key: str,
    state_keys: list[str] | tuple[str, ...],
    resize: int,
    chunk_len: int,
    max_episode_steps: int,
    openpi_policy_identity_sha256: str,
    latent_protocol: str = SAFE_LATENT_PROTOCOL,
    latent_dim: int = SAFE_LATENT_DIM,
) -> dict[str, Any]:
    """Build the immutable deployment identity checked by training/evaluation."""

    if len(openpi_policy_identity_sha256) != 64:
        raise ValueError("OpenPI policy identity must be a SHA-256 digest")
    return {
        "schema": "lightweight_rlt_runtime_identity_v2",
        "env_id": str(env_id),
        "obs_mode": str(obs_mode),
        "reward_mode": str(reward_mode),
        "control_mode": str(control_mode),
        "sim_backend": str(sim_backend),
        "render_backend": str(render_backend),
        "enhanced_determinism": bool(enhanced_determinism),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_key": str(image_key),
        "wrist_image_key": str(wrist_image_key),
        "state_keys": [str(key) for key in state_keys],
        "resize": int(resize),
        "chunk_len": int(chunk_len),
        "max_episode_steps": int(max_episode_steps),
        "openpi_policy_identity_sha256": str(openpi_policy_identity_sha256),
        "frozen_latent_protocol": str(latent_protocol),
        "frozen_latent_dim": int(latent_dim),
    }


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(num_layers)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FrozenLatentContextEncoder(nn.Module):
    """Fuse robot state/time, SAFE endpoint latent, and the reference chunk."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.latent = SafeEndpointTokenEncoder(hidden)
        self.state = nn.Sequential(
            nn.Linear(config.state_dim + 1, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.reference = nn.Sequential(
            nn.Linear(config.chunk_len * config.action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.fusion = _mlp(3 * hidden, config.context_dim, hidden, 1)

    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        ref_chunk: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        remaining = 1.0 - step_ids.to(state.dtype).reshape(-1, 1) / float(
            self.config.max_episode_steps
        )
        state_time = torch.cat([state, remaining.clamp(0.0, 1.0)], dim=-1)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("Frozen latent must have shape [B, latent_dim]")
        return self.fusion(
            torch.cat(
                [
                    self.state(state_time),
                    self.latent(latent),
                    self.reference(ref_chunk.flatten(start_dim=1)),
                ],
                dim=-1,
            )
        )


class ContinuousResidualActor(nn.Module):
    """Predict a bounded residual at six knots and interpolate it over 10 steps."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(
            config.context_dim,
            config.exploration_knots * config.action_dim,
            config.hidden_dim,
            config.num_layers,
        )
        output = self.trunk[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def _expand_knots(self, knots: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            knots.permute(0, 2, 1),
            size=self.config.chunk_len,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

    def mean(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.trunk(context)
        knots = self.config.actor_residual_limit * torch.tanh(
            raw.reshape(-1, self.config.exploration_knots, self.config.action_dim)
        )
        return self._expand_knots(knots)
