from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(in_dim)
        for _ in range(max(1, int(num_layers))):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    """Compact visual encoder for stacked ManiSkill RGB views."""

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
        visual_encoder: Literal["none", "resnet10"],
        visual_latent_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_shape = image_shape
        if visual_encoder == "none" or image_shape is None:
            self.visual: nn.Module | None = None
            self.out_dim = self.state_dim
        elif visual_encoder == "resnet10":
            self.visual = ResNetV1_10Encoder(image_shape, int(visual_latent_dim))
            self.out_dim = self.state_dim + int(visual_latent_dim)
        else:
            raise ValueError(f"Unsupported visual_encoder: {visual_encoder}")

    def forward(self, state: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        if self.visual is None:
            return state
        if images is None:
            raise ValueError("Visual RLT encoder requires images")
        return torch.cat([state, self.visual(images)], dim=-1)


class ChunkActor(nn.Module):
    """RLT-style chunk actor conditioned on ManiSkill state/RGB and a reference chunk."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int = 8,
        chunk_len: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.01,
        action_delta_scale: float = 0.1,
        output_mode: Literal["delta", "absolute"] = "delta",
        image_shape: tuple[int, ...] | None = None,
        visual_encoder: Literal["none", "resnet10"] = "none",
        visual_latent_dim: int = 256,
        action_low: tuple[float, ...] | None = None,
        action_high: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.fixed_std = float(fixed_std)
        self.action_delta_scale = float(action_delta_scale)
        self.output_mode = output_mode
        self.obs_encoder = ObservationEncoder(
            state_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=visual_latent_dim,
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(self.obs_encoder.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.ref_proj = nn.Sequential(
            nn.Linear(self.chunk_len * self.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.trunk = MLP(
            hidden_dim * 2,
            self.chunk_len * self.action_dim,
            hidden_dim,
            num_layers=num_layers,
        )
        low = action_low if action_low is not None else (-1.0,) * self.action_dim
        high = action_high if action_high is not None else (1.0,) * self.action_dim
        self.register_buffer("action_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high, dtype=torch.float32))

    def _features(
        self,
        state: torch.Tensor,
        ref_chunk: torch.Tensor,
        images: torch.Tensor | None,
    ) -> torch.Tensor:
        obs_features = self.obs_proj(self.obs_encoder(state, images))
        ref_flat = ref_chunk.reshape(ref_chunk.shape[0], self.chunk_len * self.action_dim)
        ref_features = self.ref_proj(ref_flat)
        return torch.cat([obs_features, ref_features], dim=-1)

    def mean(
        self,
        state: torch.Tensor,
        ref_chunk: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.trunk(self._features(state, ref_chunk, images))
        raw = raw.reshape(state.shape[0], self.chunk_len, self.action_dim)
        if self.output_mode == "absolute":
            return self._clip(raw)
        if self.output_mode != "delta":
            raise ValueError(f"Unsupported actor output_mode: {self.output_mode}")
        delta = torch.tanh(raw) * self.action_delta_scale
        return self._clip(ref_chunk + delta)

    def sample(
        self,
        state: torch.Tensor,
        ref_chunk: torch.Tensor,
        images: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mu = self.mean(state, ref_chunk, images)
        if deterministic:
            return mu
        noise = Normal(torch.zeros_like(mu), torch.full_like(mu, self.fixed_std)).rsample()
        return self._clip(mu + noise)

    def _clip(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.view(1, 1, self.action_dim)
        high = self.action_high.view(1, 1, self.action_dim)
        return torch.max(torch.min(action, high), low)


class QNetwork(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int = 8,
        chunk_len: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        image_shape: tuple[int, ...] | None = None,
        visual_encoder: Literal["none", "resnet10"] = "none",
        visual_latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.obs_encoder = ObservationEncoder(
            state_dim,
            image_shape=image_shape,
            visual_encoder=visual_encoder,
            visual_latent_dim=visual_latent_dim,
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(self.obs_encoder.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.chunk_len * self.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.trunk = MLP(hidden_dim * 2, 1, hidden_dim, num_layers=num_layers)

    def forward(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_features = self.obs_proj(self.obs_encoder(state, images))
        action_flat = action_chunk.reshape(action_chunk.shape[0], self.chunk_len * self.action_dim)
        action_features = self.action_proj(action_flat)
        return self.trunk(torch.cat([obs_features, action_features], dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.q1 = QNetwork(**kwargs)
        self.q2 = QNetwork(**kwargs)

    def q_values(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action_chunk, images), self.q2(state, action_chunk, images)
