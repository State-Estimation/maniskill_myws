"""Lightweight RECAP-style distributional base-policy value model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maniskill_myws.openpi_bridge.keypath import get_by_path
from maniskill_myws.openpi_bridge.obs_to_openpi import _to_uint8_hwc
from maniskill_myws.openpi_bridge.remote_policy import (
    SAFE_LATENT_DIM,
    SAFE_LATENT_PROTOCOL,
)


VALUE_MODEL_SCHEMA = "safe_visual_recap_value_v1"
VALUE_FEATURE_SCHEMA = "safe_visual_recap_value_features_v1"


def resize_value_rgb(image: object, height: int, width: int) -> np.ndarray:
    """Convert an observation image to the value model's uint8 layout."""

    value = _to_uint8_hwc(image)
    if value.shape[:2] == (height, width):
        return value.copy()
    tensor = torch.from_numpy(value).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()


def value_images_from_observation(
    observation: dict[str, Any],
    *,
    image_keys: tuple[str, ...] | list[str],
    height: int,
    width: int,
) -> np.ndarray:
    return np.stack(
        [resize_value_rgb(get_by_path(observation, key), height, width) for key in image_keys],
        axis=0,
    )


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_num_groups(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.conv1(value)))
        hidden = self.norm2(self.conv2(hidden))
        return F.gelu(hidden + self.shortcut(value))


class IndependentVisualEncoder(nn.Module):
    """Compact visual encoder trained independently of the frozen Pi0 model."""

    def __init__(self, num_views: int, feature_dim: int) -> None:
        super().__init__()
        if num_views <= 0 or feature_dim <= 0:
            raise ValueError("Visual encoder dimensions must be positive")
        self.num_views = int(num_views)
        self.stem = nn.Sequential(
            nn.Conv2d(3 * num_views, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_num_groups(32), 32),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock(32, 32, 1),
            _ResidualBlock(32, 64, 2),
            _ResidualBlock(64, 128, 2),
            _ResidualBlock(128, 256, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != self.num_views:
            raise ValueError("Images must have shape [B, views, H, W, 3] with the configured views")
        if images.shape[-1] != 3:
            raise ValueError("Value model requires RGB images")
        value = images
        if value.dtype == torch.uint8:
            value = value.float().div(127.5).sub(1.0)
        else:
            value = value.float()
            if value.numel() and float(value.detach().max()) > 1.5:
                value = value.div(127.5).sub(1.0)
            else:
                value = value.mul(2.0).sub(1.0)
        value = value.permute(0, 1, 4, 2, 3).reshape(
            value.shape[0], self.num_views * 3, value.shape[2], value.shape[3]
        )
        return self.head(self.blocks(self.stem(value)))


class SafeEndpointTokenEncoder(nn.Module):
    """Encode SAFE's four unpooled diffusion/horizon endpoint tokens."""

    token_count = 4
    token_dim = 1024

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or feature_dim % 4 != 0:
            raise ValueError("SAFE latent feature dimension must be positive and divisible by 4")
        self.input_norm = nn.LayerNorm(self.token_dim)
        self.input_projection = nn.Linear(self.token_dim, feature_dim)
        self.token_type = nn.Parameter(torch.empty(self.token_count, feature_dim))
        nn.init.normal_(self.token_type, std=0.02)
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads=4, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.pool_score = nn.Linear(feature_dim, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != SAFE_LATENT_DIM:
            raise ValueError(f"SAFE latent must have shape [B, {SAFE_LATENT_DIM}]")
        tokens = latent.reshape(-1, self.token_count, self.token_dim)
        tokens = self.input_projection(self.input_norm(tokens))
        tokens = tokens + self.token_type.unsqueeze(0)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.output_norm(tokens + attended)
        weights = torch.softmax(self.pool_score(tokens), dim=1)
        return F.gelu(torch.sum(weights * tokens, dim=1))


@dataclass(frozen=True)
class DistributionalValueConfig:
    state_dim: int
    action_dim: int = 8
    chunk_len: int = 10
    max_episode_steps: int = 500
    num_views: int = 2
    image_height: int = 128
    image_width: int = 128
    latent_dim: int = SAFE_LATENT_DIM
    modality_dim: int = 256
    feature_dim: int = 256
    hidden_dim: int = 256
    failure_value: float = -1.25
    latent_protocol: str = SAFE_LATENT_PROTOCOL

    def __post_init__(self) -> None:
        if (
            min(
                self.state_dim,
                self.action_dim,
                self.chunk_len,
                self.max_episode_steps,
                self.num_views,
                self.image_height,
                self.image_width,
                self.modality_dim,
                self.feature_dim,
                self.hidden_dim,
            )
            <= 0
        ):
            raise ValueError("Value model dimensions must be positive")
        if self.max_episode_steps % self.chunk_len != 0:
            raise ValueError("Value horizon must be divisible by chunk_len")
        if self.latent_dim != SAFE_LATENT_DIM or self.latent_protocol != SAFE_LATENT_PROTOCOL:
            raise ValueError("Value model requires the SAFE full endpoint latent protocol")
        if self.failure_value >= -1.0:
            raise ValueError("failure_value must be below the longest successful return")

    @property
    def max_remaining_chunks(self) -> int:
        return self.max_episode_steps // self.chunk_len

    @property
    def num_return_bins(self) -> int:
        return self.max_remaining_chunks + 2

    @property
    def critic_feature_dim(self) -> int:
        return self.feature_dim + self.num_return_bins + 3


class DistributionalBaseValueModel(nn.Module):
    """Predict base-policy failure and time-to-success as a categorical return."""

    def __init__(self, config: DistributionalValueConfig) -> None:
        super().__init__()
        self.config = config
        m = config.modality_dim
        self.visual = IndependentVisualEncoder(config.num_views, m)
        self.latent = SafeEndpointTokenEncoder(m)
        self.state = nn.Sequential(
            nn.Linear(config.state_dim + 1, m),
            nn.LayerNorm(m),
            nn.GELU(),
        )
        self.reference = nn.Sequential(
            nn.Linear(config.chunk_len * config.action_dim, m),
            nn.LayerNorm(m),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * m, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU(),
        )
        self.return_head = nn.Linear(config.feature_dim, config.num_return_bins)
        support = np.empty(config.num_return_bins, dtype=np.float32)
        support[0] = config.failure_value
        remaining = np.arange(config.max_remaining_chunks + 1, dtype=np.float32)
        support[1:] = -remaining / float(config.max_remaining_chunks)
        self.register_buffer("return_support", torch.from_numpy(support))

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        latent: torch.Tensor,
        ref_chunk: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError("Value state has an invalid shape")
        if ref_chunk.shape != (
            state.shape[0],
            self.config.chunk_len,
            self.config.action_dim,
        ):
            raise ValueError("Value reference action chunk has an invalid shape")
        remaining = 1.0 - step_ids.to(state.dtype).reshape(-1, 1) / float(
            self.config.max_episode_steps
        )
        state_time = torch.cat([state, remaining.clamp(0.0, 1.0)], dim=-1)
        feature = self.fusion(
            torch.cat(
                [
                    self.visual(images),
                    self.latent(latent),
                    self.state(state_time),
                    self.reference(ref_chunk.flatten(start_dim=1)),
                ],
                dim=-1,
            )
        )
        logits = self.return_head(feature)
        probability = torch.softmax(logits, dim=-1)
        expected_value = torch.sum(probability * self.return_support, dim=-1)
        entropy = -torch.sum(probability * torch.log(probability.clamp_min(1e-8)), dim=-1)
        expected_remaining = torch.sum(
            probability[:, 1:]
            * torch.arange(
                self.config.max_remaining_chunks + 1,
                device=probability.device,
                dtype=probability.dtype,
            ),
            dim=-1,
        )
        return {
            "logits": logits,
            "probability": probability,
            "feature": feature,
            "expected_value": expected_value,
            "failure_probability": probability[:, 0],
            "entropy": entropy,
            "expected_remaining_chunks": expected_remaining,
        }

    def critic_features(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                output["feature"],
                output["probability"],
                output["expected_value"][:, None],
                output["failure_probability"][:, None],
                output["entropy"][:, None],
            ],
            dim=-1,
        )

    def save(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": VALUE_MODEL_SCHEMA,
            "config": asdict(self.config),
            "model": self.state_dict(),
            "metadata": dict(metadata or {}),
        }
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
            temporary = Path(file.name)
            torch.save(payload, file)
        temporary.replace(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["DistributionalBaseValueModel", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != VALUE_MODEL_SCHEMA:
            raise ValueError("Checkpoint is not a maintained distributional V_base model")
        model = cls(DistributionalValueConfig(**dict(payload["config"]))).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model, dict(payload.get("metadata") or {})


@dataclass(frozen=True)
class ValueEstimate:
    critic_features: np.ndarray
    potential: float
    failure_probability: float
    entropy: float
    expected_remaining_chunks: float


@dataclass(frozen=True)
class ValueProgressEstimate:
    """Progress proxy derived from the successful-return distribution."""

    success_probability: float
    conditional_remaining_chunks: float
    completion_proximity: float


def value_progress_estimate(
    estimate: ValueEstimate,
    *,
    max_remaining_chunks: int,
    probability_epsilon: float = 1e-8,
) -> ValueProgressEstimate:
    """Convert unconditional time-to-success mass into a conditional proxy.

    ``expected_remaining_chunks`` includes the probability of success. Dividing
    by that probability recovers E[remaining chunks | success]. When the model
    assigns effectively no success mass, the conservative proxy is zero
    completion rather than an unstable division.
    """

    if max_remaining_chunks <= 0:
        raise ValueError("max_remaining_chunks must be positive")
    if not 0.0 < probability_epsilon < 1.0:
        raise ValueError("probability_epsilon must lie in (0,1)")
    values = (
        estimate.failure_probability,
        estimate.expected_remaining_chunks,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Value progress inputs must be finite")
    if not 0.0 <= estimate.failure_probability <= 1.0:
        raise ValueError("failure_probability must lie in [0,1]")
    if not 0.0 <= estimate.expected_remaining_chunks <= max_remaining_chunks:
        raise ValueError("expected_remaining_chunks lies outside the value support")

    success_probability = 1.0 - float(estimate.failure_probability)
    if success_probability <= probability_epsilon:
        conditional_remaining = float(max_remaining_chunks)
    else:
        conditional_remaining = float(
            np.clip(
                estimate.expected_remaining_chunks / success_probability,
                0.0,
                float(max_remaining_chunks),
            )
        )
    completion_proximity = float(
        np.clip(
            1.0 - conditional_remaining / float(max_remaining_chunks),
            0.0,
            1.0,
        )
    )
    return ValueProgressEstimate(
        success_probability=success_probability,
        conditional_remaining_chunks=conditional_remaining,
        completion_proximity=completion_proximity,
    )


@torch.no_grad()
def infer_value_estimate(
    model: DistributionalBaseValueModel,
    *,
    images: np.ndarray,
    state: np.ndarray,
    latent: np.ndarray,
    ref_chunk: np.ndarray,
    step_id: int,
) -> ValueEstimate:
    """Run a frozen value model at one base-policy chunk boundary."""

    config = model.config
    expected_images = (
        config.num_views,
        config.image_height,
        config.image_width,
        3,
    )
    arrays = {
        "images": np.asarray(images),
        "state": np.asarray(state, dtype=np.float32),
        "latent": np.asarray(latent, dtype=np.float32),
        "ref_chunk": np.asarray(ref_chunk, dtype=np.float32),
    }
    expected = {
        "images": expected_images,
        "state": (config.state_dim,),
        "latent": (config.latent_dim,),
        "ref_chunk": (config.chunk_len, config.action_dim),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Value {name} shape {arrays[name].shape} != {shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"Value {name} contains NaN or Inf")
    if not 0 <= int(step_id) <= config.max_episode_steps:
        raise ValueError("Value step_id is outside the configured horizon")
    device = next(model.parameters()).device
    output = model(
        torch.as_tensor(arrays["images"][None], device=device),
        torch.as_tensor(arrays["state"][None], device=device),
        torch.as_tensor(arrays["latent"][None], device=device),
        torch.as_tensor(arrays["ref_chunk"][None], device=device),
        torch.as_tensor([step_id], dtype=torch.long, device=device),
    )
    features = model.critic_features(output)[0].float().cpu().numpy()
    return ValueEstimate(
        critic_features=features.astype(np.float32, copy=False),
        potential=float(output["expected_value"][0].cpu()),
        failure_probability=float(output["failure_probability"][0].cpu()),
        entropy=float(output["entropy"][0].cpu()),
        expected_remaining_chunks=float(output["expected_remaining_chunks"][0].cpu()),
    )


def value_potential_shaping(
    *,
    current_potential: float,
    next_potential: float,
    gamma: float,
    duration: int,
    chunk_len: int,
    weight: float,
) -> float:
    """Potential difference matching the residual critic's macro discount."""

    values = (current_potential, next_potential, gamma, weight)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Potential shaping inputs must be finite")
    if not 0.0 <= gamma <= 1.0 or weight < 0.0:
        raise ValueError("Potential shaping gamma/weight are invalid")
    if not 1 <= duration <= chunk_len:
        raise ValueError("Potential shaping duration is outside the chunk")
    discount = float(gamma) ** (duration / float(chunk_len))
    return float(weight) * (discount * float(next_potential) - float(current_potential))


def return_bin_target(
    *,
    success: bool,
    boundary_index: int,
    boundary_count: int,
    max_remaining_chunks: int,
) -> int:
    if boundary_count <= 0 or not 0 <= boundary_index < boundary_count:
        raise ValueError("Boundary index/count are invalid")
    if not success:
        return 0
    remaining = boundary_count - boundary_index
    if not 1 <= remaining <= max_remaining_chunks:
        raise ValueError("Successful remaining-chunk target is outside support")
    return 1 + remaining
