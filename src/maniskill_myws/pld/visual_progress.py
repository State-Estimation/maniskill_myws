from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .sac import MLP, ResNetV1_10Encoder


@dataclass
class VisualProgressConfig:
    context_frames: int
    num_views: int
    image_height: int
    image_width: int
    image_channels: int = 3
    visual_latent_dim: int = 256
    embedding_dim: int = 128
    hidden_dim: int = 256

    @property
    def clip_shape(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.context_frames),
            int(self.num_views),
            int(self.image_height),
            int(self.image_width),
            int(self.image_channels),
        )


class VisualTaskProgressModel(nn.Module):
    """Vision-only continuous progress model for short RGB clips.

    Task phases are represented implicitly in the cycle-consistent embedding;
    the architecture does not prescribe a stage count or expose stage classes.
    """

    def __init__(self, config: VisualProgressConfig) -> None:
        super().__init__()
        if config.context_frames <= 0 or config.num_views <= 0:
            raise ValueError("context_frames and num_views must be positive")
        if config.image_channels != 3:
            raise ValueError("Only RGB clips are supported")
        self.config = config
        flattened_views = config.context_frames * config.num_views
        self.visual = ResNetV1_10Encoder(
            (
                flattened_views,
                config.image_height,
                config.image_width,
                config.image_channels,
            ),
            config.visual_latent_dim,
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(config.visual_latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.progress_head = MLP(
            config.visual_latent_dim,
            1,
            config.hidden_dim,
        )
        self.success_head = MLP(
            config.visual_latent_dim,
            1,
            config.hidden_dim,
        )

    def clips_to_tensor(
        self,
        clips: np.ndarray | torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(clips, device=device)
        expected_tail = self.config.clip_shape
        if tuple(tensor.shape[-5:]) != expected_tail:
            raise ValueError(
                f"Expected clip tail {expected_tail}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            tensor = tensor.float()
            if tensor.numel() and float(tensor.detach().max()) > 1.5:
                tensor = tensor / 255.0
        leading = tensor.shape[:-5]
        # (..., context, views, H, W, C) -> (N, context*views*C, H, W)
        tensor = tensor.reshape(
            -1,
            self.config.context_frames,
            self.config.num_views,
            self.config.image_height,
            self.config.image_width,
            self.config.image_channels,
        )
        tensor = tensor.permute(0, 1, 2, 5, 3, 4).reshape(
            -1,
            self.config.context_frames
            * self.config.num_views
            * self.config.image_channels,
            self.config.image_height,
            self.config.image_width,
        )
        return tensor.reshape(*leading, *tensor.shape[-3:])

    def forward(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = (
            self.config.context_frames
            * self.config.num_views
            * self.config.image_channels
        )
        if clips.ndim < 4 or clips.shape[-3] != expected_channels:
            raise ValueError(
                "forward expects clips_to_tensor output with channel count "
                f"{expected_channels}, got {tuple(clips.shape)}"
            )
        leading = clips.shape[:-3]
        flat = clips.reshape(-1, *clips.shape[-3:])
        visual_features = self.visual(flat)
        embedding = F.normalize(self.embedding_head(visual_features), dim=-1)
        progress = torch.sigmoid(self.progress_head(visual_features))
        success_logit = self.success_head(visual_features)

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(*leading, *value.shape[1:])

        return {
            "embedding": restore(embedding),
            "progress": restore(progress),
            "success_logit": restore(success_logit),
        }


def _temporal_cycle_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable A->B->A cycle matching for two successful videos."""
    if temperature <= 0:
        raise ValueError("cycle temperature must be positive")
    logits_ab = torch.einsum("bid,bjd->bij", embedding_a, embedding_b) / temperature
    probability_ab = torch.softmax(logits_ab, dim=-1)
    aligned_b = torch.einsum("bij,bjd->bid", probability_ab, embedding_b)
    logits_back = torch.einsum("bid,bjd->bij", aligned_b, embedding_a) / temperature
    points = embedding_a.shape[1]
    target = torch.arange(points, device=embedding_a.device)[None].expand(
        embedding_a.shape[0], -1
    )
    loss = F.cross_entropy(logits_back.reshape(-1, points), target.reshape(-1))
    accuracy = (logits_back.argmax(dim=-1) == target).float().mean()
    return loss, accuracy, probability_ab


def _ordered_progress_loss(
    progress: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = progress[:, 1:] - progress[:, :-1]
    loss = F.relu(float(margin) - delta).mean()
    violation = (delta < -1e-4).float().mean()
    mean_delta = delta.mean()
    return loss, violation, mean_delta


def visual_progress_training_loss(
    output_a: dict[str, torch.Tensor],
    output_b: dict[str, torch.Tensor],
    output_failure: dict[str, torch.Tensor],
    *,
    cycle_temperature: float = 0.1,
    order_margin: float = 0.02,
    preference_margin: float = 0.2,
    cycle_weight: float = 1.0,
    alignment_weight: float = 0.5,
    endpoint_weight: float = 2.0,
    order_weight: float = 1.0,
    latent_smoothness_weight: float = 0.05,
    smoothness_weight: float = 0.1,
    success_weight: float = 1.0,
    preference_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss using only temporal order and episode-level success/failure labels."""
    progress_a = output_a["progress"]
    progress_b = output_b["progress"]
    progress_f = output_failure["progress"]

    cycle_ab, cycle_acc_ab, probability_ab = _temporal_cycle_loss(
        output_a["embedding"],
        output_b["embedding"],
        temperature=cycle_temperature,
    )
    cycle_ba, cycle_acc_ba, probability_ba = _temporal_cycle_loss(
        output_b["embedding"],
        output_a["embedding"],
        temperature=cycle_temperature,
    )
    cycle_loss = 0.5 * (cycle_ab + cycle_ba)
    cycle_accuracy = 0.5 * (cycle_acc_ab + cycle_acc_ba)

    aligned_progress_b = torch.einsum(
        "bij,bjk->bik", probability_ab, progress_b
    )
    aligned_progress_a = torch.einsum(
        "bij,bjk->bik", probability_ba, progress_a
    )
    alignment_loss = 0.5 * (
        F.smooth_l1_loss(progress_a, aligned_progress_b)
        + F.smooth_l1_loss(progress_b, aligned_progress_a)
    )

    zero = torch.zeros_like(progress_a[:, 0])
    one = torch.ones_like(progress_a[:, -1])
    endpoint_loss = 0.25 * (
        F.mse_loss(progress_a[:, 0], zero)
        + F.mse_loss(progress_b[:, 0], zero)
        + F.mse_loss(progress_a[:, -1], one)
        + F.mse_loss(progress_b[:, -1], one)
    )

    order_a, violation_a, delta_a = _ordered_progress_loss(
        progress_a, margin=order_margin
    )
    order_b, violation_b, delta_b = _ordered_progress_loss(
        progress_b, margin=order_margin
    )
    order_loss = 0.5 * (order_a + order_b)
    monotonic_violation = 0.5 * (violation_a + violation_b)
    mean_progress_delta = 0.5 * (delta_a + delta_b)

    velocity_a = output_a["embedding"][:, 1:] - output_a["embedding"][:, :-1]
    velocity_b = output_b["embedding"][:, 1:] - output_b["embedding"][:, :-1]
    latent_step_distance = 0.5 * (
        velocity_a.norm(dim=-1).mean() + velocity_b.norm(dim=-1).mean()
    )
    if velocity_a.shape[1] >= 2:
        acceleration_a = velocity_a[:, 1:] - velocity_a[:, :-1]
        acceleration_b = velocity_b[:, 1:] - velocity_b[:, :-1]
        latent_smoothness_loss = 0.5 * (
            acceleration_a.norm(dim=-1).mean()
            + acceleration_b.norm(dim=-1).mean()
        )
    else:  # pragma: no cover - sequence_points >= 3
        latent_smoothness_loss = progress_a.new_zeros(())

    if progress_a.shape[1] >= 3:
        curvature_a = progress_a[:, 2:] - 2.0 * progress_a[:, 1:-1] + progress_a[:, :-2]
        curvature_b = progress_b[:, 2:] - 2.0 * progress_b[:, 1:-1] + progress_b[:, :-2]
        smoothness_loss = 0.5 * (curvature_a.abs().mean() + curvature_b.abs().mean())
    else:  # pragma: no cover - dataset enforces >= 3 sequence points
        smoothness_loss = progress_a.new_zeros(())

    sequence_points = progress_a.shape[1]
    outcome_weight = torch.linspace(
        0.25,
        1.0,
        sequence_points,
        device=progress_a.device,
        dtype=progress_a.dtype,
    )[None, :, None]
    positive_a = F.binary_cross_entropy_with_logits(
        output_a["success_logit"],
        torch.ones_like(output_a["success_logit"]),
        reduction="none",
    )
    positive_b = F.binary_cross_entropy_with_logits(
        output_b["success_logit"],
        torch.ones_like(output_b["success_logit"]),
        reduction="none",
    )
    negative_f = F.binary_cross_entropy_with_logits(
        output_failure["success_logit"],
        torch.zeros_like(output_failure["success_logit"]),
        reduction="none",
    )
    # Give the two positive sequences half weight each so classes are balanced.
    success_loss = (
        0.5 * (positive_a * outcome_weight).mean()
        + 0.5 * (positive_b * outcome_weight).mean()
        + (negative_f * outcome_weight).mean()
    ) / 2.0

    successful_terminal = 0.5 * (progress_a[:, -1] + progress_b[:, -1])
    failure_terminal = progress_f[:, -1]
    progress_preference = F.relu(
        float(preference_margin) - (successful_terminal - failure_terminal)
    ).mean()
    successful_terminal_logit = 0.5 * (
        output_a["success_logit"][:, -1] + output_b["success_logit"][:, -1]
    )
    failure_terminal_logit = output_failure["success_logit"][:, -1]
    outcome_preference = F.relu(
        1.0 - (successful_terminal_logit - failure_terminal_logit)
    ).mean()
    preference_loss = progress_preference + 0.25 * outcome_preference

    total = (
        float(cycle_weight) * cycle_loss
        + float(alignment_weight) * alignment_loss
        + float(endpoint_weight) * endpoint_loss
        + float(order_weight) * order_loss
        + float(latent_smoothness_weight) * latent_smoothness_loss
        + float(smoothness_weight) * smoothness_loss
        + float(success_weight) * success_loss
        + float(preference_weight) * preference_loss
    )

    with torch.no_grad():
        success_probability = torch.sigmoid(
            torch.cat(
                [
                    output_a["success_logit"],
                    output_b["success_logit"],
                    output_failure["success_logit"],
                ],
                dim=1,
            )
        )
        success_target = torch.cat(
            [
                torch.ones_like(output_a["success_logit"]),
                torch.ones_like(output_b["success_logit"]),
                torch.zeros_like(output_failure["success_logit"]),
            ],
            dim=1,
        )
        success_accuracy = (
            (success_probability >= 0.5) == (success_target >= 0.5)
        ).float().mean()

    metrics: dict[str, torch.Tensor] = {
        "loss": total.detach(),
        "cycle_loss": cycle_loss.detach(),
        "cycle_accuracy": cycle_accuracy.detach(),
        "alignment_loss": alignment_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
        "order_loss": order_loss.detach(),
        "monotonic_violation": monotonic_violation.detach(),
        "mean_progress_delta": mean_progress_delta.detach(),
        "latent_smoothness_loss": latent_smoothness_loss.detach(),
        "latent_step_distance": latent_step_distance.detach(),
        "smoothness_loss": smoothness_loss.detach(),
        "success_bce": success_loss.detach(),
        "success_accuracy": success_accuracy.detach(),
        "preference_loss": preference_loss.detach(),
        "success_start_progress": (
            0.5 * (progress_a[:, 0].mean() + progress_b[:, 0].mean())
        ).detach(),
        "success_terminal_progress": successful_terminal.mean().detach(),
        "failure_terminal_progress": failure_terminal.mean().detach(),
        "success_probability_mean": success_probability.mean().detach(),
    }
    return total, metrics


class VisualProgressEnsemble(nn.Module):
    """Independent visual progress models; disagreement is epistemic uncertainty."""

    def __init__(self, config: VisualProgressConfig, ensemble_size: int = 1) -> None:
        super().__init__()
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.config = config
        self.models = nn.ModuleList(
            [VisualTaskProgressModel(config) for _ in range(int(ensemble_size))]
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.models)

    def predict_batch(
        self,
        clips: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Predict a batch of causal clips with one forward pass per member.

        Args:
            clips: ``(B, context, views, H, W, C)`` clips, or one clip without
                the leading batch dimension.

        Returns:
            Numpy arrays whose leading dimension is ``B``.  Keeping this path
            batched is important when relabelling complete H5 trajectories.
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            tensor = self.models[0].clips_to_tensor(clips, device=device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[0] <= 0:
                raise ValueError(
                    "predict_batch expects one or more clips, got tensor shape "
                    f"{tuple(tensor.shape)}"
                )
            outputs = [model(tensor) for model in self.models]
            progress = torch.stack([output["progress"] for output in outputs], dim=0)
            success = torch.stack(
                [torch.sigmoid(output["success_logit"]) for output in outputs], dim=0
            )
            embedding = torch.stack(
                [output["embedding"] for output in outputs], dim=0
            )
            result = {
                "progress": progress.mean(dim=0).squeeze(-1).cpu().numpy(),
                "progress_uncertainty": progress.std(
                    dim=0, unbiased=False
                ).squeeze(-1).cpu().numpy(),
                "success_probability": success.mean(dim=0).squeeze(-1).cpu().numpy(),
                "success_uncertainty": success.std(
                    dim=0, unbiased=False
                ).squeeze(-1).cpu().numpy(),
                "embedding": embedding.mean(dim=0).cpu().numpy(),
                "embedding_uncertainty": embedding.std(
                    dim=0, unbiased=False
                ).mean(dim=-1).cpu().numpy(),
            }
        self.train(was_training)
        return result

    def predict(
        self,
        clip: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Predict one causal clip, preserving the original scalar API."""
        result = self.predict_batch(clip)
        batch_size = int(np.asarray(result["progress"]).shape[0])
        if batch_size != 1:
            raise ValueError(
                f"predict expects exactly one clip, received batch size {batch_size}; "
                "use predict_batch for batched inference"
            )
        return {
            "progress": float(result["progress"][0]),
            "progress_uncertainty": float(result["progress_uncertainty"][0]),
            "success_probability": float(result["success_probability"][0]),
            "success_uncertainty": float(result["success_uncertainty"][0]),
            "embedding": np.asarray(result["embedding"][0], dtype=np.float32),
            "embedding_uncertainty": float(result["embedding_uncertainty"][0]),
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "ensemble_size": self.ensemble_size,
                "model": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["VisualProgressEnsemble", dict[str, Any]]:
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        ensemble = cls(
            VisualProgressConfig(**payload["config"]),
            ensemble_size=int(payload.get("ensemble_size", 1)),
        ).to(device)
        ensemble.load_state_dict(payload["model"])
        return ensemble, dict(payload.get("metadata", {}))
