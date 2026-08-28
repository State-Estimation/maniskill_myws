#!/usr/bin/env python3
"""Train and calibrate the lightweight RECAP-style distributional V_base."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maniskill_myws.openpi_bridge.remote_policy import SAFE_LATENT_DIM  # noqa: E402
from maniskill_myws.rlt.value_dataset import (  # noqa: E402
    ValueBoundaryDataset,
    read_value_dataset_metadata,
    scan_value_episodes,
    stratified_value_split,
)
from maniskill_myws.rlt.value_model import (  # noqa: E402
    DistributionalBaseValueModel,
    DistributionalValueConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--modality-dim", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--failure-value", type=float, default=-1.25)
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="maniskill-recap-value")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.modality_dim,
        args.feature_dim,
        args.hidden_dim,
    ) <= 0:
        parser.error("training and model dimensions must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must lie in (0,1)")
    if not 0.0 < args.failure_threshold < 1.0:
        parser.error("--failure-threshold must lie in (0,1)")
    if min(args.weight_decay, args.grad_clip, args.num_workers) < 0:
        parser.error("weight decay, grad clip, and workers must be non-negative")
    return args


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target.astype(bool)
    positive_count = int(np.count_nonzero(positive))
    negative_count = len(target) - positive_count
    if min(positive_count, negative_count) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index + 1 < bins
            else probability <= edges[index + 1]
        )
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(target[selected]))
            )
    return result if total else float("nan")


@torch.no_grad()
def _evaluate(
    model: DistributionalBaseValueModel,
    loader: DataLoader,
    *,
    device: torch.device,
    failure_threshold: float,
    episodes,
) -> dict[str, float]:
    model.eval()
    losses: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    failure_probabilities: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    boundary_indices: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(
            batch["images"],
            batch["state"],
            batch["latent"],
            batch["ref_chunk"],
            batch["step_id"],
        )
        loss = F.cross_entropy(output["logits"], batch["target"], reduction="none")
        losses.append(loss.cpu().numpy())
        predictions.append(output["logits"].argmax(dim=-1).cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        failure_probabilities.append(output["failure_probability"].cpu().numpy())
        episode_indices.append(batch["episode_index"].cpu().numpy())
        boundary_indices.append(batch["boundary_index"].cpu().numpy())
    loss_array = np.concatenate(losses)
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    failure_probability = np.concatenate(failure_probabilities)
    episode_index = np.concatenate(episode_indices)
    boundary_index = np.concatenate(boundary_indices)
    failure_target = target == 0
    decision = failure_probability >= failure_threshold
    tp = int(np.count_nonzero(decision & failure_target))
    fn = int(np.count_nonzero(~decision & failure_target))
    fp = int(np.count_nonzero(decision & ~failure_target))
    tn = int(np.count_nonzero(~decision & ~failure_target))
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for ep, boundary, probability in zip(
        episode_index, boundary_index, failure_probability, strict=True
    ):
        grouped[int(ep)].append((int(boundary), float(probability)))
    detected_failures = 0
    false_trigger_successes = 0
    lead_chunks: list[int] = []
    for ep_index, values in grouped.items():
        values.sort()
        first = next(
            (boundary for boundary, probability in values if probability >= failure_threshold),
            None,
        )
        episode = episodes[ep_index]
        if episode.success:
            false_trigger_successes += int(first is not None)
        elif first is not None:
            detected_failures += 1
            lead_chunks.append(episode.boundaries - first)
    failed_episode_count = sum(not episode.success for episode in episodes)
    successful_episode_count = sum(episode.success for episode in episodes)
    return {
        "nll": float(np.mean(loss_array)),
        "bin_accuracy": float(np.mean(prediction == target)),
        "failure_auc": _binary_auc(failure_target, failure_probability),
        "failure_brier": float(
            np.mean((failure_probability - failure_target.astype(np.float32)) ** 2)
        ),
        "failure_ece": _ece(failure_target, failure_probability),
        "boundary_failure_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "boundary_failure_fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "episode_failure_recall": (
            detected_failures / failed_episode_count if failed_episode_count else float("nan")
        ),
        "successful_episode_false_trigger_rate": (
            false_trigger_successes / successful_episode_count
            if successful_episode_count
            else float("nan")
        ),
        "mean_detection_lead_chunks": (
            float(np.mean(lead_chunks)) if lead_chunks else float("nan")
        ),
    }


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix a value run with {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    metadata = read_value_dataset_metadata(dataset_path)
    episodes = scan_value_episodes(dataset_path)
    train_episodes, validation_episodes = stratified_value_split(
        episodes,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    state_dim = int(metadata.get("state_dim", 0))
    if state_dim <= 0:
        import h5py

        with h5py.File(dataset_path, "r") as file:
            state_dim = int(file[episodes[0].name]["states"].shape[1])
    config = DistributionalValueConfig(
        state_dim=state_dim,
        action_dim=int(metadata["action_dim"]),
        chunk_len=int(metadata["chunk_len"]),
        max_episode_steps=int(metadata["max_episode_steps"]),
        num_views=2,
        image_height=int(metadata["value_image_size"]),
        image_width=int(metadata["value_image_size"]),
        latent_dim=SAFE_LATENT_DIM,
        modality_dim=args.modality_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        failure_value=args.failure_value,
    )
    train_dataset = ValueBoundaryDataset(
        dataset_path,
        train_episodes,
        max_remaining_chunks=config.max_remaining_chunks,
    )
    validation_dataset = ValueBoundaryDataset(
        dataset_path,
        validation_episodes,
        max_remaining_chunks=config.max_remaining_chunks,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = DistributionalBaseValueModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_metadata: dict[str, Any] = {
        "dataset": str(dataset_path.resolve()),
        "dataset_metadata": metadata,
        "args": vars(args),
        "train_seeds": [episode.seed for episode in train_episodes],
        "validation_seeds": [episode.seed for episode in validation_episodes],
        "target": "categorical_monte_carlo_failure_plus_remaining_chunks",
        "critic_feature_schema": "feature_plus_distribution_value_failure_entropy",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    wandb_run = None
    if args.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or output_dir.name,
            config={"model": config.__dict__, **vars(args)},
        )
    best_nll = math.inf
    history_path = output_dir / "history.jsonl"
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_items = 0
            for raw_batch in train_loader:
                batch = _to_device(raw_batch, device)
                output = model(
                    batch["images"],
                    batch["state"],
                    batch["latent"],
                    batch["ref_chunk"],
                    batch["step_id"],
                )
                loss = F.cross_entropy(output["logits"], batch["target"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                train_loss += float(loss.detach().cpu()) * len(batch["target"])
                train_items += len(batch["target"])
            metrics = _evaluate(
                model,
                validation_loader,
                device=device,
                failure_threshold=args.failure_threshold,
                episodes=validation_episodes,
            )
            record = {
                "epoch": epoch,
                "train_nll": train_loss / train_items,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
            print(json.dumps(record, sort_keys=True, allow_nan=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(record, step=epoch)
            model.save(
                output_dir / "value_last.pt",
                metadata={**run_metadata, "epoch": epoch, "validation": metrics},
            )
            if metrics["nll"] < best_nll:
                best_nll = metrics["nll"]
                model.save(
                    output_dir / "value_best.pt",
                    metadata={**run_metadata, "epoch": epoch, "validation": metrics},
                )
    finally:
        train_dataset.close()
        validation_dataset.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
