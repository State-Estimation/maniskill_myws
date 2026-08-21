#!/usr/bin/env python
"""Train the official ActProbe network on ManiSkill action features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ACTPROBE_ROOT = ROOT / "third_party" / "actprobe"
ACTPROBE_PI0_CODE = ACTPROBE_ROOT / "benchmarks" / "pi0_libero" / "code"
EXPECTED_ACTPROBE_COMMIT = "d5dfbcc98ee5e5766f8aa548c657d3446e40e272"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ACTPROBE_ROOT))
sys.path.insert(0, str(ACTPROBE_PI0_CODE))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seen-train-ratio", type=float, default=0.75)
    parser.add_argument("--unseen-episode-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--timestamp-scale", type=float, default=100.0)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.epochs,
        args.patience,
        args.eval_every,
        args.timestamp_scale,
    ) <= 0:
        parser.error("training counts and timestamp scale must be positive")
    if not args.seeds or min(args.seeds) < 0:
        parser.error("at least one non-negative seed is required")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be unique")
    return args


def _require_official_actprobe_checkout() -> None:
    model_path = ACTPROBE_PI0_CODE / "lib" / "methods" / "actprobe.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Official ActProbe checkout not found at {ACTPROBE_ROOT}"
        )
    commit = subprocess.run(
        ["git", "-C", str(ACTPROBE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_ACTPROBE_COMMIT:
        raise RuntimeError(
            f"Official ActProbe commit {commit} != supported {EXPECTED_ACTPROBE_COMMIT}"
        )


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_episode(episode: Any, *, task_id: str) -> dict[str, Any]:
    return {
        "raw": episode.features,
        "label": 0 if episode.success else 1,
        "task_id": task_id,
        "length": len(episode.features),
        "episode_id": episode.seed,
    }


def main() -> None:
    args = _parse_args()
    _require_official_actprobe_checkout()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from lib.methods.actprobe import ActProbeNet, LANG_DIM, N_FEAT
    from lib_shared.metrics import q_auc
    from maniskill_myws.rlt.actprobe import (
        ACTPROBE_FEATURE_NAMES,
        load_actprobe_dataset,
        stratified_actprobe_split_indices,
    )

    if N_FEAT != len(ACTPROBE_FEATURE_NAMES):
        raise RuntimeError("Official ActProbe feature dimension is not two")
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"ActProbe dataset not found: {dataset_path}")
    dataset = load_actprobe_dataset(dataset_path)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    task_id = str(dataset.metadata.get("env_id", "maniskill_task"))
    task_embedding = np.zeros((LANG_DIM,), dtype=np.float32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but PyTorch CUDA is unavailable")

    class EpisodeDataset(Dataset):
        def __init__(self, episodes, mean, std, max_len):
            self.episodes = list(episodes)
            self.mean = mean
            self.std = std
            self.max_len = max_len

        def __len__(self):
            return len(self.episodes)

        def __getitem__(self, index):
            episode = self.episodes[index]
            raw = episode["raw"]
            length = len(raw)
            features = (raw - self.mean) / (self.std + 1e-7)
            timestamp = (
                np.arange(length, dtype=np.float32) / args.timestamp_scale
            ).reshape(-1, 1)
            model_input = np.hstack([features, timestamp]).astype(np.float32)
            if length < self.max_len:
                model_input = np.vstack(
                    [
                        model_input,
                        np.zeros(
                            (self.max_len - length, model_input.shape[1]),
                            dtype=np.float32,
                        ),
                    ]
                )
            return (
                torch.from_numpy(model_input),
                torch.from_numpy(task_embedding),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(float(episode["label"]), dtype=torch.float32),
                episode["task_id"],
            )

    def collate(batch):
        inputs, languages, lengths, labels, task_ids = zip(*batch)
        return (
            torch.stack(inputs),
            torch.stack(languages),
            torch.stack(lengths),
            torch.stack(labels),
            list(task_ids),
        )

    def validation_q_auc(model, loader) -> float:
        model.eval()
        results = []
        with torch.no_grad():
            for inputs, languages, lengths, labels, task_ids in loader:
                inputs = inputs.to(device)
                languages = languages.to(device)
                lengths = lengths.to(device)
                scores = model(inputs, languages, lengths)
                for index, (length, label, result_task_id) in enumerate(
                    zip(lengths.tolist(), labels.tolist(), task_ids, strict=True)
                ):
                    results.append(
                        {
                            "scores": scores[index, :length].cpu().numpy(),
                            "length": length,
                            "label": int(label),
                            "task_id": result_task_id,
                        }
                    )
        if len({result["label"] for result in results}) < 2:
            return 0.5
        return float(q_auc(results, mode="taskmax", q=0.25))

    dataset_digest = _dataset_sha256(dataset_path)
    for seed in args.seeds:
        checkpoint_dir = output_root / f"seed{seed}"
        checkpoint_path = checkpoint_dir / "actprobe.pt"
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite ActProbe checkpoint: {checkpoint_path}"
            )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        split_indices = stratified_actprobe_split_indices(
            [episode.success for episode in dataset.episodes],
            seen_train_ratio=args.seen_train_ratio,
            unseen_episode_ratio=args.unseen_episode_ratio,
            seed=seed,
        )
        split_episodes = {
            split: [
                _model_episode(dataset.episodes[index], task_id=task_id)
                for index in indices
            ]
            for split, indices in split_indices.items()
        }
        train_episodes = split_episodes["train"]
        validation_episodes = split_episodes["val_seen"]
        train_features = np.concatenate(
            [episode["raw"] for episode in train_episodes], axis=0
        )
        norm_mean = train_features.mean(axis=0).astype(np.float32)
        norm_std = (train_features.std(axis=0) + 1e-8).astype(np.float32)
        max_len = max(
            episode["length"]
            for episode in train_episodes + validation_episodes
        )
        train_loader = DataLoader(
            EpisodeDataset(train_episodes, norm_mean, norm_std, max_len),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        validation_loader = DataLoader(
            EpisodeDataset(validation_episodes, norm_mean, norm_std, max_len),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )

        model = ActProbeNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        best_auc = float("-inf")
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        final_loss = float("nan")
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for inputs, languages, lengths, labels, _ in train_loader:
                inputs = inputs.to(device)
                languages = languages.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                scores = model(inputs, languages, lengths)
                mask = (
                    torch.arange(scores.shape[1], device=device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                targets = labels.unsqueeze(1).expand_as(scores)
                loss = (
                    nn.functional.binary_cross_entropy(
                        scores.clamp(1e-7, 1 - 1e-7),
                        targets,
                        reduction="none",
                    )
                    * mask
                ).sum() / mask.float().sum()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            final_loss = float(np.mean(losses))

            if epoch % args.eval_every != 0:
                continue
            auc = validation_q_auc(model, validation_loader)
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.eval_every
            if epochs_without_improvement >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "schema": "maniskill_official_actprobe_checkpoint_v1",
            "official_actprobe_commit": EXPECTED_ACTPROBE_COMMIT,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_digest,
            "dataset_metadata": dataset.metadata,
            "feature_names": list(ACTPROBE_FEATURE_NAMES),
            "task_embedding": "constant_zero_1024_single_task",
            "task_id": task_id,
            "split_protocol": "safe_stratified_episode_60_20_20_v1",
            "seen_train_ratio": args.seen_train_ratio,
            "unseen_episode_ratio": args.unseen_episode_ratio,
            "split_indices": split_indices,
            "seed": seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "eval_every": args.eval_every,
            "timestamp_scale": args.timestamp_scale,
            "best_epoch": best_epoch,
            "best_val_q025_taskmax_auc": best_auc,
            "final_train_loss": final_loss,
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "norm_mean": norm_mean.tolist(),
                "norm_std": norm_std.tolist(),
                "arch_variant": "full",
                "feat_indices": [0, 1],
                "config": config,
            },
            checkpoint_path,
        )
        (checkpoint_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "stage": "actprobe.training_complete",
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "best_epoch": best_epoch,
                    "best_val_q025_taskmax_auc": best_auc,
                    "split_sizes": {
                        split: len(indices)
                        for split, indices in split_indices.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
