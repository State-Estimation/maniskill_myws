#!/usr/bin/env python
"""Train a vision-only full-task progress model without frame/stage labels."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_KEYS = [
    "sensor_data/base_camera/rgb",
    "sensor_data/hand_camera/rgb",
]


def _validate_metadata(
    files: list[Path], *, expected_env_id: str, expected_control_mode: str
) -> None:
    for h5_path in files:
        metadata_path = h5_path.with_suffix(".json")
        if not metadata_path.exists():
            raise ValueError(f"Missing replay metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        env_info = metadata.get("env_info", {})
        actual_env_id = env_info.get("env_id")
        actual_mode = env_info.get("env_kwargs", {}).get("control_mode")
        if actual_env_id != expected_env_id:
            raise ValueError(
                f"Replay env_id mismatch in {metadata_path}: "
                f"{actual_env_id!r} != {expected_env_id!r}"
            )
        if actual_mode != expected_control_mode:
            raise ValueError(
                f"Replay control_mode mismatch in {metadata_path}: "
                f"{actual_mode!r} != {expected_control_mode!r}"
            )


def _to_device(raw: Any, device):
    import torch

    if torch.is_tensor(raw):
        return raw.to(device, non_blocking=True)
    return torch.as_tensor(raw, device=device)


def _augment_clips(clips, *, strength: float):
    import torch

    if strength <= 0:
        return clips
    # One photometric transform per clip; apply it consistently to every
    # concatenated context/view channel so motion cues are not corrupted.
    leading = clips.shape[:-3]
    shape = (*leading, 1, 1, 1)
    brightness = 1.0 + (2.0 * torch.rand(shape, device=clips.device) - 1.0) * strength
    contrast = 1.0 + (2.0 * torch.rand(shape, device=clips.device) - 1.0) * strength
    mean = clips.mean(dim=(-2, -1), keepdim=True)
    augmented = (clips - mean) * contrast + mean
    augmented = augmented * brightness
    if strength > 0:
        augmented = augmented + torch.randn_like(augmented) * (0.02 * strength)
    return augmented.clamp_(0.0, 1.0)


def _loss_kwargs(args) -> dict[str, float]:
    return {
        "cycle_temperature": args.cycle_temperature,
        "order_margin": args.order_margin,
        "preference_margin": args.preference_margin,
        "cycle_weight": args.cycle_weight,
        "alignment_weight": args.alignment_weight,
        "endpoint_weight": args.endpoint_weight,
        "order_weight": args.order_weight,
        "latent_smoothness_weight": args.latent_smoothness_weight,
        "smoothness_weight": args.smoothness_weight,
        "success_weight": args.success_weight,
        "preference_weight": args.preference_weight,
    }


def _average_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for metrics in metric_rows:
        for key, value in metrics.items():
            scalar = float(value.detach().float().cpu()) if hasattr(value, "detach") else float(value)
            totals[key] = totals.get(key, 0.0) + scalar
    denominator = max(1, len(metric_rows))
    return {key: value / denominator for key, value in totals.items()}


def _forward_model(model, raw_batch, device, *, augment_strength: float):
    tensor_a = model.clips_to_tensor(raw_batch["success_a_images"], device=device)
    tensor_b = model.clips_to_tensor(raw_batch["success_b_images"], device=device)
    tensor_f = model.clips_to_tensor(raw_batch["failure_images"], device=device)
    if augment_strength > 0:
        tensor_a = _augment_clips(tensor_a, strength=augment_strength)
        tensor_b = _augment_clips(tensor_b, strength=augment_strength)
        tensor_f = _augment_clips(tensor_f, strength=augment_strength)
    return model(tensor_a), model(tensor_b), model(tensor_f)


def _evaluate(ensemble, loader, device, args) -> dict[str, float]:
    import torch

    from maniskill_myws.pld.visual_progress import visual_progress_training_loss

    ensemble.eval()
    rows: list[dict[str, Any]] = []
    disagreement_progress: list[float] = []
    disagreement_success: list[float] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_validation_batches and batch_index >= args.max_validation_batches:
                break
            member_outputs = []
            for model in ensemble.models:
                output_a, output_b, output_f = _forward_model(
                    model,
                    raw_batch,
                    device,
                    augment_strength=0.0,
                )
                _, metrics = visual_progress_training_loss(
                    output_a,
                    output_b,
                    output_f,
                    **_loss_kwargs(args),
                )
                rows.append(metrics)
                member_outputs.append((output_a, output_b, output_f))
            if len(member_outputs) > 1:
                progress = torch.stack(
                    [
                        torch.cat(
                            [row[0]["progress"], row[1]["progress"], row[2]["progress"]],
                            dim=1,
                        )
                        for row in member_outputs
                    ],
                    dim=0,
                )
                success = torch.stack(
                    [
                        torch.sigmoid(
                            torch.cat(
                                [
                                    row[0]["success_logit"],
                                    row[1]["success_logit"],
                                    row[2]["success_logit"],
                                ],
                                dim=1,
                            )
                        )
                        for row in member_outputs
                    ],
                    dim=0,
                )
                disagreement_progress.append(
                    float(progress.std(dim=0, unbiased=False).mean().cpu())
                )
                disagreement_success.append(
                    float(success.std(dim=0, unbiased=False).mean().cpu())
                )
    metrics = _average_metrics(rows)
    metrics["ensemble_progress_disagreement"] = float(
        np.mean(disagreement_progress) if disagreement_progress else 0.0
    )
    metrics["ensemble_success_disagreement"] = float(
        np.mean(disagreement_success) if disagreement_success else 0.0
    )
    ensemble.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="SolarPanelStatic-v2")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument(
        "--offline-h5-dir",
        type=str,
        default="dataset/Pi0_rollout_SolarPanelStatic_pd_joint_pos",
    )
    parser.add_argument("--offline-h5-glob", type=str, default=None)
    parser.add_argument("--image-keys", type=str, nargs="+", default=DEFAULT_IMAGE_KEYS)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--context-stride", type=int, default=2)
    parser.add_argument("--sequence-points", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-samples", type=int, default=64)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--visual-latent-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--augmentation-strength", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-validation-batches", type=int, default=0)

    parser.add_argument("--cycle-temperature", type=float, default=0.1)
    parser.add_argument("--order-margin", type=float, default=0.02)
    parser.add_argument("--preference-margin", type=float, default=0.2)
    parser.add_argument("--cycle-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-weight", type=float, default=2.0)
    parser.add_argument("--order-weight", type=float, default=1.0)
    parser.add_argument("--latent-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--preference-weight", type=float, default=0.5)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pld/SolarPanelStatic-v2/visual_progress_selfsupervised",
    )
    parser.add_argument("--checkpoint-name", type=str, default="visual_progress_best.pt")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="maniskill-pld")
    parser.add_argument("--wandb-name", type=str, default="solar-visual-progress-selfsup")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    args = parser.parse_args()

    if args.image_size <= 0 or args.sequence_points < 3:
        raise SystemExit("--image-size must be positive and --sequence-points >= 3")
    if args.batch_size <= 0 or args.updates <= 0 or args.ensemble_size <= 0:
        raise SystemExit("--batch-size, --updates and --ensemble-size must be positive")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from maniskill_myws.pld import train_common as common

    torch = common.prepare_runtime(args.seed, register_envs=False)
    from torch.utils.data import DataLoader

    from maniskill_myws.pld.h5_replay import find_h5_files
    from maniskill_myws.pld.visual_progress import (
        VisualProgressConfig,
        VisualProgressEnsemble,
        visual_progress_training_loss,
    )
    from maniskill_myws.pld.visual_progress_dataset import (
        H5VisualProgressPairDataset,
        scan_visual_episodes,
        split_visual_episodes,
    )

    h5_dir = (
        str((REPO_ROOT / args.offline_h5_dir).resolve())
        if args.offline_h5_dir
        else None
    )
    files = find_h5_files(h5_dir=h5_dir, h5_glob=args.offline_h5_glob)
    if not files:
        raise SystemExit(f"No H5 files found under {h5_dir or args.offline_h5_glob}")
    _validate_metadata(
        files,
        expected_env_id=args.env_id,
        expected_control_mode=args.control_mode,
    )
    try:
        episodes = scan_visual_episodes(files)
    except BlockingIOError as e:
        raise SystemExit(
            "Offline H5 is locked by another writer; wait for rollout collection to finish."
        ) from e
    train_episodes, validation_episodes = split_visual_episodes(
        episodes,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not validation_episodes:
        raise SystemExit("A held-out episode split is required for visual progress training")

    dataset_kwargs = dict(
        image_keys=args.image_keys,
        image_size=args.image_size,
        context_frames=args.context_frames,
        context_stride=args.context_stride,
        sequence_points=args.sequence_points,
    )
    train_dataset = H5VisualProgressPairDataset(
        files,
        train_episodes,
        samples_per_epoch=args.train_samples_per_epoch,
        seed=args.seed,
        deterministic=False,
        **dataset_kwargs,
    )
    validation_dataset = H5VisualProgressPairDataset(
        files,
        validation_episodes,
        samples_per_epoch=args.validation_samples,
        seed=args.seed + 10_000,
        deterministic=True,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device, cuda_info = common.resolve_torch_device(args.device)
    config = VisualProgressConfig(
        context_frames=args.context_frames,
        num_views=len(args.image_keys),
        image_height=args.image_size,
        image_width=args.image_size,
        visual_latent_dim=args.visual_latent_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    ensemble = VisualProgressEnsemble(config, args.ensemble_size).to(device)
    optimizers = [
        torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for model in ensemble.models
    ]

    print(
        "visual_progress_dataset",
        dict(
            files=len(files),
            train_episodes=len(train_episodes),
            validation_episodes=len(validation_episodes),
            successful_episodes=sum(episode.success for episode in episodes),
            failed_episodes=sum(not episode.success for episode in episodes),
            image_keys=args.image_keys,
            image_size=args.image_size,
            context_frames=args.context_frames,
            context_stride=args.context_stride,
            sequence_points=args.sequence_points,
            ensemble_size=args.ensemble_size,
            visual_only=True,
            forbidden_state_inputs=[
                "qpos",
                "qvel",
                "tcp_pose",
                "brush_pose",
                "clean_coverage",
                "cleaning_contact",
            ],
            device=str(device),
            **cuda_info,
        ),
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / args.checkpoint_name
    latest_path = output_dir / "visual_progress_latest.pt"
    last_path = output_dir / "visual_progress_last.pt"
    metadata = {
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "image_keys": list(args.image_keys),
        "image_size": args.image_size,
        "context_frames": args.context_frames,
        "context_stride": args.context_stride,
        "phase_representation": "implicit_cycle_consistent_embedding",
        "visual_only": True,
        "dense_labels_used": False,
        "stage_annotations_used": False,
        "episode_outcome_used": True,
        "offline_files": [str(path) for path in files],
    }

    wandb_run = None
    if args.wandb_enabled and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as e:
            raise SystemExit("wandb is not installed; omit --wandb-enabled") from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output_dir),
            config={key: value for key, value in vars(args).items()},
        )

    ensemble.train()
    iterator = iter(train_loader)
    best_validation = float("inf")
    use_amp = bool(args.amp and device.type == "cuda")
    for update in range(1, args.updates + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)

        member_metrics: list[dict[str, Any]] = []
        for member_index, (model, optimizer) in enumerate(
            zip(ensemble.models, optimizers, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                output_a, output_b, output_f = _forward_model(
                    model,
                    raw_batch,
                    device,
                    augment_strength=args.augmentation_strength,
                )
                loss, metrics = visual_progress_training_loss(
                    output_a,
                    output_b,
                    output_f,
                    **_loss_kwargs(args),
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            optimizer.step()
            metrics = dict(metrics)
            metrics["grad_norm"] = grad_norm.detach()
            member_metrics.append(metrics)

        if update % max(1, args.log_every) == 0 or update == 1:
            train_metrics = _average_metrics(member_metrics)
            print("visual_progress_train", dict(update=update, **train_metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_metrics.items()},
                    step=update,
                )

        if update % max(1, args.eval_every) == 0 or update == args.updates:
            validation_metrics = _evaluate(
                ensemble,
                validation_loader,
                device,
                args,
            )
            print(
                "visual_progress_validation",
                dict(update=update, **validation_metrics),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"validation/{key}": value
                        for key, value in validation_metrics.items()
                    },
                    step=update,
                )
            ensemble.save(
                latest_path,
                metadata={
                    **metadata,
                    "update": update,
                    "validation": validation_metrics,
                },
            )
            validation_loss = validation_metrics.get("loss", float("inf"))
            if validation_loss < best_validation:
                best_validation = validation_loss
                ensemble.save(
                    best_path,
                    metadata={
                        **metadata,
                        "best_update": update,
                        "best_validation": validation_metrics,
                    },
                )
                print("saved best visual progress ensemble:", best_path, flush=True)

    ensemble.save(
        last_path,
        metadata={
            **metadata,
            "updates": args.updates,
            "best_validation_loss": best_validation,
        },
    )
    print(
        "visual progress training complete:",
        dict(best=str(best_path), latest=str(latest_path), last=str(last_path)),
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.finish()
    train_dataset.close()
    validation_dataset.close()


if __name__ == "__main__":
    main()
