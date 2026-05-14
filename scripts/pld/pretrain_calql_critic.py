#!/usr/bin/env python
"""Stage 1A: Cal-QL pretrain the residual SAC critics from offline replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maniskill_myws.pld import train_common as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="OpenSafeDoor-v2")
    common.add_offline_replay_args(parser)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", "--offline-pretrain-updates", dest="updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=1_000)

    common.add_sac_model_args(parser)
    parser.add_argument("--calql-alpha", type=float, default=0.5)
    parser.add_argument("--calql-n-actions", type=int, default=1)
    parser.add_argument("--calql-temp", type=float, default=1.0)
    parser.add_argument("--calql-no-importance-sample", action="store_true")
    parser.add_argument("--calql-backup-entropy", action="store_true")
    parser.add_argument(
        "--visual-encoder-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional pretrained ResNetV1-10/visual encoder checkpoint. "
            "Loaded into actor and critics before Cal-QL."
        ),
    )

    parser.add_argument("--output-dir", type=str, default="outputs/pld/OpenSafeDoor-v2")
    parser.add_argument("--checkpoint-name", type=str, default="calql_critic.pt")
    common.add_wandb_args(parser)
    args = parser.parse_args()

    common.prepare_runtime(args.seed, register_envs=False)
    import torch

    from maniskill_myws.pld.sac import ResidualSAC, SACConfig

    offline_data, h5_files, _ = common.load_offline_replay(args)
    offline_buffer = common.make_offline_buffer(offline_data, batch_size=args.batch_size)

    device, cuda_info = common.resolve_torch_device(args.device)
    print("devices:", dict(rl_device=str(device), **cuda_info), flush=True)

    action_dim = offline_data.action_dim
    low = (-1.0,) * action_dim
    high = (1.0,) * action_dim
    image_shape = offline_data.image_shape if args.use_visual_rl else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.checkpoint_name

    wandb_run = common.init_wandb(
        args,
        output_dir=output_dir,
        offline_files=len(h5_files),
        offline_transitions=offline_data.size,
        state_dim=offline_data.state_dim,
        action_dim=action_dim,
        action_low=low,
        action_high=high,
    )
    common.wandb_log(
        wandb_run,
        dict(
            offline_update=0,
            **{
                "offline/files": len(h5_files),
                "offline/transitions": offline_data.size,
                "offline/state_dim": offline_data.state_dim,
                "offline/action_dim": action_dim,
                "offline/mc_return_min": float(np.min(offline_data.mc_returns)),
                "offline/mc_return_max": float(np.max(offline_data.mc_returns)),
            },
        ),
    )

    agent = ResidualSAC(
        SACConfig(
            state_dim=offline_data.state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            actor_lr=args.lr,
            critic_lr=args.lr,
            alpha_lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            action_scale=args.action_scale,
            target_entropy=args.target_entropy,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            grad_clip_norm=args.grad_clip_norm,
            calql_alpha=args.calql_alpha,
            calql_n_actions=args.calql_n_actions,
            calql_temp=args.calql_temp,
            calql_importance_sample=not args.calql_no_importance_sample,
            calql_backup_entropy=args.calql_backup_entropy,
            visual_encoder="resnet10" if args.use_visual_rl else "none",
            image_shape=image_shape,
            visual_latent_dim=args.visual_latent_dim,
            action_low=low,
            action_high=high,
        ),
        device=device,
    )
    if args.visual_encoder_checkpoint:
        loaded = agent.load_visual_encoder(args.visual_encoder_checkpoint)
        print(
            "loaded visual encoder checkpoint:",
            dict(path=args.visual_encoder_checkpoint, tensors=loaded),
            flush=True,
        )

    metrics: dict[str, float] = {}
    for i in range(max(0, int(args.updates))):
        batch = offline_buffer.sample(args.batch_size)
        metrics = agent.pretrain_critic_calql(batch)
        if (i + 1) % max(1, args.log_every) == 0 or (i + 1) == args.updates:
            print("offline_calql_update", i + 1, metrics)
            common.wandb_log(
                wandb_run,
                dict(
                    offline_update=i + 1,
                    **{
                        "offline/pretrain_method": "calql",
                        **{f"offline/{k}": float(v) for k, v in metrics.items()},
                    },
                ),
            )

    agent.save(ckpt_path)
    common.wandb_log(
        wandb_run,
        dict(env_step=0, **{"checkpoint/offline_pretrained": 1.0}),
    )
    print("saved Cal-QL critic checkpoint:", ckpt_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
