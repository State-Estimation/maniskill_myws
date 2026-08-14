# Mean Frozen-Latent Advantage-BC

This branch maintains one RL post-training method. Pi0 is frozen and exposes
the mean of its final-denoise action-suffix hidden tokens. A lightweight TD3
actor proposes a smooth six-knot residual over each 50-step Pi0 action chunk.
A conservative twin-Q advantage gate decides whether to execute the residual.

Successful executed residuals are used for actor self-imitation only when
their conservative advantage over the zero residual exceeds the configured
threshold. Replay sampling balances zero residuals, successful nonzero
residuals, and failed nonzero residuals.

The method has no independent vision encoder, temporal latent, outcome or
failure detector, candidate ensemble, PCA branch, or task-stage router.

## OpenPI server

```bash
cd third_party/openpi
CUDA_VISIBLE_DEVICES=0 uv run python ../../scripts/pi0/serve.py \
  --config pi0_maniskill \
  --checkpoint ../../checkpoints_openpi/pi0_maniskill/ms_pi0_maniskill_SolarPanelStatic-v20_pd_joint_pos_success_traj_32batch/29999 \
  --port 8011 --xla-safe --frozen-action-latent
```

## Fresh training

Fresh training must not pass any resume arguments.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/rlt/train_frozen_latent_residual.py \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --render-backend sapien_cuda:0 --enhanced-determinism \
  --server ws://127.0.0.1:8011 --device cuda:0 --seed 15000 \
  --chunk-len 50 --max-episode-steps 500 --total-env-steps 200000 \
  --buffer-capacity 50000 --batch-size 128 --updates-per-chunk 2 \
  --critic-warmup-updates 2000 --online-explore-probability 0.35 \
  --independent-online-exploration --exploration-std-start 0.30 \
  --exploration-std-end 0.30 --base-control-episode-probability 0.20 \
  --replay-nonzero-fraction 0.50 --replay-nonzero-success-fraction 0.50 \
  --actor-success-bc-weight 2.0 \
  --actor-success-bc-min-q-advantage 0.05 \
  --conservative-q-weight 0.10 --min-q-advantage 0.08 \
  --max-online-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 --wandb-enabled --wandb-new-run \
  --wandb-project maniskill-frozen-latent-td3 \
  --output-dir outputs/rlt/SolarPanelStatic-v2_frozen_latent_advantage_bc
```

## Exact continuation

Continuation requires checkpoint, replay, history, and trainer state from the
same snapshot generation. Set `--total-env-steps` to the new cumulative target
and use a new output directory.

```bash
python -u scripts/rlt/train_frozen_latent_residual.py [same training flags] \
  --total-env-steps 250000 \
  --resume-checkpoint <run>/frozen_latent_residual.pt \
  --resume-replay <run>/online_replay.npz \
  --resume-history <run>/history.jsonl \
  --resume-trainer-state <run>/trainer_state.json \
  --output-dir <continued-run>
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/rlt/eval_frozen_latent_residual.py \
  --checkpoint <run>/frozen_latent_residual.pt \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --render-backend sapien_cuda:0 --enhanced-determinism \
  --server ws://127.0.0.1:8011 --device cuda:0 \
  --start-seed 69000 --num-seeds 100 --min-q-advantage 0.08 \
  --max-intervention-chunks-per-episode 2 --intervention-cooldown-chunks 1 \
  --output-dir <run>/eval
```

Validated independent runs reached 87/100 and 90/100 paired RL success from
fresh RL initialization. The frozen Pi0 checkpoint is the only pretrained
component.
