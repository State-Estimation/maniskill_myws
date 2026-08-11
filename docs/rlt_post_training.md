# Frozen-Pi0 Residual RL

This repository maintains one implementation family for `SolarPanelStatic-v2`:

1. mean-latent frozen-Pi0 TD3, including the causal exploration/replay variant;
2. its 5-bin temporal-latent continuation.

The temporal checkpoint is the current parent for future experiments. The two
paths use the same Pi0, `pd_joint_pos` action schema, CPU PhysX simulator,
50-step action chunks, and real environment rewards.

## Runtime contract

At each 50-step boundary, Pi0 returns:

```text
reference chunk: 50 x 8 pd_joint_pos actions
mean latent:     1024-D final-denoise action-suffix feature
temporal latent: 5 x 1024 ordered pools (temporal mode only)
```

The residual actor receives the robot state (`qpos`, `qvel`, `tcp_pose`), the
remaining horizon, the Pi0 reference chunk, and the frozen latent. It predicts
six smooth residual knots which are interpolated to a full `50 x 8` residual.
The executed action is always:

```text
clip(reference + normalized_residual * action_scale, action_low, action_high)
```

The actor is zero initialized. The twin critic compares the residual chunk with
the exact Base chunk. Deployment accepts a residual only when the pessimistic
twin-Q advantage is at least `0.10`; otherwise the exact Pi0 chunk is executed.

There is no raw RGB encoder in the residual network, no progress reward, no
stage router, no action candidate rollout, and no policy teacher. The frozen
Pi0 latent is the only visual/action-aligned signal.

The environment is the only reward source:

- stable brush grasp and lift: `0.25`, once per episode;
- task completion: `1.0`, with episode termination.

## Causal TD3

The causal variant keeps the mean latent and TD3 architecture. It changes only
the data-generation loop so the critic sees useful positive and negative local
actions:

- at each eligible boundary, online exploration is independent of Q acceptance;
- exploration probability is `0.35` with smooth normalized noise `0.30`;
- `20%` of episodes are Base-only control episodes;
- replay batches target `50%` nonzero residuals and balanced successful/failed
  nonzero outcomes;
- the actor receives an auxiliary success self-imitation loss on successful
  executed residuals;
- deployment allows up to two interventions with one-chunk cooldown.

This breaks the failure mode where a high-Q but wrong actor proposal is always
accepted and never perturbed. It does not change the reward or action schema.

The preregistered Causal TD3 run reached `84/100` on seeds `32000--32099`
(Base `67/100`, paired gain `+17pp`, 18 rescues, 1 regression). It improved on
the original mean-latent checkpoint but did not meet the `90/100` continuation
gate.

## Temporal latent

The temporal continuation starts from the Causal checkpoint. Let `H` be the
`50 x 1024` final-denoise action-token hidden state from Pi0. The legacy feature
is the global mean:

```text
z_mean = mean(H, axis=0)
```

Temporal mode additionally computes five ordered ten-token pools:

```text
z_bins[k] = mean(H[10*k:10*(k+1)], axis=0)
```

The exact mean branch is retained. Centered temporal differences are passed
through a small `1024 -> 64 -> 256` adapter with a scalar gate initialized to
zero, so the upgraded policy is behavior-exact at initialization. Old mean
replay is migrated by repeating the exact mean across the five bins; only new
online samples contain temporal contrast.

Temporal latent is a feature representation, not five action phases: the
actor still outputs one complete 50-step residual chunk, and interventions are
still made only at chunk boundaries.

The temporal run used an additional `100k` CPU-PhysX environment steps and
reached `91/100` on fixed seeds `35000--35099` (Base `66/100`, 26 rescues, 1
regression). This number includes the Causal training/data improvements, so it
is not an isolated ablation of temporal pooling.

## OpenPI server

For mean-latent training/evaluation:

```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
CUDA_VISIBLE_DEVICES=0 uv run python ../../scripts/pi0/serve.py \
  --config pi0_maniskill \
  --checkpoint ../../checkpoints_openpi/pi0_maniskill/ms_pi0_maniskill_SolarPanelStatic-v20_pd_joint_pos_success_traj_32batch/29999 \
  --port 8011 --xla-safe --frozen-action-latent
```

For temporal-latent training/evaluation, replace the final flag with:

```bash
--frozen-action-temporal-latent
```

The client validates the protocol, latent shape, action horizon, policy
identity, and runtime identity before execution.

## Training commands

The pre-Causal Frozen-Latent TD3 path is reproducible from
`configs/rlt/solarpanel_frozen_latent_td3_reproduction.yaml`. It restores the
original CQ0.1 objective: mean latent, zero residual initialization, uniform
replay, the historical 2% independent epsilon override, no successful-residual
self-imitation, and an exact hard stop at 50,000 environment steps.
Legacy schema-v2 replay is accepted only as a non-exact initial dataset; it is
validated and migrated into the current replay layout without inventing RNG or
snapshot state. The configuration contains the 50k training command and paired
100-seed evaluation command.

The historical reference is Base `65/100` versus Frozen TD3 `80/100` on seeds
`30000--30099` (`+15pp`, 15 rescues, 0 regressions). The later historical 100k
run used a legacy semantic resume that intentionally discarded replay/trainer
RNG state, so it is not presented as an exact-resume reproduction here.

The 2026-07-30 clean rerun completed exactly 50,000 steps and verified the v2
migration metadata, but evaluated at Base `65/100` versus Frozen TD3 `65/100`
(1 rescue, 1 regression, only 8 interventions). Its first three episodes match
the historical training metrics to floating-point precision; the first rollout
length divergence occurs at episode 3 (`177` versus `179` steps), after which
closed-loop CUDA differences compound. Therefore the historical checkpoint is
the reproduced Frozen result, while a fresh stochastic retrain should not be
claimed to deterministically reproduce its `80/100` outcome.

The Causal continuation is defined in
`configs/rlt/solarpanel_goal95_causal_td3_stage50k.yaml`. A typical invocation
is:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/train_frozen_latent_residual.py \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --render-backend sapien_cuda:0 --no-enhanced-determinism \
  --server ws://127.0.0.1:8011 --seed 2000 --device cuda:0 \
  --chunk-len 50 --max-episode-steps 500 --total-env-steps 150000 \
  --resume-checkpoint outputs/rlt/SolarPanelStatic-v2_frozen_latent_td3_cq01_knot_seed2000_resume2x_100k/frozen_latent_residual.pt \
  --resume-replay outputs/rlt/SolarPanelStatic-v2_frozen_latent_td3_cq01_knot_seed2000_resume2x_100k/online_replay.npz \
  --temporal-latent-bins 1 \
  --independent-online-exploration --online-explore-probability 0.35 \
  --base-control-episode-probability 0.20 \
  --replay-nonzero-fraction 0.50 --replay-nonzero-success-fraction 0.50 \
  --updates-per-chunk 2 --conservative-q-weight 0.10 \
  --max-online-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 --min-q-advantage 0.10 \
  --wandb-enabled --wandb-new-run --wandb-project maniskill-myws-rlt \
  --output-dir outputs/rlt/solarpanel_causal_retrain
```

The Temporal continuation is defined in
`configs/rlt/solarpanel_goal95_temporal_latent_stage100k.yaml` and uses:

```text
--upgrade-mean-checkpoint <causal mean checkpoint>
--upgrade-mean-replay <causal mean replay>
--temporal-latent-bins 5
--temporal-adapter-dim 64
```

Both runs require online W&B logging for any new experiment. The trainer keeps
the actor locked until critic warmup and minimum success/failure/nonzero replay
coverage are present.

## Paired evaluation

Standard paired evaluation uses CPU PhysX and the same OpenPI server:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/eval_frozen_latent_residual.py \
  --checkpoint outputs/rlt/SolarPanelStatic-v2_goal95_temporal_latent_seed5000_100k/frozen_latent_residual.pt \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --render-backend sapien_cuda:0 --no-enhanced-determinism \
  --server ws://127.0.0.1:8011 \
  --device cuda:0 --start-seed 35000 --num-seeds 100 \
  --min-q-advantage 0.10 --max-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 \
  --wandb-enabled --output-dir outputs/rlt/temporal_eval_35000
```

For the live SAPIEN viewer and simultaneous Base/RL TCP traces, add:

```text
--render-mode human --live-paired-trajectories --real-time
```

The live overlay draws Base in blue and residual RL in orange. It performs no
Base prepass; before the first intervention, the two same-seed environments are
checked step by step for identical state and Pi0 chunks.

## Maintained files

- `scripts/rlt/train_frozen_latent_residual.py`
- `scripts/rlt/eval_frozen_latent_residual.py`
- `src/maniskill_myws/rlt/frozen_latent_rl.py`
- `src/maniskill_myws/rlt/policies.py`
- `src/maniskill_myws/rlt/backend.py`, `reset.py`, `state.py`
- `configs/rlt/solarpanel_goal95_causal_td3_stage50k.yaml`
- `configs/rlt/solarpanel_frozen_latent_td3_reproduction.yaml`
- `configs/rlt/solarpanel_goal95_temporal_latent_stage100k.yaml`

The parent checkpoint paths and results needed to reproduce each maintained
path are recorded in these configuration files.
