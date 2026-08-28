# ManiSkill Frozen-Latent V-Gated Advantage-BC

This branch keeps the validated SolarPanelStatic-v2 post-training pipeline:

- Pi0 remains frozen and exposes the full 4096-dimensional SAFE endpoint latent.
- `V_base` is a frozen visual value model trained offline on base-policy rollouts.
- A hysteretic V gate decides when residual RL may intervene.
- The actor explores with persistent 10-step macro-knot residuals and learns only
  from terminally successful gated episodes (Positive Success BC).
- Environment rewards are recorded for audit; no process reward or simulator
  branch is used as an online learning target.

The two retained configurations are:

- `configs/rlt/solarpanel_vgate_positive_success_bc_44900.yaml`: exact seed44900
  from-scratch pilot, 20k environment steps.
- `configs/rlt/solarpanel_vgate_positive_success_bc_noise_curriculum.yaml`:
  exact snapshot resume at 20,495 steps, with noise/background-probe annealing
  to 200k steps.

## Prerequisites

Start the frozen Pi0 service with the SAFE latent protocol on port 8012. The
commands below use CPU PhysX and GPU rendering, so the simulation does not
consume the model GPU.

```bash
cd /home/sisyphus/Projects/maniskill_myws
```

## Train the 44900 pilot

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/train_value_guided_bandit.py \
  --server ws://127.0.0.1:8012 \
  --value-checkpoint outputs/rlt/value/SolarPanelStatic-v2_safe_visual_recap_value_seed41000/value_best.pt \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu --render-backend sapien_cuda:0 \
  --device cuda:0 --chunk-len 10 --max-episode-steps 500 \
  --seed 44900 --total-env-steps 20000 --batch-size 128 --updates-per-chunk 5 \
  --behavior-policy persistent_actor --exploration-noise-mode macro_knots \
  --exploration-knots 6 --exploration-burst-chunks 5 \
  --max-exploration-bursts-per-episode 2 --exploration-burst-cooldown-chunks 5 \
  --candidate-count 12 --num-scorers 5 \
  --candidate-noise-start 0.30 --candidate-noise-end 0.30 \
  --candidate-explore-probability-start 1.0 --candidate-explore-probability-end 0.35 \
  --candidate-explore-anneal-steps 50000 \
  --persistent-noise-correlation 0.95 --persistent-gripper-noise-scale 0.5 \
  --background-probe-episode-probability 0.80 \
  --background-probe-max-chunks-per-episode 5 --background-probe-max-boundary 15 \
  --background-probe-min-value-improvement -0.20 \
  --actor-safety-enabled --actor-safety-min-value-improvement -0.02 \
  --actor-safety-cooldown-chunks 3 --gate-min-active-chunks 1 \
  --gate-immediate-max-entropy 1.5 --warmup-transitions 256 \
  --actor-success-bc-weight 10.0 --actor-success-bc-min-value-improvement 0.01 \
  --actor-l2-weight 0.10 --actor-smoothness-weight 0.20 \
  --actor-value-objective-weight 0.0 --actor-awr-weight 0.0 \
  --output-dir outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_macro_budget100_pilot20k \
  --wandb-enabled --wandb-project maniskill-vgate-advantage-bc \
  --wandb-run-name panel-successbc-positive-seed44900-pilot20k
```

## Continue with noise curriculum

The curriculum must resume the pilot checkpoint, replay, and trainer state as
one snapshot. It holds the initial exploration settings through 20k steps,
anneals noise from 0.30 to 0.10 and background probes from 0.80 to 0.25 over
the next 100k steps, then holds the final values until 200k.

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/train_value_guided_bandit.py \
  --server ws://127.0.0.1:8012 \
  --value-checkpoint outputs/rlt/value/SolarPanelStatic-v2_safe_visual_recap_value_seed41000/value_best.pt \
  --resume-checkpoint outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_macro_budget100_pilot20k/value_guided_bandit.pt \
  --resume-replay outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_macro_budget100_pilot20k/online_bandit_replay.npz \
  --resume-trainer-state outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_macro_budget100_pilot20k/trainer_state.json \
  --env-id SolarPanelStatic-v2 --obs-mode rgb --reward-mode sparse \
  --control-mode pd_joint_pos --sim-backend physx_cpu --render-backend sapien_cuda:0 \
  --device cuda:0 --chunk-len 10 --max-episode-steps 500 \
  --seed 44900 --total-env-steps 200000 --batch-size 128 --updates-per-chunk 5 \
  --behavior-policy persistent_actor --exploration-noise-mode macro_knots \
  --exploration-knots 6 --exploration-burst-chunks 5 \
  --max-exploration-bursts-per-episode 2 --exploration-burst-cooldown-chunks 5 \
  --candidate-count 12 --num-scorers 5 \
  --candidate-noise-start 0.30 --candidate-noise-end 0.10 \
  --candidate-noise-anneal-start-step 20000 --candidate-noise-anneal-steps 100000 \
  --candidate-explore-probability-start 1.0 --candidate-explore-probability-end 0.35 \
  --candidate-explore-anneal-steps 50000 \
  --persistent-noise-correlation 0.95 --persistent-gripper-noise-scale 0.5 \
  --background-probe-episode-probability 0.80 --background-probe-episode-probability-end 0.25 \
  --background-probe-anneal-start-step 20000 --background-probe-anneal-steps 100000 \
  --background-probe-max-chunks-per-episode 5 --background-probe-max-boundary 15 \
  --background-probe-min-value-improvement -0.20 \
  --actor-safety-enabled --actor-safety-min-value-improvement -0.02 \
  --actor-safety-cooldown-chunks 3 --gate-min-active-chunks 1 \
  --gate-immediate-max-entropy 1.5 --warmup-transitions 256 \
  --actor-success-bc-weight 10.0 --actor-success-bc-min-value-improvement 0.01 \
  --actor-l2-weight 0.10 --actor-smoothness-weight 0.20 \
  --actor-value-objective-weight 0.0 --actor-awr-weight 0.0 \
  --output-dir outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_noise_curriculum_resume200k \
  --wandb-enabled --wandb-project maniskill-vgate-advantage-bc \
  --wandb-run-name panel-positivebc-seed44900-noise-curriculum-resume200k
```

## Paired evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/eval_value_guided_bandit.py \
  --checkpoint outputs/rlt/SolarPanelStatic-v2_vgate_successbc_positive_seed44900_noise_curriculum_resume200k/value_guided_bandit.pt \
  --value-checkpoint outputs/rlt/value/SolarPanelStatic-v2_safe_visual_recap_value_seed41000/value_best.pt \
  --server ws://127.0.0.1:8012 --env-id SolarPanelStatic-v2 \
  --obs-mode rgb --reward-mode sparse --control-mode pd_joint_pos \
  --sim-backend physx_cpu --render-backend sapien_cuda:0 --device cpu \
  --start-seed 53000 --num-seeds 100 \
  --output-dir /tmp/panel44900_noise_eval
```

The validated noise-curriculum checkpoint reached 82% paired success on the
100-seed development evaluation (base 58%, 25 rescues, 1 regression). The
historical 91% temporal-latent archive is retained separately at
`outputs/rlt/SolarPanelStatic-v2_goal95_temporal_latent_seed5000_100k` and is
not part of the maintained pipeline.

## Offline V_base training

The retained supporting scripts are `scripts/rlt/collect_value_rollouts.py`
and `scripts/rlt/train_value_model.py`. The base rollout dataset is
`outputs/rlt/value_rollouts/SolarPanelStatic-v2_safe_base.h5`; the frozen value
checkpoint used by both online runs is
`outputs/rlt/value/SolarPanelStatic-v2_safe_visual_recap_value_seed41000/value_best.pt`.

## Tests

```bash
pytest -q tests/test_rlt_value_guided_bandit.py tests/test_rlt_value_model.py \
  tests/test_rlt_backend.py tests/test_rlt_policies.py tests/test_rlt_reset.py
```
