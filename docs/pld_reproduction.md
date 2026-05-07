# PLD Reproduction Notes

This repo now contains a lightweight implementation path for reproducing the
paper's Probe, Learn, Distill (PLD) workflow on the custom ManiSkill tasks.

The implementation is intentionally staged so we can reuse the existing assets:

- ManiSkill custom tasks in `src/maniskill_myws/tasks`
- pi0/openpi remote policy bridge in `src/maniskill_myws/openpi_bridge`
- existing successful OpenSafeDoor-v2 replay data in
  `dataset/Replayed_traj_data_openSafeDoor2`
- existing H5-to-LeRobot conversion script in `scripts/convert_traj_to_lerobot.py`

## Paper-To-Code Mapping

PLD Stage 1: specialist acquisition

- Paper: freeze VLA base policy `pi_b`, train a lightweight residual actor
  `pi_delta(s, a_b)` with off-policy RL.
- Code: `scripts/pld/train_residual_sac.py`.
- Model: state-only residual SAC actor conditioned on `(state, base_action)`.
- Offline buffer: successful replay H5 files, loaded by
  `maniskill_myws.pld.h5_replay`.
- Online buffer: new transitions collected while using the frozen base policy.

PLD Stage 2: hybrid data collection

- Paper: run the base policy for `T_base`, then let residual specialist take over
  with `a = a_base + a_delta`.
- Code: `scripts/pld/collect_hybrid_pld.py`.
- Output: ManiSkill `RecordEpisode` `.h5 + .json`, compatible with
  `scripts/convert_traj_to_lerobot.py`.
- Default probing: `T_base` is uniformly sampled from `[0, 0.6T]`, matching the
  paper's ablation plateau around `alpha = 0.6`.

PLD Stage 3: distillation/SFT

- Paper: supervised fine-tune the VLA on curated PLD trajectories.
- Code path: use the existing conversion and openpi finetuning tools:
  `scripts/convert_traj_to_lerobot.py` then `scripts/pi0/finetune_maniskill.py`.

## Current Scope

This is a practical MVP for this workspace, not a byte-for-byte reproduction of
the authors' full internal stack.

- The residual RL specialist is state-only: by default it uses
  `agent/qpos`, `agent/qvel`, and `extra/tcp_pose`, giving a 25D state on the
  existing OpenSafeDoor-v2 replay data.
- The paper also uses a ResNetV1-10 visual encoder for residual RL. This repo
  keeps state-only as the default for fast debugging, and enables visual RL with
  `--use-visual-rl`.
- The paper initializes critics with Cal-QL. This implementation provides
  critic-only Cal-QL warm-start via `--offline-pretrain-method calql`.
- The paper uses OTF to select the best action among base/edit candidates for
  sampling and TD backup. This implementation adapts OTF to PLD by holding the
  remote base-policy action fixed and sampling multiple residual edits.
- Existing replay trajectories do not store a separate frozen VLA action. The
  loader defaults to `--offline-base-action-mode action`, meaning the recorded
  action is used as the offline base-action proxy. Online PLD training should
  still use the real base policy via `--base-policy remote_openpi`.

## Verified Existing Data

In `robotwin-cu130`, the loader sees:

```text
files: 6
transitions: 37369
state_dim: 25
action_dim: 7
reward_min: 0.0
reward_max: 1.0
```

The default offline path is:

```text
dataset/Replayed_traj_data_openSafeDoor2
```

For the closest match to Algorithm 1, the offline buffer should contain
successful frozen base-policy rollouts, not teleop/joystick demonstrations. To
collect that dataset from a running OpenPI server:

```bash
conda run -n robotwin-cu130 python scripts/pld/collect_base_policy_dataset.py \
  --env-id OpenSafeDoor-v2 \
  --server ws://127.0.0.1:8000 \
  --num-successes 50 \
  --max-attempts 200 \
  --output-dir dataset/Pi0_rollout_OpenSafeDoor-v2 \
  --trajectory-name pi0_base_policy
```

Then train PLD with `--offline-h5-dir dataset/Pi0_rollout_OpenSafeDoor-v2`.

## Stage 1: Train Residual Specialist

For a real PLD run, start the pi0/openpi server first:

```bash
cd third_party/openpi
uv run python ../../scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

Then train the residual specialist from the ManiSkill environment:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2 \
  --total-env-steps 250000
```

To use the Cal-QL critic warm-start described in PLD, replace the default SAC
offline pretrain with critic-only Cal-QL pretraining:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2_calql \
  --total-env-steps 250000 \
  --offline-pretrain-method calql \
  --offline-pretrain-updates 1000
```

This computes discounted Monte Carlo return-to-go values from the successful H5
trajectories and uses them as Cal-QL lower bounds for sampled policy actions.
Only `Q1/Q2` are updated during this phase; the residual actor stays randomly
initialized until online residual SAC starts.

To enable OTF during residual SAC, add OTF candidate counts for backup and/or
online rollout:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2_otf \
  --total-env-steps 250000 \
  --offline-pretrain-method calql \
  --offline-pretrain-updates 20000 \
  --otf-backup-actions 4 \
  --otf-rollout-actions 4
```

`--otf-backup-actions N` samples `N` residual edits for the next state and backs
up the highest critic value, matching the OTF paper's hard-Q TD target.
`--otf-rollout-actions N` samples `N` residual edits during online rollout and
executes the critic-best action. By default the unedited base action is also
included as a candidate, so `N=4` means five total candidates. Use
`--otf-no-base-candidate` only if you explicitly want edited actions to compete
without the base-policy fallback. Use `--otf-backup-entropy` if you want the OTF
target to use SAC soft values `Q - alpha log pi` instead.

To train the residual actor/critic from proprioception plus RGB, add
`--use-visual-rl`. The visual path loads the base and wrist camera RGB streams,
feeds them through a compact ResNetV1-10-style encoder, and concatenates the
visual latent with `--state-keys` before the actor/Q MLPs:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2_visual \
  --total-env-steps 250000 \
  --offline-pretrain-method calql \
  --offline-pretrain-updates 20000 \
  --use-visual-rl \
  --rl-image-size 128
```

For quick debugging, use a smaller image size such as `--rl-image-size 64` and a
smaller `--buffer-capacity`; full visual replay with two 128x128 cameras is much
heavier than the default state-only replay. Warmup buffers are shape-specific:
a state-only `.npz` warmup buffer cannot be reused when `--use-visual-rl` is
enabled, because it does not contain RGB observations.

To reuse the base-policy warmup across runs, add a persistent warmup buffer path:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2 \
  --total-env-steps 250000 \
  --warmup-buffer-path outputs/pld/OpenSafeDoor-v2/warmup_buffer.npz
```

On the first run, the script saves the online replay gathered during
`--warmup-episodes`. Later runs load the `.npz` automatically, skip warmup, and
resume counting `env_step` from the saved warmup transition count.

To watch training live, add realtime rendering:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2 \
  --total-env-steps 250000 \
  --render-mode human \
  --render-every 1
```

To log PLD residual training to Weights & Biases, add:

```bash
  --wandb-enabled \
  --wandb-project maniskill-pld \
  --wandb-name opensafedoor_residual_sac_seed0
```

The script records offline pretrain metrics, per-episode success/return, online
training metrics, warmup-buffer load/save events, and checkpoint saves.

After training, evaluate every saved residual checkpoint with deterministic
actions:

```bash
conda run -n robotwin-cu130 python scripts/pld/eval_residual_checkpoints.py \
  --checkpoint-dir outputs/pld/OpenSafeDoor-v2 \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --num-episodes 20 \
  --start-seed 0 \
  --wandb-enabled \
  --wandb-project maniskill-pld \
  --wandb-name opensafedoor_checkpoint_eval
```

For a base-policy-only baseline on the same seeds:

```bash
conda run -n robotwin-cu130 python scripts/pld/eval_residual_checkpoints.py \
  --mode base \
  --checkpoint-dir outputs/pld/OpenSafeDoor-v2 \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --num-episodes 20 \
  --start-seed 0
```

The evaluator writes `checkpoint_eval.csv` and `checkpoint_eval.json` under
`<checkpoint-dir>/eval/`. By default all checkpoints use the same eval seeds so
their success rates are directly comparable.

If the checkpoint was trained/evaluated as an OTF policy, add the same rollout
candidate count to checkpoint eval, for example `--otf-rollout-actions 4`.
Visual checkpoints automatically require the same camera keys; override
`--rl-image-keys` only if the checkpoint was trained with non-default cameras.

To also visualize the gripper TCP path in the viewer, add
`--visualize-tcp-path`. Blue markers show the current base-policy future action
chunk projected from the current TCP pose by accumulating each action's
`dx,dy,dz`. Orange markers show the true TCP positions reached after executing
`a_base + a_delta`:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --offline-h5-dir dataset/Replayed_traj_data_openSafeDoor2 \
  --output-dir outputs/pld/OpenSafeDoor-v2 \
  --total-env-steps 250000 \
  --render-mode human \
  --render-every 1 \
  --visualize-tcp-path \
  --base-chunk-max-actions 16 \
  --base-chunk-position-scale 0.1 \
  --path-every 2
```

`--base-chunk-position-scale` should match the controller's position scale. For
Panda `pd_ee_delta_pose`, normalized actions map to roughly `[-0.1, 0.1]`
meters, so `0.1` is the default.

Use a larger `--render-every`, such as `5` or `10`, if rendering makes training
too slow.

If you only want to verify the offline loading and SAC update path without
starting SAPIEN rendering:

```bash
conda run -n robotwin-cu130 python scripts/pld/train_residual_sac.py \
  --total-env-steps 0 \
  --offline-pretrain-updates 10 \
  --max-offline-transitions 2048 \
  --device cpu \
  --output-dir /tmp/maniskill_pld_smoke
```

## Stage 2: Collect Hybrid PLD Data

After Stage 1 produces `outputs/pld/OpenSafeDoor-v2/residual_sac.pt`, collect
hybrid takeover trajectories:

```bash
conda run -n robotwin-cu130 python scripts/pld/collect_hybrid_pld.py \
  --checkpoint outputs/pld/OpenSafeDoor-v2/residual_sac.pt \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --num-episodes 100 \
  --probe-alpha 0.6 \
  --output-dir data/pld/OpenSafeDoor-v2
```

By default, failed trajectories are dropped. Add `--keep-failed` if you want to
inspect failures.

## Stage 3: Convert And Fine-Tune pi0

Convert successful hybrid trajectories to LeRobot:

```bash
conda run -n robotwin-cu130 python scripts/convert_traj_to_lerobot.py \
  --h5-dir data/pld/OpenSafeDoor-v2 \
  --repo-id local/pld_open_safe_door_v2 \
  --image-key obs/sensor_data/base_camera/rgb \
  --wrist-image-key obs/sensor_data/hand_camera/rgb \
  --state-keys obs/agent/qpos obs/agent/qvel obs/extra/tcp_pose \
  --actions-key actions \
  --task-from env_default \
  --myws-root .
```

Then launch the existing openpi fine-tuning helper:

```bash
cd third_party/openpi
uv run python ../../scripts/pi0/finetune_maniskill.py \
  --openpi-root . \
  --config pi05_libero \
  --repo-id local/pld_open_safe_door_v2 \
  --assets-base-dir ../../assets_openpi \
  --checkpoint-base-dir ../../checkpoints_openpi \
  --num-train-steps 30000
```

## Useful Knobs

- `--action-scale`: residual magnitude `xi`; the paper suggests `0.5` for
  LIBERO and `0.1` for SimplerEnv. OpenSafeDoor-v2 is closer to the latter, so
  `0.1` is a safer default.
- `--warmup-episodes`: base-policy-only online warmup; paper uses `100`.
- `--warmup-buffer-path`: persist the warmup online replay so later runs can
  skip redoing the base-policy warmup.
- `--offline-pretrain-method calql`: use Cal-QL critic-only warm-start instead
  of the older SAC-style offline pretrain.
- `--use-visual-rl`: enable RGB + proprio residual RL with a ResNetV1-10-style
  encoder. Keep disabled for fast state-only debugging.
- `--rl-image-size`: image size stored in replay. `128` matches the current H5
  camera resolution; lower values reduce memory and speed up iteration.
- `--wandb-enabled`: turn on wandb logging for offline pretrain, RL training,
  warmup-buffer events, and checkpoints.
- `--offline-fraction`: offline/online replay mix; paper samples them equally,
  and the default is `0.5`.
- `--probe-alpha`: max probing horizon ratio for PLD data collection; start at
  `0.6`.
- `--cql-alpha`: optional conservative Q penalty during online SAC updates; the
  default is `0.0`, so Cal-QL pretraining is the only conservative critic
  regularization unless this is explicitly enabled.

## Environment Note

In this terminal session, `robotwin-cu130` can load H5 and run torch updates, but
SAPIEN environment creation failed because no supported render device was
available. The scripts are written for the normal ManiSkill runtime where your
OpenSafeDoor-v2 task can be instantiated with RGB observations.
