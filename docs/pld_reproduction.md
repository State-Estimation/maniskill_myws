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
- The paper also uses a visual encoder for residual RL. That can be added later,
  but state-only is much easier to debug and already preserves the core PLD
  mechanism: base-policy probing, residual takeover, and SFT data generation.
- The paper initializes critics with Cal-QL. This implementation provides normal
  SAC updates plus an optional CQL-style penalty via `--cql-alpha`. Treat it as
  a pragmatic first pass before adding full Cal-QL.
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

- `--action-scale`: residual magnitude `xi`; paper uses `0.5`.
- `--warmup-episodes`: base-policy-only online warmup; paper uses `100`.
- `--offline-fraction`: offline/online replay mix; paper samples them equally.
- `--probe-alpha`: max probing horizon ratio for PLD data collection; start at
  `0.6`.
- `--cql-alpha`: optional conservative Q penalty during SAC updates.

## Environment Note

In this terminal session, `robotwin-cu130` can load H5 and run torch updates, but
SAPIEN environment creation failed because no supported render device was
available. The scripts are written for the normal ManiSkill runtime where your
OpenSafeDoor-v2 task can be instantiated with RGB observations.
