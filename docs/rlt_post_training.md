# ManiSkill RLT Post-Training

This branch contains a ManiSkill-native PyTorch implementation of the RLT
online post-training structure without RL-token conditioning.

## Mapping From RLT To ManiSkill

- Original RLT actor input: `z_rl + proprio + ref_chunk`
- ManiSkill RLT actor input: `state + optional RGB encoder + ref_chunk`
- Original RLT action default: 7D real-robot action
- ManiSkill RLT action: strict `pd_joint_pos` 8D action

The 8D action layout is:

```text
[7 Panda arm joint position targets, 1 gripper target]
```

## Model

Core code lives in `src/maniskill_myws/rlt`:

- `networks.py`: chunk actor, twin critic, optional ResNetV1-10 RGB encoder
- `replay.py`: fixed-length chunk replay buffer
- `dataset.py`: ManiSkill rollout H5 to Base-policy replay chunks
- `trainer.py`: TD target, actor/critic updates, pd_joint_pos regularizers
- `hil.py`: keyboard-controlled human-in-the-loop Base/RLT gate
- `state.py`: ManiSkill state/RGB adapters
- `policies.py`: zero/random/OpenPI reference chunk providers

The actor is conditioned on a frozen reference chunk from OpenPI. By default it
predicts a bounded correction around that reference:

```text
action_chunk = clip(ref_chunk + tanh(raw_delta) * action_delta_scale)
```

Set `actor_output_mode="absolute"` in `RLTTrainConfig` if you want the actor to
predict absolute chunks directly.

## Loss

The critic follows RLT-style chunk TD learning:

```text
target = discounted_chunk_reward + gamma^T * min(Q1_target, Q2_target)
```

The actor uses:

```text
actor_loss = bc_weight * BC
           - q_weight * Q
           + correction_weight * correction
           + smoothness_weight * smoothness
```

For ManiSkill `pd_joint_pos`, the regularizers are redefined:

- `correction`: MSE between predicted and reference chunks for the first 7 arm
  joint dimensions only
- `smoothness`: MSE between predicted and target step-to-step arm joint deltas
- `gripper_smoothness`: computed on the 8th dimension, down-weighted by
  `gripper_smoothness_weight` because gripper commands can be discontinuous

## Online Training

Start the OpenPI server first, then run:

```bash
python scripts/rlt/train_chunk_rlt_online.py \
  --env-id OpenSafeDoor-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --control-mode pd_joint_pos \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --chunk-len 50 \
  --warmup-transitions 600 \
  --updates-per-chunk 5 \
  --output-dir outputs/rlt/OpenSafeDoor-v2
```

`--chunk-len` should match the action horizon of the reference policy checkpoint
unless you intentionally want to truncate the chunk and replan more often. The
default `pi0_maniskill` config uses `Pi0Config()` whose action horizon is 50.

For visual RLT, add:

```bash
  --use-visual-rlt \
  --rlt-image-size 128
```

### Prefill warmup from existing rollouts

Existing ManiSkill RecordEpisode `.h5` rollouts can replace live Base-policy
warmup collection:

```bash
python scripts/rlt/train_chunk_rlt_online.py \
  --env-id TurnGlobeValve-v1 \
  --reward-mode sparse \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --chunk-len 50 \
  --warmup-transitions 600 \
  --warmup-dataset \
    dataset/Pi0_rollout_TurnGlobeValve-v1_pd_joint_pos/pi0_base_policy.h5 \
  --offline-updates 1000 \
  --output-dir outputs/rlt/TurnGlobeValve-v1_prefilled
```

The loader uses contiguous rollout actions as both `ref_chunk` and
`action_chunk`, labels every loaded step as `BASE`, and preserves rewards,
termination, success, optional RGB observations, and episode boundaries. It
loads at most `--warmup-transitions` chunks by default; override that limit with
`--warmup-dataset-transitions`. Repeat `--warmup-dataset` to load more than one
compatible H5 file.

The sibling JSON metadata is checked against `--env-id`, `--control-mode`, and
`--reward-mode`. Use a matching reward mode because mixing sparse and dense
critic targets is normally invalid. `--allow-warmup-metadata-mismatch` is
available only for intentionally transformed datasets. Nonstandard H5 fields
can be selected with `--warmup-action-key` and `--warmup-reward-key`.

`--offline-updates` trains from the prefilled buffer before the first online
step. If the H5 contains fewer than `--warmup-transitions` chunks, online Base
rollouts continue filling the remainder and HIL keeps RLT locked until the
threshold is reached.

### Human-in-the-loop intervention timing

Add `--hil-keyboard --render-mode human` when a person should decide exactly
when RLT is allowed to intervene:

```bash
python scripts/rlt/train_chunk_rlt_online.py \
  --env-id TurnGlobeValve-v1 \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8000 \
  --chunk-len 50 \
  --hil-keyboard \
  --hil-mode hold \
  --render-mode human \
  --real-time \
  --output-dir outputs/rlt/TurnGlobeValve-v1_hil
```

Click the SAPIEN viewer once so it has keyboard focus. In the default `hold`
mode, Base/OpenPI controls the robot unless `R` is held; releasing `R` returns
control to Base immediately. Holding `B` forces Base even if `R` is also held.
Press `Q` to stop safely and save the checkpoint.

For latched control, use `--hil-mode latch`: press `R` once to enable RLT and
press `B` to return to Base. Keys can be changed with `--hil-rlt-key`,
`--hil-base-key`, and `--hil-quit-key`.

Keyboard HIL enables real-time pacing by default, using the environment's
`control_freq` (20 Hz for the four custom tasks). `--real-time` can also enable
it explicitly, while `--no-real-time` disables it. Real-time pacing is needed
for reliable human input because an unthrottled simulator can finish an episode
before a key can be held or tapped.

The gate is polled before every low-level environment step, so a single chunk
can contain both Base and RLT actions. Replay stores the per-step source and
marks the aggregate transition as `MIXED`. RLT remains locked until
`--warmup-transitions` Base chunks have been collected or prefilled; this
argument counts chunk transitions, not individual environment steps.

## Evaluation

Evaluate the final RLT checkpoint against the same OpenPI reference server:

```bash
python scripts/rlt/eval_chunk_rlt.py \
  --env-id TurnGlobeValve-v1 \
  --checkpoint outputs/rlt/TurnGlobeValve-v1_pi0_chunk50/maniskill_rlt.pt \
  --server ws://127.0.0.1:8000 \
  --start-seed 108 \
  --num-seeds 100 \
  --max-steps 600 \
  --output-name TurnGlobeValve-v1_rlt_chunk50_eval_108_207
```

For live visualization of a single rollout:

```bash
python scripts/rlt/eval_chunk_rlt.py \
  --env-id TurnGlobeValve-v1 \
  --checkpoint outputs/rlt/TurnGlobeValve-v1_pi0_chunk50/maniskill_rlt.pt \
  --server ws://127.0.0.1:8000 \
  --seed 108 \
  --num-seeds 1 \
  --max-steps 600 \
  --render-mode human \
  --real-time
```

The implementation intentionally rejects non-8D actions so it cannot silently
train on a controller other than ManiSkill `pd_joint_pos`.
