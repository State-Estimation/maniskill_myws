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
- `trainer.py`: TD target, actor/critic updates, pd_joint_pos regularizers
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
