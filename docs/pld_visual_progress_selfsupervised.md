# Pure-visual self-supervised task progress

This model predicts full-task progress from RGB clips without reading robot or
environment state. It is separate from the privileged `clean_coverage` progress
head documented in `pld_progress_head.md`.

## Supervision contract

Model inputs:

- `sensor_data/base_camera/rgb`;
- `sensor_data/hand_camera/rgb`;
- a short causal image history.

The dataset intentionally never returns `qpos`, `qvel`, `tcp_pose`, object pose,
contact, reward, or `clean_coverage`. Training uses only:

- frame order inside a trajectory;
- two successful trajectories for cross-video temporal cycle consistency;
- the episode-level success/failure result;
- start/end anchors of successful trajectories.

There are no frame-level progress labels and no grasp/move/contact phase
annotations. The model does not prescribe a phase count. Grasp, transport,
contact, and cleaning transitions are represented implicitly in the continuous
cycle-consistent embedding.

## Objectives

- temporal cycle consistency aligns variable-speed successful videos;
- aligned-progress consistency maps corresponding visual states to the same
  progress;
- successful start/end anchoring maps complete trials from zero to one;
- ordinal and smoothness losses organize successful visual states in time;
- latent temporal smoothness regularizes the continuous embedding without
  imposing discrete phase classes or boundaries;
- episode outcome BCE and terminal trajectory preference calibrate autonomous
  failure rollouts;
- three independently initialized members provide ensemble disagreement rather
  than the collapsed single-head variance used by the privileged model.

## Train on SolarPanelStatic-v2

```bash
cd /home/sisyphus/Projects/maniskill_myws
conda run --no-capture-output -n mani_skill \
  python -u scripts/pld/run_train_from_yaml.py \
  configs/pld/solar_panel_visual_progress_selfsupervised.yaml
```

Outputs:

- `visual_progress_best.pt`: best complete held-out loss;
- `visual_progress_latest.pt`: most recent validation checkpoint;
- `visual_progress_last.pt`: final checkpoint.

The checkpoint metadata records `visual_only=true`,
`dense_labels_used=false`, `stage_annotations_used=false`, and
`phase_representation=implicit_cycle_consistent_embedding`.

## Live rollout curve

Start the OpenPI websocket server, then run on seeds excluded from training:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mani_skill \
python -u scripts/pld/eval_visual_progress_rollout.py \
  --checkpoint outputs/pld/SolarPanelStatic-v2/visual_progress_implicit/visual_progress_best.pt \
  --env-id SolarPanelStatic-v2 \
  --control-mode pd_joint_pos \
  --base-policy remote_openpi \
  --server ws://127.0.0.1:8010 \
  --num-seeds 20 \
  --start-seed 10000 \
  --rollout-window-steps 50 \
  --env-device cuda:0 \
  --progress-device cuda:0 \
  --live-plot \
  --plot-width 6 \
  --plot-height 5 \
  --realtime-render
```

Each seed writes `progress.csv`, `progress_curve.png`, and `summary.json`.
`--live-plot` opens the updating progress curve, while `--realtime-render`
opens the ManiSkill environment viewer. The latter requires a graphical display.
Use `--render-every N` to reduce viewer refresh overhead.
The progress window defaults to 6 by 5 inches. Use `--plot-width` and
`--plot-height` to make it smaller or larger.

For a simulation-only audit, add:

```bash
  --audit-progress-key extra/clean_coverage \
  --audit-progress-goal 0.6
```

The audit value is read after prediction and is only written to evaluation
outputs; it is never an input to the model. It audits the cleaning portion, not
the unlabeled grasp/move/contact transitions.

## Interpreting curves

A successful trial should start near zero, move continuously through the latent
trajectory, and terminate near one. Peaks in `latent_change` indicate candidate
phase transitions but are not fixed stage IDs. A missed-grasp trial should
plateau and its eventual success probability should fall. Do not enforce a
monotonic clamp during evaluation: regressions and recovery attempts are useful
diagnostics.

Before using this model as an RL potential, inspect latent-change keyframes and
failure curves. Use a bounded potential difference rather than the absolute prediction:

```text
r_shape = lambda * (gamma * progress(next_obs) - progress(obs))
```

Gate shaping on ensemble disagreement and preserve the environment reward in a
separate log channel.
