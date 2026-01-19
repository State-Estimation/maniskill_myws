# maniskill_myws

Custom ManiSkill tasks + imitation learning workspace (local-source development).

## Quick start (local-source ManiSkill)

- **Install ManiSkill (your local source checkout)**: follow your ManiSkill repo instructions, then make sure `import mani_skill` works.
- **Install this workspace**:

```bash
cd /your/path/to/maniskill_myws
pip install -e .[runtime,dev]
```

## Create the custom environment

This workspace uses a `register()` entrypoint to ensure all tasks are imported and registered:

```python
import maniskill_myws
maniskill_myws.register()
```

Then you can create the environment:

```python
import gymnasium as gym
import maniskill_myws

maniskill_myws.register()
env = gym.make("TurnGlobeValve-v1", obs_mode="state", reward_mode="none", render_mode=None)
obs, info = env.reset(seed=0)
```

Available env IDs (in this repo):
- `TurnGlobeValve-v1`
- `OpenSafeDoor-v1` (asset `101593`, knob rotate > 90deg then door open > 60deg)

## Scripts

- **Check env can be created**:

```bash
python scripts/check_env.py --env-id TurnGlobeValve-v1
```

- **Run ManiSkill's random-action demo with your custom task (supports `--render-mode human`)**:

```bash
python scripts/ms_demo_random_action.py -e TurnGlobeValve-v1 --render-mode human
```

- **Run any ManiSkill demo module (generic wrapper)**:

```bash
python scripts/run_maniskill_demo.py mani_skill.examples.demo_random_action -e TurnGlobeValve-v1 --render-mode human
```

- **Manual-control demos 注意事项**:
  `mani_skill.examples.demo_manual_control(_continuous)` 脚本内部会执行 `env.render().cpu().numpy()`，因此它们期望 `render_mode` 返回图像（例如 `sensors`/`rgb_array`），**不兼容 `--render-mode human`**。
  如果你想用 SAPIEN Viewer，请保持 `--render-mode sensors`，并额外加 `--enable-sapien-viewer`：

```bash
python scripts/run_maniskill_demo.py mani_skill.examples.demo_manual_control_continuous -e TurnGlobeValve-v1 --render-mode sensors --enable-sapien-viewer
```

- **Record random rollouts (debugging the imitation pipeline)**:

```bash
python scripts/record_random.py --env-id TurnGlobeValve-v1 --out-dir data/demos/debug --num-episodes 5
```

Trajectory output is ManiSkill's native `.h5 + .json` format.

## openpi (π0/π0.5) integration

This workspace can be used to train / run openpi VLA policies on custom ManiSkill tasks by:
- converting RecordEpisode trajectories (`.h5`) to a LeRobot dataset
- running openpi fine-tuning with existing configs (LIBERO-style)
- rolling out the trained policy in ManiSkill

See: `docs/openpi_integration.md`

### Minimal “GraspVLA-style” two-process workflow (server + ManiSkill client)

1) **Start π0 server** (separate env/machine with openpi installed):

```bash
python scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

2) **Run ManiSkill visualization client** (your `conda activate mani_skill` env):

```bash
pip install -e /home/sisyphus/Projects/openpi/packages/openpi-client
python scripts/pi0/run_pi0_remote.py --server ws://127.0.0.1:8000 --env-id TurnGlobeValve-v1 --save-images
```

## Imitation learning (Behavior Cloning baseline)

WIP
