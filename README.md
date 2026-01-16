# maniskill_myws

Custom ManiSkill tasks + imitation learning workspace (local-source development).

## Quick start (local-source ManiSkill)

- **Install ManiSkill (your local source checkout)**: follow your ManiSkill repo instructions, then make sure `import mani_skill` works.
- **Install this workspace**:

```bash
cd /home/sisyphus/Projects/maniskill_myws
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

## Imitation learning (Behavior Cloning baseline)

Once you have a `.h5` trajectory file (recorded with `RecordEpisode`), you can train a minimal BC policy:

```bash
python scripts/bc_train.py --h5 path/to/trajectory.h5 --out outputs/bc_ckpt.pt
```

Notes:
- This BC script expects trajectories to include `obs` and `actions` in the `.h5`.
- It flattens nested observation/action dicts by concatenating leaf datasets.
