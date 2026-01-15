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
