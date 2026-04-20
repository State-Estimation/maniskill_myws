# maniskill_myws

自定义 ManiSkill 任务 + openpi(π0/π0.5) 训练/推理工作区。

## 快速开始（本地 ManiSkill 源码）

1) 安装 ManiSkill（本地源码或已有环境），确保 `import mani_skill` 可用。  
2) 安装本工作区：

```bash
cd /path/to/maniskill_myws
python -m pip install -e .[runtime,dev]
```

## 环境注册与创建

```python
import maniskill_myws
maniskill_myws.register()
```

```python
import gymnasium as gym
import maniskill_myws

maniskill_myws.register()
env = gym.make("TurnGlobeValve-v1", obs_mode="state", reward_mode="none", render_mode=None)
obs, info = env.reset(seed=0)
```

已注册环境：
- `TurnGlobeValve-v1`
- `OpenSafeDoor-v1`
- `OpenSafeDoor-v2`
- `BrushSolarPanel-v1`
- `OpenSafetyHook-v1`
- `TakeSafetyHook-v1`
- `SolarPanelStatic-v1`
- `StackCube-v2`（VLA 标准采集传感器）

## 常用脚本

1) **检查环境可用性**
```bash
python scripts/check_env.py --env-id TurnGlobeValve-v1
```

2) **运行 ManiSkill 官方 demo（随机动作）**
```bash
python scripts/run_maniskill_demo.py mani_skill.examples.demo_random_action -e TurnGlobeValve-v1 --render-mode human
```

3) **手动控制 demo 注意事项**
`demo_manual_control(_continuous)` 期望 `render_mode` 返回图像，**不兼容 `--render-mode human`**。需要 SAPIEN Viewer 时使用：
```bash
python scripts/run_maniskill_demo.py mani_skill.examples.demo_manual_control_continuous \
  -e TurnGlobeValve-v1 --render-mode sensors --enable-sapien-viewer
```

4) **录制随机轨迹（用于 imitation pipeline 调试）**
```bash
python scripts/record_random.py \
  --env-id TurnGlobeValve-v1 \
  --out-dir data/demos/debug \
  --num-episodes 5 \
  --obs-mode rgb
```
输出为 ManiSkill 原生 `.h5 + .json`。如果你准备用 `scripts/convert_traj_to_lerobot.py`
直接转 LeRobot/openpi 数据，建议采集时显式使用 `--obs-mode rgb`，或者先用 replay 工具补齐 `sensor_data/*` 图像。

## openpi(π0/π0.5) 集成（Submodule 方案）

本仓库使用 **Git submodule** 固定 openpi 版本，路径为：
`third_party/openpi`

初始化子模块：
```bash
cd /path/to/maniskill_myws
git submodule update --init --recursive
```

更详细说明见：
- `docs/openpi_integration.md`
- `docs/maniskill_dataset_standard.md`

### 训练端环境（openpi uv 环境）
```bash
cd third_party/openpi
uv sync
```

### 客户端环境（ManiSkill）
```bash
python -m pip install -e third_party/openpi/packages/openpi-client
```

### 最小两进程流程（server + ManiSkill client）

1) **启动 π0 服务端（openpi uv 环境）**
```bash
cd third_party/openpi
uv run python ../../scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

自定义微调 checkpoint 也可以直接这样启动。`serve.py` 会优先从 checkpoint 下的 `assets/*/norm_stats.json`
自动推断训练时使用的 `repo_id/asset_id`；只有在 checkpoint 里有多份资产、或者你想强制覆盖时，才需要额外传
`--repo-id` 或 `--norm-stats`。

2) **启动 ManiSkill 客户端（mani_skill 环境）**
```bash
python scripts/pi0/run_pi0_remote.py \
  --server ws://127.0.0.1:8000 \
  --env-id TurnGlobeValve-v1 \
  --save-images
```

3) **多 seed 评测（可选）**
```bash
python scripts/pi0/run_pi0_remote_multi_seed.py \
  --server ws://127.0.0.1:8000 \
  --env-id OpenSafeDoor-v2 \
  --num-seeds 20 \
  --start-seed 0 \
  --render-mode human \
  --save-videos \
  --video-views both
```

说明：
- `--render-mode human` 会打开 ManiSkill 实时渲染窗口；无桌面环境时不要加这个参数。
- `--video-views base|wrist|both` 控制保存 mp4 时使用哪个观测视角。
- 默认 `base` 对应 `sensor_data/base_camera/rgb`，`wrist` 对应 `sensor_data/hand_camera/rgb`。
- 如果你想改视频里“base”所对应的实际相机位姿，需要修改任务里的 `base_camera` 定义；`run_pi0_remote_multi_seed.py` 只决定保存哪个观测流。

## 训练/数据工具入口

- `.h5 -> LeRobot`：`scripts/convert_traj_to_lerobot.py`
- 数据验证：`scripts/pi0/validate_lerobot_dataset.py`
- 一键微调：`scripts/pi0/finetune_maniskill.py`

> 这些脚本的完整参数与示例见 `docs/openpi_integration.md`。
