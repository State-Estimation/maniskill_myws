# maniskill_myws

自定义 ManiSkill 任务 + openpi(π0/π0.5) 训练/推理工作区。

## 快速开始（本地 ManiSkill 源码）

1) 安装 ManiSkill（本地源码或已有环境），确保 `import mani_skill` 可用。  
2) 安装本工作区：

```bash
cd /your/path/to/maniskill_myws
pip install -e .[runtime,dev]
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
- `OpenSafeDoor-v1`（asset `101593`，旋钮>90° 且门>60°）
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
python scripts/record_random.py --env-id TurnGlobeValve-v1 --out-dir data/demos/debug --num-episodes 5
```
输出为 ManiSkill 原生 `.h5 + .json`。

## openpi(π0/π0.5) 集成（Submodule 方案）

本仓库使用 **Git submodule** 固定 openpi 版本，路径为：
`third_party/openpi`

初始化子模块：
```bash
cd /your/path/to/maniskill_myws
git submodule update --init --recursive
```

更详细说明见：
- `docs/openpi_integration.md`
- `docs/maniskill_dataset_standard.md`

### 训练端环境（openpi uv 环境）
```bash
cd /your/path/to/maniskill_myws/third_party/openpi
uv sync
```

### 客户端环境（ManiSkill）
```bash
pip install -e /your/path/to/maniskill_myws/third_party/openpi/packages/openpi-client
```

### 最小两进程流程（server + ManiSkill client）

1) **启动 π0 服务端（openpi uv 环境）**
```bash
cd /your/path/to/maniskill_myws/third_party/openpi
uv run python /your/path/to/maniskill_myws/scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

2) **启动 ManiSkill 客户端（mani_skill 环境）**
```bash
python /your/path/to/maniskill_myws/scripts/pi0/run_pi0_remote.py \
  --server ws://127.0.0.1:8000 \
  --env-id TurnGlobeValve-v1 \
  --save-images
```

## 训练/数据工具入口

- `.h5 -> LeRobot`：`scripts/convert_traj_to_lerobot.py`
- 数据验证：`scripts/pi0/validate_lerobot_dataset.py`
- 一键微调：`scripts/pi0/finetune_maniskill.py`

> 这些脚本的完整参数与示例见 `docs/openpi_integration.md`。
