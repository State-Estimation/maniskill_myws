## ManiSkill 标准数据集格式（maniskill_myws 统一规范）

目标：让自定义任务的数据结构与 openpi/LIBERO 风格一致，便于统一转换与训练。

---

## A) 统一的观测与控制配置（采集/回放/rollout 都使用）

**环境建议参数**
- `robot_uids="panda_wristcam"`（提供 wrist 相机）
- `control_mode="pd_ee_delta_pose"`（7D action）
- `obs_mode="rgb"`（标准 RGB 观测）
- `reward_mode="none"`（数据采集不需要奖励）

**标准相机**
- `base_camera`: 环境静态相机（默认 128x128）
- `hand_camera`: 来自 `panda_wristcam` 的腕部相机

**推荐状态向量（统一维度）**
仅使用所有任务通用的 proprio 信息：
- `obs/agent/qpos`
  - Panda: 9D（7 关节 + 2 指尖）
- `obs/agent/qvel`
  - Panda: 9D
- `obs/extra/tcp_pose`
  - 7D pose (xyz + quat)

> 任务特有的 object pose/goal **不要**放进统一 state（可在调试阶段保留，但转换时不使用）。

---

## B) 标准 LeRobot Dataset 结构

我们统一保存以下字段（与 openpi LIBERO 保持一致）：
- `image`: HWC uint8（来自 `base_camera`）
- `wrist_image`: HWC uint8（来自 `hand_camera`，**必须存在**）
- `state`: 1D float32（由 `qpos + qvel + tcp_pose` 拼接）
- `actions`: 7D float32（`pd_ee_delta_pose`）
- `task`: str（作为 prompt；统一从 `DEFAULT_TASK_PROMPT` 获取）

---

## C) StackCube-v2 任务（标准化示例）

本仓库提供 `StackCube-v2`，用于标准化流程验证：
- **base + side** 两路环境相机（可选；训练时只用 base）
- **默认 panda_wristcam**（wrist 图像自动存在）
- 任务 prompt 推荐：`"stack the red cube on the green cube"`

创建方式：
```python
import gymnasium as gym
import maniskill_myws

maniskill_myws.register()
env = gym.make("StackCube-v2", obs_mode="rgb", reward_mode="none", control_mode="pd_ee_delta_pose")
```

---

## D) 轨迹 replay → 标准数据集

你可以用 ManiSkill 官方 replay 工具，把已有轨迹“回放成标准观测格式”：

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /path/to/original_trajectory.h5 \
  --obs-mode rgb \
  --target-control-mode pd_ee_delta_pose \
  --save-traj \
  --render-mode sensors
```

> replay 会输出新的 `.h5 + .json`，其中 obs 已对齐到 `rgb` + `pd_ee_delta_pose`。

---

## E) 转换成 LeRobot dataset（统一格式）

```bash
python /home/sisyphus/Projects/maniskill_myws/scripts/convert_traj_to_lerobot.py \
  --h5-dir "/path/to/replayed_h5_dir" \
  --repo-id "local/maniskill_stackcube_v2" \
  --robot-type "panda" \
  --fps 10 \
  --image-key "obs/sensor_data/base_camera/rgb" \
  --wrist-image-key "obs/sensor_data/hand_camera/rgb" \
  --state-keys "obs/agent/qpos" "obs/agent/qvel" "obs/extra/tcp_pose" \
  --actions-key "actions" \
  --task-from env_default
```

---

## F) 训练提示

- openpi 训练配置中设置 `prompt_from_task=True`
- 优先在 `pi0_base` 上微调，以验证完整管线
