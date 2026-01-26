## 在 `maniskill_myws` 集成 openpi (π0/π0.5) 的工程化方案（推荐：不 fork openpi）

目标：用 ManiSkill 作为 **rollout 平台 + 数据来源**（RecordEpisode `.h5`），把数据转成 openpi 训练使用的 **LeRobot dataset**，并在 ManiSkill 环境中直接加载/运行 π0 策略。

本方案强调三点：
- **可复现**：明确依赖与版本固定方式
- **低耦合**：`openpi` 保持外部依赖（可 editable 开发），ManiSkill 适配代码只放在 `maniskill_myws`
- **与 openpi LIBERO 例子一致**：两路相机 + state + 7D action

---

## 1) 依赖与仓库组织（工程规范）

### 推荐目录与职责
- `src/maniskill_myws/openpi_bridge/`: 只放“桥接层”
  - ManiSkill obs → openpi policy 输入 dict
  - openpi 输出 action chunk → ManiSkill `pd_ee_delta_pose` action（7D）流式执行
- `scripts/`:
  - `inspect_traj_h5.py`: 查看 RecordEpisode `.h5` 的键与 shape（先解决“我的数据到底长什么样”）
  - `convert_traj_to_lerobot.py`: `.h5` → LeRobot dataset（与 openpi LIBERO 示例同 schema）
  - `pi0/serve.py`: π0/π0.5 websocket 推理服务端（建议在 openpi 的 uv 环境中运行）
  - `pi0/run_pi0_remote.py`: ManiSkill 渲染/可视化客户端（连接 websocket server）
  - `pi0/finetune_maniskill.py`: 一键微调（计算 norm stats + 启动 openpi 训练）

### 环境解耦（推荐做法）
建议把 **推理/训练** 和 **仿真/渲染** 分开环境：
- **推理/训练端（openpi uv 环境）**：运行 openpi (JAX/Flax) + checkpoint
- **仿真/渲染端（mani_skill conda 环境）**：运行 ManiSkill + `openpi-client`（轻量 websocket client）

```bash
conda activate mani_skill

# 1) 安装你的 workspace
pip install -e /home/sisyphus/Projects/maniskill_myws[runtime,dev]

# 2) 安装 openpi-client（客户端只需要它，不需要装 openpi/JAX）
pip install -e /home/sisyphus/Projects/maniskill_myws/third_party/openpi/packages/openpi-client
```

> 推理/训练端使用 openpi 的 uv 环境：`cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi && uv sync`

### OpenPI 子模块（可复现）
仓库内已通过 Git submodule 固定 openpi 版本：
```bash
cd /home/sisyphus/Projects/maniskill_myws
git submodule update --init --recursive
```

---

## 2) 数据标准：对齐 openpi 的 LIBERO 训练输入

openpi 的 LIBERO 转换脚本定义的 LeRobot dataset 特征（我们建议保持一致）：
- `image`: (H,W,3) uint8 或 image dtype（第三人称/基座相机）
- `wrist_image`: (H,W,3)（腕部相机，必须存在）
- `state`: 1D float32（建议 8~32 维均可；openpi 会 pad 到 `action_dim`）
- `actions`: 7D float32（对应 `pd_ee_delta_pose`：一般是 6D Δpose + gripper）
- `task`: str（统一从 `DEFAULT_TASK_PROMPT` 获取）

训练时在 openpi config 里设置 `prompt_from_task=True`，即可把 `task` 当作 prompt 使用。

---

## 3) 从 ManiSkill `.h5` 转成 LeRobot dataset（训练数据）

### 3.1 先 inspect 一下你的 `.h5`
```bash
conda activate mani_skill
python /home/sisyphus/Projects/maniskill_myws/scripts/inspect_traj_h5.py \
  --h5 /home/sisyphus/Projects/maniskill_myws/data/demos/debug/20260116_103802.h5
```

### 3.2 转换为 LeRobot dataset（本地）
示例（你需要根据 inspect 输出选择图片与 state 的来源键）。
如果你有**多任务** `.h5`，把不同任务的 `.h5 + .json` 放到同一个目录下，然后用 `--h5-dir` 一次性转换，并用 `--task-from env_default` 自动把 `DEFAULT_TASK_PROMPT` 写入 LeRobot 的 `task` 字段：

```bash
conda activate mani_skill

python /home/sisyphus/Projects/maniskill_myws/scripts/convert_traj_to_lerobot.py \
  --h5-dir "/home/sisyphus/Projects/maniskill_myws/data/demos/multitask_h5" \
  --repo-id "local/maniskill_myws_multitask" \
  --robot-type "panda" \
  --fps 10 \
  --image-key "obs/sensors/base_camera/rgb" \
  --wrist-image-key "obs/sensors/wrist_camera/rgb" \
  --state-keys "obs/extra/tcp_pose" "obs/agent/qpos" \
  --actions-key "actions" \
  --task-from env_default
```

> 说明：键名只是示例，真正的键要以你的 obs_mode/RecordEpisode 输出为准。

---

## 4) 用 openpi 的现有训练 config 直接训练（无需改 openpi 源码）

训练与 norm stats 计算**都在 openpi 的 uv 环境**中完成。推荐使用 myws 的一键脚本：

### 推荐：用 myws 提供的“一键微调脚本”（计算 norm stats + 启动训练）

我们提供了一个脚本 `scripts/pi0/finetune_maniskill.py`，用于在 openpi(uv) 环境中对 ManiSkill 的 LeRobot 数据集进行微调：

```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
uv run python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/finetune_maniskill.py \
  --openpi-root /home/sisyphus/Projects/maniskill_myws/third_party/openpi \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --exp-name ms_pi05_v1 \
  --assets-base-dir /home/sisyphus/Projects/maniskill_myws/assets_openpi \
  --checkpoint-base-dir /home/sisyphus/Projects/maniskill_myws/checkpoints_openpi \
  --overwrite
```

如果你只想先计算 norm stats（不训练）：
```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
uv run python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/finetune_maniskill.py \
  --openpi-root /home/sisyphus/Projects/maniskill_myws/third_party/openpi \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --exp-name ms_pi05_v1 \
  --assets-base-dir /home/sisyphus/Projects/maniskill_myws/assets_openpi \
  --checkpoint-base-dir /home/sisyphus/Projects/maniskill_myws/checkpoints_openpi \
  --only-norm-stats
```

如果你显存不足，优先用 openpi 提供的 LoRA 例子（`pi0_libero_low_mem_finetune`）。

### 4.1 转换是否正确？（离线验证清单）
在“任务表现”之前，先确认**数据格式/管线完全正确**。推荐在 openpi(uv) 环境中做离线验证：

```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
uv run python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/validate_lerobot_dataset.py \
  --openpi-root /home/sisyphus/Projects/maniskill_myws/third_party/openpi \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --assets-base-dir /home/sisyphus/Projects/maniskill_myws/assets_openpi \
  --num-batches 1 \
  --save-images /home/sisyphus/Projects/maniskill_myws/outputs/validate_samples
```

你应该看到：
- observation 里有 `images/*`、`state`（以及 `prompt` 若 `prompt_from_task=True`）
- `actions` 维度是 `(B, horizon, 7)` 或等价结构（取决于 config 的 action 序列键）
- 保存出来的样例图像内容正常（不是全黑/全 0），腕部相机若任务无该视角可能会“缺失/全 0”

---

## 5) 远程推理服务端（策略与 myws 完全解耦）

如果你希望 rollout 时 **策略进程与 ManiSkill 进程完全分离**（依赖隔离/跨机器推理），推荐使用 openpi 自带的 websocket 推理服务端：

### 5.1 服务端（GPU 机器 / policy 环境，openpi uv）
推荐直接用 openpi 的默认 LIBERO expert（π0.5-LIBERO）：

```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
uv run python scripts/serve_policy.py --env LIBERO --port 8000
```

或者用 myws 的 wrapper（支持在 import JAX 前设置 `XLA_FLAGS`）：

```bash
cd /home/sisyphus/Projects/maniskill_myws/third_party/openpi
uv run python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

### 5.2 客户端（你的 myws / ManiSkill 环境）
用 ManiSkill 侧脚本连接服务端（只需 `ws://host:port`）。
注意：`run_pi0_remote.py` **默认不落盘**，只有显式 `--save-images/--save-video/--save-trajectory` 才会保存输出。

```bash
conda activate mani_skill

python /home/sisyphus/Projects/maniskill_myws/scripts/pi0/run_pi0_remote.py \
  --env-id TurnGlobeValve-v1 \
  --server ws://<server_ip>:8000 \
  --obs-mode rgb \
  --image-key sensor_data/base_camera/rgb \
  --wrist-image-key sensor_data/hand_camera/rgb \
  --state-key extra/tcp_pose \
  --render-mode human
```

说明：
- 客户端侧会把图片 **resize_with_pad 到 224x224 并转 uint8**，减少带宽/延迟（与 openpi 推荐一致）
- server 会返回 action chunk；客户端会 open-loop 执行 chunk 里的逐步 action


