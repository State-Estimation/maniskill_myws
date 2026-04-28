## 在 `maniskill_myws` 集成 openpi (π0/π0.5) 的工程化方案（推荐：不 fork openpi）

目标：用 ManiSkill 作为 **rollout 平台 + 数据来源**（RecordEpisode `.h5`），把数据转成 openpi 训练使用的 **LeRobot dataset**，并在 ManiSkill 环境中直接加载/运行 π0 策略。

本文默认你当前工作目录在仓库根目录 `maniskill_myws/`。

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
python -m pip install -e .[runtime,dev]

# 2) 安装 openpi-client（客户端只需要它，不需要装 openpi/JAX）
python -m pip install -e third_party/openpi/packages/openpi-client
```

> 推理/训练端使用 openpi 的 uv 环境：`cd third_party/openpi && uv sync`

### OpenPI 子模块（可复现）
仓库内已通过 Git submodule 固定 openpi 版本：
```bash
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
python scripts/inspect_traj_h5.py \
  --h5 data/demos/debug/example.h5
```

### 3.2 转换为 LeRobot dataset（本地）
示例（你需要根据 inspect 输出选择图片与 state 的来源键）。
如果你有**多任务** `.h5`，把不同任务的 `.h5 + .json` 放到同一个目录下，然后用 `--h5-dir` 一次性转换，并用 `--task-from env_default` 自动把 `DEFAULT_TASK_PROMPT` 写入 LeRobot 的 `task` 字段：

```bash
conda activate mani_skill

python scripts/convert_traj_to_lerobot.py \
  --h5-dir "data/demos/multitask_h5" \
  --repo-id "local/maniskill_myws_multitask" \
  --robot-type "panda" \
  --fps 10 \
  --image-key "obs/sensor_data/base_camera/rgb" \
  --wrist-image-key "obs/sensor_data/hand_camera/rgb" \
  --state-keys "obs/agent/qpos" "obs/agent/qvel" "obs/extra/tcp_pose" \
  --actions-key "actions" \
  --task-from env_default
```

> 说明：上面的键名对应当前仓库任务在 `obs_mode=rgb` 时最常见的 RecordEpisode 输出。
> 如果你的 `.h5` 是用 `record_random.py` 默认的 `--obs-mode state_dict` 采的，里面通常不会有 `sensor_data/*/rgb`，需要先 replay 成 `rgb`，或者重新采集。

---

## 4) 用 openpi 的现有训练 config 直接训练（无需改 openpi 源码）

训练与 norm stats 计算**都在 openpi 的 uv 环境**中完成。推荐使用 myws 的一键脚本：

### 推荐：用 myws 提供的“一键微调脚本”（计算 norm stats + 启动训练）

我们提供了一个脚本 `scripts/pi0/finetune_maniskill.py`，用于在 openpi(uv) 环境中对 ManiSkill 的 LeRobot 数据集进行微调：

```bash
cd third_party/openpi
uv run python ../../scripts/pi0/finetune_maniskill.py \
  --openpi-root . \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --exp-name ms_pi05_v1 \
  --assets-base-dir ../../assets_openpi \
  --checkpoint-base-dir ../../checkpoints_openpi \
  --overwrite
```

如果你只想先计算 norm stats（不训练）：
```bash
cd third_party/openpi
uv run python ../../scripts/pi0/finetune_maniskill.py \
  --openpi-root . \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --exp-name ms_pi05_v1 \
  --assets-base-dir ../../assets_openpi \
  --checkpoint-base-dir ../../checkpoints_openpi \
  --only-norm-stats
```

如果你显存不足，优先用 openpi 提供的 LoRA 例子（`pi0_libero_low_mem_finetune`）。

### 4.1 转换是否正确？（离线验证清单）
在“任务表现”之前，先确认**数据格式/管线完全正确**。推荐在 openpi(uv) 环境中做离线验证：

```bash
cd third_party/openpi
uv run python ../../scripts/pi0/validate_lerobot_dataset.py \
  --openpi-root . \
  --config pi05_libero \
  --repo-id local/maniskill_myws_multitask \
  --assets-base-dir ../../assets_openpi \
  --num-batches 1 \
  --save-images ../../outputs/validate_samples
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
cd third_party/openpi
uv run python scripts/serve_policy.py --env LIBERO --port 8000
```

或者用 myws 的 wrapper（支持在 import JAX 前设置 `XLA_FLAGS`）：

```bash
cd third_party/openpi
uv run python ../../scripts/pi0/serve.py \
  --config pi05_libero \
  --checkpoint gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

对于你自己微调出来的 checkpoint，`serve.py` 会优先从 checkpoint 下的
`assets/*/norm_stats.json` 自动推断训练时的 `repo_id/asset_id`，从而保证评测时读取的 norm 统计与训练时一致。
只有当 checkpoint 里存在多份资产，或者你想手动覆盖时，才需要额外传 `--repo-id` 或 `--norm-stats`。

### 5.2 客户端（你的 myws / ManiSkill 环境）
用 ManiSkill 侧脚本连接服务端（只需 `ws://host:port`）。
注意：`run_pi0_remote.py` **默认不落盘**，只有显式 `--save-images/--save-video/--save-trajectory` 才会保存输出。

```bash
conda activate mani_skill

python scripts/pi0/run_pi0_remote.py \
  --env-id TurnGlobeValve-v1 \
  --server ws://<server_ip>:8000 \
  --obs-mode rgb \
  --image-key sensor_data/base_camera/rgb \
  --wrist-image-key sensor_data/hand_camera/rgb \
  --state-keys agent/qpos agent/qvel extra/tcp_pose \
  --render-mode human \
  --visualize-tcp-path
```

说明：
- 客户端侧会把图片 **resize_with_pad 到 224x224 并转 uint8**，减少带宽/延迟（与 openpi 推荐一致）
- server 会返回 action chunk；客户端会 open-loop 执行 chunk 里的逐步 action
- `--visualize-tcp-path` 会在实时 viewer 中绘制当前 action chunk：蓝色是模型预测的未来 TCP chunk，橙色是实际执行后的 TCP 轨迹采样。marker 只在 `env.render()` 时显示，不会进入策略输入图像。

### 5.3 Multi-seed 评测与实时渲染

如果你想一次评测多个 seed，可以用：

```bash
conda activate mani_skill

python scripts/pi0/run_pi0_remote_multi_seed.py \
  --env-id OpenSafeDoor-v2 \
  --server ws://<server_ip>:8000 \
  --num-seeds 20 \
  --start-seed 0 \
  --render-mode human \
  --visualize-tcp-path \
  --base-chunk-max-actions 16 \
  --base-chunk-position-scale 0.1 \
  --save-videos \
  --video-views both
```

说明：
- `--render-mode human`：打开 ManiSkill 实时渲染窗口
- `--save-videos`：为每个 seed 保存 mp4
- `--video-views base|wrist|both`：选择输出 `base_camera`、`hand_camera` 或两者都保存
- `--image-key` / `--wrist-image-key`：覆盖默认观测键；默认分别是 `sensor_data/base_camera/rgb` 和 `sensor_data/hand_camera/rgb`
- `--image-every`：每隔多少步保存一帧（同时影响图片和视频采样密度）
- `--base-chunk-max-actions` / `--base-chunk-position-scale`：控制蓝色 chunk 的长度和 action 到 TCP 位移的投影比例
- `--base-path-color` / `--residual-path-color` / `--path-radius`：控制 viewer 中 marker 的颜色和大小

注意：
- `video-views` 只决定保存哪个观测流，不会改变任务里相机本身的位姿
- 如果你修改了任务中的 `base_camera`，那么 `base` 视频会自动使用新的相机视角
- chunk marker 为了避免影响视觉输入，只显示在实时 `render()` 窗口中；保存的 obs 视频默认保持干净
