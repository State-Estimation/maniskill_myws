# maniskill_myws

自定义 ManiSkill 任务 + openpi(π0/π0.5) 训练/推理工作区。

## 当前 RL 主线：Mean Frozen-Latent Advantage-BC

当前 `frozen-latent-advantage-bc` 分支只维护一套 RL post-training 方案：

1. 冻结 π0 的全部参数。π0 每个 chunk 输出 50 步 reference action，并返回最后一次去噪中 action-suffix hidden token 的 1024 维均值。
2. 轻量 actor 读取机器人状态、mean frozen latent、reference action 和当前时间位置，预测 6 个 knot；插值后得到完整的 50 步 residual action chunk。
3. Conservative twin-Q 比较 residual 与 zero-residual 的价值。只有 advantage 达到阈值且满足 intervention budget/cooldown 时才执行 residual，否则保持原始 π0 action。
4. 成功执行且 Q advantage 足够高的 residual 会进入 Advantage-BC 自模仿项；replay 按 zero residual、成功 nonzero residual、失败 nonzero residual 进行平衡采样。

该方案没有独立 vision encoder、temporal latent、失败预测器、candidate ensemble、PCA 或 task-stage router。视觉信息只来自冻结 π0 的 action-suffix latent；RL 侧从零初始化，不读取任何旧 RL checkpoint 或 replay。这里的“从零训练”不包括 π0，π0 仍使用预训练 checkpoint，但训练期间始终冻结。

当前 `SolarPanelStatic-v2` 奖励协议包含一次稳定夹取事件奖励 `0.25` 和任务成功奖励 `1.0`。即使命令使用 `--reward-mode sparse`，训练也不是只有 terminal success reward；checkpoint 会绑定该奖励协议并在推理时校验。

### 已验证结果

以下结果均为独立的 RL 从零训练，并使用 100 个 paired seeds 比较相同 π0 Base 与 RLT：

| RL 训练 | 评测 seeds | Base | RLT | rescue | regression |
| --- | --- | ---: | ---: | ---: | ---: |
| seed 11000，150k env steps | 8100-8199 | 66% | 87% | 21 | 0 |
| seed 15000，200k env steps | 69000-69099 | 65% | 90% | 26 | 1 |

当前最新 checkpoint：

```text
outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k_wandb/frozen_latent_residual.pt
```

这是当前工作区的本地路径。`outputs/` 被 Git 忽略，权重不会随仓库 clone 自动获得；在其他机器上需要先同步该 run 的完整目录，或将下面命令中的 checkpoint 路径替换为实际位置。

### 1. 启动 Frozen-Latent π0 服务

服务端必须带 `--frozen-action-latent`，否则不会返回 RL 所需的 mean latent：

```bash
cd third_party/openpi
CUDA_VISIBLE_DEVICES=0 uv run python ../../scripts/pi0/serve.py \
  --config pi0_maniskill \
  --checkpoint ../../checkpoints_openpi/pi0_maniskill/ms_pi0_maniskill_SolarPanelStatic-v20_pd_joint_pos_success_traj_32batch/29999 \
  --port 8011 \
  --xla-safe \
  --frozen-action-latent
```

下面的训练和评测命令均从仓库根目录运行，并假设 ManiSkill 环境名为 `mani_skill`。

### 2. 从零训练复现配置

不要传入任何 `--resume-*`、`--initial-replay` 或旧 RL checkpoint。输出目录必须是新的目录：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/train_frozen_latent_residual.py \
  --env-id SolarPanelStatic-v2 \
  --obs-mode rgb \
  --reward-mode sparse \
  --control-mode pd_joint_pos \
  --sim-backend physx_cpu \
  --render-backend sapien_cuda:0 \
  --enhanced-determinism \
  --server ws://127.0.0.1:8011 \
  --device cuda:0 \
  --seed 15000 \
  --chunk-len 50 \
  --max-episode-steps 500 \
  --total-env-steps 200000 \
  --buffer-capacity 50000 \
  --batch-size 128 \
  --updates-per-chunk 2 \
  --critic-warmup-updates 2000 \
  --fixed-std 0.10 \
  --online-explore-probability 0.35 \
  --independent-online-exploration \
  --exploration-std-start 0.30 \
  --exploration-std-end 0.30 \
  --base-control-episode-probability 0.20 \
  --replay-nonzero-fraction 0.50 \
  --replay-nonzero-success-fraction 0.50 \
  --actor-success-bc-weight 2.0 \
  --actor-success-bc-min-q-advantage 0.05 \
  --conservative-q-weight 0.10 \
  --min-q-advantage 0.08 \
  --max-online-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 \
  --wandb-enabled \
  --wandb-new-run \
  --wandb-project maniskill-frozen-latent-td3 \
  --wandb-run-name frozen_mean_advantage_bc_seed15000_scratch200k \
  --wandb-tags frozen-latent advantage-bc scratch replication \
  --output-dir outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k
```

训练目录会生成以下可恢复快照：

- `frozen_latent_residual.pt`：actor、critics、优化器、模型配置与 runtime identity。
- `online_replay.npz`：精确执行 action 和 frozen latent replay。
- `history.jsonl`：episode/训练指标。
- `trainer_state.json`：训练进度、RNG、warmup 统计与 snapshot generation。
- `run_config.json`：完整命令配置和奖励、表示、intervention 协议。

### 3. 精确断点续训

四个恢复文件必须来自同一个 snapshot generation；不能只加载 `.pt` 后继续训练。复制上一节的完整训练命令，保留其余参数，仅将累计目标和输出目录替换为新值，并追加以下参数：

```text
--total-env-steps 250000
--resume-checkpoint <old-run>/frozen_latent_residual.pt
--resume-replay <old-run>/online_replay.npz
--resume-history <old-run>/history.jsonl
--resume-trainer-state <old-run>/trainer_state.json
--output-dir <new-run>
```

### 4. Paired 推理与批量评测

下面命令复现最新 checkpoint 的 100-seed 评测。部署 Q-gate 目前没有写入 checkpoint，因此 `0.08/2/1` 三项不能省略：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/eval_frozen_latent_residual.py \
  --checkpoint outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k_wandb/frozen_latent_residual.pt \
  --server ws://127.0.0.1:8011 \
  --start-seed 69000 \
  --num-seeds 100 \
  --min-q-advantage 0.08 \
  --max-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 \
  --output-dir outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k_wandb/eval_reproduction
```

评测器依次运行同 seed 的 Base 与 RLT，输出 `paired_results.jsonl` 和 `summary.json`。它会校验环境、动作空间、奖励协议、π0 checkpoint identity 和 frozen-latent protocol，并拒绝向已有评测结果目录追加数据。

### 5. 实时双轨迹渲染

`seed=69090` 在最新评测中是 Base 失败、RLT 成功，适合观察第一个 action chunk 的 residual 修正：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/eval_frozen_latent_residual.py \
  --checkpoint outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k_wandb/frozen_latent_residual.pt \
  --server ws://127.0.0.1:8011 \
  --render-mode human \
  --real-time \
  --live-paired-trajectories \
  --seeds 69090 \
  --min-q-advantage 0.08 \
  --max-intervention-chunks-per-episode 2 \
  --intervention-cooldown-chunks 1 \
  --output-dir outputs/rlt/SolarPanelStatic-v2_advantage_bc_seed15000_scratch200k_wandb/realtime_seed69090
```

蓝线表示 Base TCP，橙线表示 RLT TCP，两套环境以同 seed lockstep 执行。轨迹默认每 5 步更新一次，避免每帧重建完整 line set；`--real-time` 按环境控制频率节流。关闭窗口会中止本次 rollout，重复运行时需要更换 `--output-dir`。

更完整的实现约束见 [`docs/rlt_post_training.md`](docs/rlt_post_training.md)。

## ActProbe action-space failure detector

ActProbe 是与 frozen-latent RL 解耦的 shadow detector。它不读取视觉或 latent，
只使用 OpenPI 每次推理返回的完整 action horizon 计算两个官方特征：

- `ACM/action_norm`：即将执行的前 `chunk_len` 步 action 的 RMS。
- `TCE/chunk_mse`：上一预测去掉已执行前缀后，与当前预测重叠 horizon 的 MSE。

本适配直接导入 `third_party/actprobe` 的官方 `ActProbeNet`，固定官方 commit
`d5dfbcc98ee5e5766f8aa548c657d3446e40e272`。本地代码只负责 ManiSkill rollout、
数据协议和可与 SAFE 对齐的划分/评测，不复制官方网络。单任务 detector 中 prompt
恒定，不包含分类信息，因此使用固定 1024 维零 embedding；这保留官方
LSTM-MLP 结构，同时避免下载 Qwen3。多任务训练时必须改为官方 Qwen embedding。

初始化依赖：

```bash
git submodule update --init --recursive
conda create -n actprobe python=3.10
conda activate actprobe
pip install -r requirements-actprobe.txt
pip install -e .
```

ActProbe 不要求 OpenPI 返回特殊 latent。启动普通 OpenPI 服务即可：

```bash
cd third_party/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -u ../../scripts/pi0/serve.py \
  --config pi0_maniskill \
  --checkpoint ../../checkpoints_openpi/pi0_maniskill/ms_pi0_maniskill_TakeSafetyHook1_pd_joint_pos_success_truncated_32batch/29999 \
  --port 8012 --xla-safe
```

从仓库根目录采集 base-policy action feature。执行 chunk 为 10，但始终保存并使用
π0 原始 50 步预测计算 TCE；环境实际执行的 action 仍按现有 action-space projection
处理。数据会绑定任务源码、OpenPI checkpoint 内容和两种 horizon：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mani_skill \
python -u scripts/rlt/collect_actprobe_rollouts.py \
  --env-id TakeSafetyHook-v1 \
  --server ws://127.0.0.1:8012 \
  --sim-backend physx_cpu --render-backend sapien_cuda:0 \
  --start-seed 61000 --num-episodes 600 \
  --chunk-len 10 --max-episode-steps 500 \
  --output outputs/actprobe/TakeSafetyHook-v1_chunk10_seed61000/rollouts.npz
```

训练直接使用官方约 23K 参数 ActProbe 网络、逐时刻 episode label BCE、AdamW、
cosine schedule、400 epoch 上限与 50 epoch early stopping。每个 seed 都从零初始化；
归一化统计只由 train split 拟合：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n actprobe \
python -u scripts/rlt/train_official_actprobe.py \
  --dataset outputs/actprobe/TakeSafetyHook-v1_chunk10_seed61000/rollouts.npz \
  --output-dir outputs/actprobe/TakeSafetyHook-v1_chunk10_seed61000/checkpoints \
  --seeds 0 1 2 --device cuda:0
```

离线评测同时报告官方 `q=0.25 taskmax AUC`，以及与 SAFE 相同的 full-episode
ROC-AUC/AP 和由 `val_seen` 成功轨迹校准到目标 5% episode FPR 后的 Recall/FPR。
`val_unseen` 只用于最终测试，不参与 checkpoint 或阈值选择：

```bash
conda run --no-capture-output -n actprobe \
python -u scripts/rlt/eval_official_actprobe.py \
  --checkpoints outputs/actprobe/TakeSafetyHook-v1_chunk10_seed61000/checkpoints/seed*/actprobe.pt \
  --device cpu \
  --output outputs/actprobe/TakeSafetyHook-v1_chunk10_seed61000/eval_aggregate.json
```

与 SAFE 正式比较时，两者必须使用相同任务源码、OpenPI checkpoint、episode seeds、
执行 chunk 和最大步数分别采集。两套 detector 使用相同的成功/失败分层
train/`val_seen`/`val_unseen` 算法和 seed；ActProbe 采集不应复用 SAFE `.npz`，因为
SAFE 数据只保存 latent，没有完整 action horizon，无法事后恢复 TCE。

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
- `TakeSafetyHook-v2`
- `SolarPanelStatic-v1`
- `SolarPanelStatic-v2`
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
- `docs/rlt_post_training.md`

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
  --config pi0_maniskill \
  --checkpoint ../../checkpoints_openpi/pi0_maniskill/ms_pi0_maniskill_v1/<step> \
  --port 8000
```

ManiSkill 训练/部署请使用 `pi0_maniskill` 或 `pi05_maniskill`；`pi0_libero` / `pi05_libero` 只保留给 openpi 官方 LIBERO 配置。`serve.py` 会优先从 checkpoint 下的 `assets/*/norm_stats.json`
自动推断训练时使用的 `repo_id/asset_id`；只有在 checkpoint 里有多份资产、或者你想强制覆盖时，才需要额外传
`--repo-id` 或 `--norm-stats`。

历史 checkpoint 如果仍保存在 `checkpoints_openpi/pi0_libero/...` 目录下，也请按训练时的动作语义选择 `--config pi0_maniskill` 或 `--config pi0_maniskill_legacy`；目录名不决定推理 transform。

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
- Frozen-latent residual RL：`scripts/rlt/train_frozen_latent_residual.py`
- Paired RL 评测与实时双轨迹：`scripts/rlt/eval_frozen_latent_residual.py`

> 这些脚本的完整参数与示例见 `docs/openpi_integration.md`。
