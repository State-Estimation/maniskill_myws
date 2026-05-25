# `RecordEpisode` 轨迹录制逻辑详解

## 1. 整体架构

`RecordEpisode` (位于 `mani_skill.utils.wrappers.record`) 是一个 `gym.Wrapper`，录制 episode 时同时写入两个文件：

| 文件 | 内容 |
|---|---|
| `<name>.h5` | HDF5 格式的轨迹数据（actions, obs, env_states, rewards, terminated, truncated, success, fail） |
| `<name>.json` | 轨迹元数据（env_info, episodes 列表，每条含 episode_id, episode_seed, reset_kwargs, control_mode, elapsed_steps） |

## 2. 核心数据流

```
__init__()
  ├─ 创建 .h5 文件 (mode="w")
  ├─ 初始化 _json_data = {env_info, episodes: []}
  └─ _episode_id = -1, _trajectory_buffer = None

reset(seed=seed, save=True)
  ├─ if buffer is not None:
  │     flush_trajectory(save=save)    ← 先把 buffer 里上一集的轨迹写入磁盘
  ├─ super().reset(seed=seed)          ← 然后开始新 episode
  └─ buffer = Step(obs, action, ...)   ← 记录首个观测帧

step(action)
  └─ buffer 中 append (state, obs, action, reward, terminated, truncated, ...)

reset()  ← 再次调用，回到上面的 flush → super().reset() 循环

close()
  ├─ flush_trajectory()                ← 写入最后一条轨迹
  ├─ clean_trajectories()              ← 重编号为 traj_0, traj_1, ... 
  └─ _h5_file.close()
```

## 3. `flush_trajectory(save=True)` 详解

```python
def flush_trajectory(self, ..., save=True):
    flush_count = 0
    for env_idx in env_idxs_to_flush:
        start_ptr = buffer.env_episode_ptr[env_idx]   # 本 episode 在 buffer 中的起始位置
        end_ptr = len(buffer.done)                      # buffer 末尾（本 episode 结束位置）
        if ignore_empty_transition and end_ptr - start_ptr <= 1:
            continue                                    # 空 episode（仅一个 dummy 帧）跳过
        flush_count += 1
        if save:                                        # ← save=False 时整个块跳过
            self._episode_id += 1
            group = h5_file.create_group(f"traj_{episode_id}")  # 写入 HDF5
            # ... actions, terminated, truncated, rewards, env_states, obs, success, fail ...
            json_data["episodes"].append(episode_info)          # 追加 JSON 条目
            dump_json(json_path, json_data)                     # 落盘

    if flush_count > 0:                                 # ← 只依赖 flush_count，不依赖 save
        # 截断 buffer，释放已 flush 数据的内存
        buffer.env_episode_ptr[env_idxs_to_flush] = len(buffer.done) - 1
        buffer = buffer[slice(min_env_ptr, N)]
```

### `save` 参数的作用总结

| | `save=True`（默认） | `save=False` |
|---|---|---|
| `flush_count` | 递增 | 递增（**关键！**） |
| `_episode_id` | 递增 | **不变** |
| h5 写入 | 是 | **否** |
| JSON 追加 | 是 | **否** |
| buffer 截断 | 是 | 是（因为 `flush_count > 0`） |

## 4. `reset()` 中的 `save` 参数

`reset()` 签名为：

```python
def reset(self, *args, seed=None, options=None, save=True, **kwargs):
```

**`save` 是 `reset()` 的直接参数，不是 `options` 里的字段。** 写入 `options` 的键值不会影响 flush 行为，只会原封不动记录到 `reset_kwargs` 中。

**时序上，`save` 作用于刚结束的那条旧轨迹，不是即将开始的新轨迹：**

```
reset(save=False)
  → flush_trajectory(save=False)  # 跳过旧轨迹的写入
  → super().reset(seed=seed)      # 用相同 seed 开始新轨迹
```

## 5. 丢弃轨迹的正确方式

关键约束：轨迹数据在 `reset()` 调用时才从内存 buffer flush 到磁盘。在此之前数据只存在于 `_trajectory_buffer` 中。因此丢弃的时机窗口是 **step 循环内标记 → reset 前触发**。

### 正确做法

```python
# 在 step 循环中检测 d 键
if key_pressed('d'):
    env.reset(seed=current_seed, save=False, options={"reconfigure": True})
    # save=False → 刚结束的这条轨迹不写入 .h5/.json
    # seed 不变 → 用同一个 seed 重新开局
    break  # 跳出当前 episode 循环，继续下一轮
```

### 为什么不会造成 seed/id 错位

```
正常流程:
    reset(seed=0, save=True)  → 录制 E0 → reset(seed=1, save=True)  → flush E0→traj_0(seed=0)
    录制 E1 → reset(seed=2, save=True)  → flush E1→traj_1(seed=1)

丢弃流程:
    录制 E1 → d 键 → reset(seed=1, save=False)  → flush E1 跳过！_episode_id 不变！
    录制 E2(seed=1) → reset(seed=2, save=True)  → flush E2→traj_1(seed=1)

最终结果:
    traj_0 (seed=0), traj_1 (seed=1)  ← 一对一，无错位
```

`save=False` 同时阻止了 `_episode_id` 递增和文件写入，但 buffer 截断正常执行。所以被丢弃的轨迹完全不留痕迹，后续正常轨迹的编号和 seed 保持连续。

### 错误做法

```python
# ❌ 放在 options 里 —— 不生效
env.reset(seed=seed, options={"save_trajectory": False, "reconfigure": True})
# options 内容是纯元数据，不会影响 flush 行为

# ❌ 不传 save=False —— seed 不变但轨迹仍然写入
env.reset(seed=seed, options={"reconfigure": True})
# 上一集被写入，新一集又用相同 seed 录制 → 同一 seed 出现多个 ID

# ❌ 不传 seed —— seed 被 env 自增，无法重录同一 seed
env.reset(save=False, options={"reconfigure": True})
# 跳过了上一集的写入，但 seed 已经变了
```

### 与子类化方案的对比

| | `reset(save=False)` | `DiscardableRecordEpisode` 子类 |
|---|---|---|
| 需手动管理 seed | 是 | 否 |
| 丢弃时机 | `reset()` 时决定 | `step()` 循环内提前标记 |
| 改动源码 | 不需要 | 不需要（子类） |
| 代码量 | 现有 API | 需额外引入子类 |

现有 API 已完全满足需求，不需要子类化。

## 6. HDF5 轨迹文件结构

```
.h5 文件
├── traj_0/              # Group，命名格式 traj_{episode_id}
│   ├── obs/             # [T+1, ...] 观测 (T 步动作对应 T+1 个观测帧)
│   ├── actions/         # [T, action_dim]
│   ├── terminated/      # [T] bool
│   ├── truncated/       # [T] bool
│   ├── rewards/         # [T]
│   ├── env_states/      # [T+1, ...] 可 set_state_dict 回放
│   ├── success/         # [T] bool (可选)
│   └── fail/            # [T] bool (可选)
├── traj_1/
└── ...

.json 文件
{
  "env_info": { "env_id": "...", "env_kwargs": {...}, "max_episode_steps": 600 },
  "episodes": [
    { "episode_id": 0, "episode_seed": 0, "control_mode": "...",
      "elapsed_steps": 100, "reset_kwargs": {...}, "success": false },
    { "episode_id": 1, ... },
  ],
  "source_type": "teleoperation"
}
```

注意：`env_states` 和 `obs` 是 `[T+1]` 帧（多一帧初始状态），`actions` / `rewards` / `terminated` / `truncated` / `success` / `fail` 是 `[T]` 帧。

## 7. `clean_trajectories()` (close 时自动调用)

将 h5 group 和 JSON episodes 重新连续编号：

```python
# 删除空轨迹 (elapsed_steps == 0)
# 重命名 traj_{old_id} → traj_{new_id}
# 同步更新 json_data["episodes"][i]["episode_id"] = new_id
```

所以最终文件中 `episode_id` 始终是 `0, 1, 2, ...` 连续递增，与 JSON 列表下标一致。
