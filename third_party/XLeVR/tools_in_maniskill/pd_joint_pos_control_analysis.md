# PDJointPosController 控制逻辑分析

核心文件：
- [pd_joint_pos.py](mani_skill/agents/controllers/pd_joint_pos.py)
- [base_controller.py](mani_skill/agents/controllers/base_controller.py)

---

## 1. 初始化 (`__init__` → 继承自 `BaseController`)

```
BaseController.__init__()
├── _initialize_joints()              # 解析 joint_names，获取 active joints
├── _initialize_action_space()        # 基于 joint limits 创建动作空间 Box
├── _clip_and_scale_action_space()    # 如果 normalize_action=True，将 action space 归一化到 [-1, 1]
└── 设置 self.action_space（单环境 or 批量化）
```

`_sim_steps = sim_freq // control_freq` 决定了每个控制周期内执行多少次物理步进。

---

## 2. 动作处理核心：`set_action(action)`

[pd_joint_pos.py:76-93](mani_skill/agents/controllers/pd_joint_pos.py#L76-L93)

```
set_action(action)
│
├── _preprocess_action(action)
│   ├── 检查维度 (num_envs, action_dim)
│   └── 若 normalize_action=True: 反归一化到原始动作空间
│
├── _step = 0                          # 重置插值计数器
├── _start_qpos = self.qpos            # 记录起始关节位置
│
├── 计算 _target_qpos:
│   ├── use_delta=True, use_target=True  → _target_qpos += action (累积式)
│   ├── use_delta=True, use_target=False → _target_qpos = _start_qpos + action (增量式)
│   └── use_delta=False                  → _target_qpos = broadcast(action) (绝对值)
│
└── 若 interpolate=True:  计算 _step_size = (_target_qpos - _start_qpos) / _sim_steps
    若 interpolate=False: 直接 set_drive_targets(_target_qpos)
```

---

## 3. 仿真步进：`before_simulation_step()`

[pd_joint_pos.py:95-101](mani_skill/agents/controllers/pd_joint_pos.py#L95-L101)

```
每个 sim step 调用一次:
├── _step += 1
└── 若 interpolate=True:
    targets = _start_qpos + _step_size * _step  # 线性插值
    set_drive_targets(targets)                    # 更新 PD 目标
```

插值使得目标位置在控制周期内平滑过渡，而非阶跃跳变。

---

## 4. PD 驱动属性设置：`set_drive_property()`

[pd_joint_pos.py:38-52](mani_skill/agents/controllers/pd_joint_pos.py#L38-L52)

```
对每个 joint 设置:
├── stiffness (刚度 Kp)
├── damping   (阻尼 Kd)
├── force_limit
├── friction
└── drive_mode (默认 "force")
```

底层调用 SAPIEN 的 `joint.set_drive_properties()`，物理引擎在每个仿真步中计算：

```
τ = Kp * (q_target - q_current) + Kd * (0 - q_vel_current)
```

---

## 5. 重置：`reset()`

[pd_joint_pos.py:54-69](mani_skill/agents/controllers/pd_joint_pos.py#L54-L69)

```
reset()
├── _step = 0
├── 首次调用: _start_qpos, _target_qpos = current_qpos
└── 后续调用: 仅更新 _reset_mask=True 的环境的 _start_qpos, _target_qpos
    （支持并行环境的部分重置）
```

---

## 6. `PDJointPosMimicController`（派生类）

[pd_joint_pos.py:129-258](mani_skill/agents/controllers/pd_joint_pos.py#L129-L258)

用于模拟 **随动关节**（mimic joints）。核心差异：

- **action space 缩小**：仅包含直接控制的关节（`control_joint_indices`）
- **mimic 关节跟随**：`q_mimic = q_control * multiplier + offset`
- `set_action()` 中只对 `control_joint_indices` 写入动作，然后自动更新随动关节目标

---

## 关键配置参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `stiffness` | (无) | PD 控制器比例增益 Kp |
| `damping` | (无) | PD 控制器微分增益 Kd |
| `force_limit` | 1e10 | 关节驱动力上限 |
| `use_delta` | False | True=增量控制, False=绝对位置控制 |
| `use_target` | False | True=累积增量到上一次目标, False=基于当前 qpos 增量 |
| `interpolate` | False | True=在控制周期内线性插值目标位置 |
| `normalize_action` | True | True=将 action space 归一化到 [-1,1] |
| `drive_mode` | "force" | 驱动模式（force/acceleration/velocity） |
