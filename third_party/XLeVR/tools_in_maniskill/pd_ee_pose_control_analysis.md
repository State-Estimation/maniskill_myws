# PD EE Pose 控制器原理与配置指南

## 概述

`pd_ee_pose.py` 实现了笛卡尔空间末端执行器（End-Effector）控制器，核心流程为：

```
action (3D位置 / 6D位姿)
    │
    ▼
compute_target_pose()   ← 计算笛卡尔空间目标位姿
    │
    ▼
kinematics.compute_ik() ← 逆运动学求解关节角度
    │
    ▼
set_drive_targets()     ← 设置关节 PD 目标
    │
    ▼
SAPIEN PhysX PD 驱动    ← 底层物理引擎执行力矩控制
```

文件定义了两个控制器，均继承自 `PDJointPosController`：

| 类 | 自由度 | Action 维度 | 说明 |
|---|---|---|---|
| `PDEEPosController` | 3 (仅位置) | `[dx, dy, dz]` | 旋转保持当前值不变 |
| `PDEEPoseController` | 6 (位置+旋转) | `[dx, dy, dz, drx, dry, drz]` | 旋转用 XYZ 欧拉角，内部转四元数 |

---

## 核心机制

### 1. Delta vs Absolute 模式 (`use_delta`)

- **Delta 模式** (`use_delta=True`, 默认)：action 表示相对于当前末端位姿的**增量**
  - 例如 action `[0.1, 0, 0]` = 从当前位置向指定方向移动 0.1m
- **Absolute 模式** (`use_delta=False`)：action 直接表示基座坐标系下的**绝对目标位姿**

```python
# 源码: compute_target_pose() 中的逻辑
if self.config.use_delta:
    delta_pose = Pose.create(action)
    if self.config.frame == "root_translation":
        target_pose = delta_pose * prev_ee_pose_at_base  # 左乘
    elif self.config.frame == "body_translation":
        target_pose = prev_ee_pose_at_base * delta_pose  # 右乘
else:
    target_pose = Pose.create(action)  # 直接作为绝对位姿
```

### 2. 坐标系选择 (`frame`)

#### 位置控制的坐标系

| frame 值 | 含义 | 示例 |
|---|---|---|
| `root_translation` | 平移在世界/基座坐标系下进行 | `[0.1, 0, 0]` 始终向世界 x 轴移动 |
| `body_translation` | 平移在末端自身坐标系下进行 | `[0.1, 0, 0]` 向末端当前朝向的 x 轴移动 |

#### 旋转控制的坐标系（仅 PDEEPoseController）

| frame 中 rotation 部分 | 含义 | 四元数运算 |
|---|---|---|
| `root_aligned_body_rotation` | 绕世界轴旋转（左乘） | `q_new = delta_q * q_cur` |
| `body_aligned_body_rotation` | 绕自身轴旋转（右乘） | `q_new = q_cur * delta_q` |

完整 frame 选项（`PDEEPoseControllerConfig`）：

- `"root_translation:root_aligned_body_rotation"` （默认）
- `"root_translation:body_aligned_body_rotation"`
- `"body_translation:root_aligned_body_rotation"`
- `"body_translation:body_aligned_body_rotation"`

### 3. `use_target` 机制

控制 action 的**累积基准**：

| `use_target` | 行为 |
|---|---|
| `False` (默认) | 每次 action 基于**当前实际末端位姿**计算目标，更贴近真实遥操作 |
| `True` | 维护虚拟目标位姿 `_target_pose`，action 基于**上次目标位姿**叠加 |

`use_target=True` 的好处：当 IK 无解时不会丢失累积的偏移量。环境 reset 时 `_target_pose` 会被重置为当前末端位姿。

```python
# 源码: set_action() 中的逻辑
if self.config.use_target:
    prev_ee_pose_at_base = self._target_pose      # 用虚拟目标
else:
    prev_ee_pose_at_base = self.ee_pose_at_base    # 用实际位姿
self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
```

### 4. `interpolate` 插值

控制在两个控制步之间的运动方式：

| `interpolate` | 行为 |
|---|---|
| `False` (默认) | 直接将 IK 结果设置为关节驱动目标，交给物理引擎 PD 控制 |
| `True` | 线性插值，将总位移均匀分配到每个物理步 |

```python
# 源码: set_action() 中的逻辑
if self.config.interpolate:
    self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
else:
    self.set_drive_targets(self._target_qpos)

# before_simulation_step() 中逐物理步执行插值
if self.config.interpolate:
    targets = self._start_qpos + self._step_size * self._step
    self.set_drive_targets(targets)
```

### 5. IK 求解器

默认使用 **Levenberg-Marquardt** 算法，通过 `delta_solver_config` 配置：

```python
delta_solver_config: dict = field(
    default_factory=lambda: dict(type="levenberg_marquardt", alpha=1.0)
)
```

- `type`: `"levenberg_marquardt"` 或 `"pseudo_inverse"`
- `alpha`: 缩放 IK 输出的关节增量，控制收敛速度

如果 IK 无解，控制器会退化到当前关节位置 `_start_qpos`，不做任何移动。

### 6. `normalize_action`

- `True` (默认)：将 action 的每个维度归一化到 `[-1, 1]`，再映射到 `[lower, upper]`
- `False`：直接使用原始 action 值

### 7. GPU 仿真限制

GPU 仿真模式下对 frame 有约束：
- `PDEEPosController`：仅支持 `frame == "root_translation"`
- `PDEEPoseController`：仅支持 `frame == "root_translation:root_aligned_body_rotation"`

---

## 底层 PD 控制

最终通过 SAPIEN 的 `set_joint_drive_targets()` 设置每个关节的 PD 目标位置。物理引擎在每个仿真步根据以下参数计算驱动力矩：

```
τ = stiffness * (q_target - q_current) - damping * q_vel
```

相关参数在 Config 中设置：`stiffness`, `damping`, `force_limit`, `friction`。

---