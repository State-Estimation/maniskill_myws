# 自定义机械臂（以 Panda 为例）

本文以 Panda 机械臂为例，讲解在将 ManiSkill 作为库文件使用时，如何通过子类化自定义机械臂，包括继承链分析、控制器配置机制、以及修改 `pd_ee_delta_pose` 控制器的旋转上下限。

---

## 1. 继承链

```
BaseAgent                        mani_skill/agents/base_agent.py
  └── Panda                      mani_skill/agents/robots/panda/panda.py
        ├── PandaWristCam        mani_skill/agents/robots/panda/panda_wristcam.py
        └── PandaStick           mani_skill/agents/robots/panda/panda_stick.py
```

- **BaseAgent**: 定义了 agent 的完整生命周期——加载 URDF/MJCF 模型、初始化控制器（`_controller_configs`）、传感器（`_sensor_configs`）、action/state 管理等。
- **Panda**: 实现了 Panda 机械臂的所有控制器配置、关键帧（keyframes）、关节名称、抓取检测等。
- **PandaWristCam**: 仅比 Panda 多挂了一个腕部 RGB 摄像头，其余完全继承 Panda。

---

## 2. 控制器配置机制

所有控制器的配置通过 `_controller_configs` 这个 **property** 返回。`BaseAgent.__init__` 在初始化时会调用 `self.set_control_mode()`，后者读取 `_controller_configs[self._control_mode]`，根据配置字典实例化 `CombinedController`。

关键代码链路：

1. `BaseAgent.__init__` ([base_agent.py:86-117]) 调用 `set_control_mode()`
2. `set_control_mode` ([base_agent.py:249-288]) 从 `_controller_configs` 读取当前 control_mode 的配置，传给 `CombinedController`
3. `CombinedController` 将每个子控制器的配置（如 `arm_pd_ee_delta_pose`）实例化为对应的 Controller 对象
4. Controller 在 `_initialize_action_space` 中用 `config.rot_lower` / `config.rot_upper` 构建 `spaces.Box` 的 low/high

因此，**修改控制器参数** = **覆盖 `_controller_configs` property 并修改配置 dataclass 上的字段**。

---

## 3. `PDEEPoseControllerConfig` 关键参数

定义在 [pd_ee_pose.py:268-289]。

| 参数 | 类型 | 默认值 | Panda 实际值 | 说明 |
|---|---|---|---|---|
| `pos_lower` | float / list[3] | 必填 | -0.1 | 每步平移量下限 (m) |
| `pos_upper` | float / list[3] | 必填 | 0.1 | 每步平移量上限 (m) |
| `rot_lower` | float / list[3] | `-2π` | **-0.1** | 每步旋转量下限 (rad)，绕 X/Y/Z 轴的欧拉角 |
| `rot_upper` | float / list[3] | `2π` | **0.1** | 每步旋转量上限 (rad)，绕 X/Y/Z 轴的欧拉角 |
| `stiffness` | float / list | None | 1e3 | PD 控制器刚度 |
| `damping` | float / list | None | 1e2 | PD 控制器阻尼 |
| `force_limit` | float / list | 1e10 | 100 | 关节驱动力上限 |
| `use_delta` | bool | True | True | `True` 为增量式控制 |
| `use_target` | bool | False | False | `True` 时维护一个虚拟目标位姿，动作总是相对于该目标 |
| `interpolate` | bool | False | False | 是否插值 |
| `frame` | str | `"root_translation:root_aligned_body_rotation"` | 同默认 | 参考坐标系 |
| `normalize_action` | bool | True | True | 是否归一化到 [-1, 1] |
| `ee_link` | str | 必填 | `"panda_hand_tcp"` | 末端执行器 link 名称 |
| `urdf_path` | str | 必填 | panda v2 URDF 路径 | URDF 文件路径 |

### `rot_lower` / `rot_upper` 的工作方式

在 [pd_ee_pose.py:205-237]，旋转动作的处理流程是：

1. **Action Space 构建**（第 205-222 行）：`spaces.Box(low=[pos_lower×3, rot_lower×3], high=[pos_upper×3, rot_upper×3])`
2. **Clip & Scale**（第 224-237 行）：
   - 位置部分：直接 clip
   - 旋转部分：先按 L2 范数 clip 到 1 以内（防止姿态超球面），**再乘以 `rot_lower`**（注意这里用的是 `rot_lower` 作为 scaling factor，要求 `rot_lower == -rot_upper`）
3. **姿态计算**（第 239-265 行）：旋转量通过 `euler_angles_to_matrix(delta_rot, "XYZ")` 转换为旋转矩阵，再转为四元数

> **注意**：`rot_lower` 的绝对值决定了每步最大旋转幅度。Panda 默认 ±0.1 rad（约 ±5.7°），意味着每步最多旋转约 5.7 度。

---

## 4. Panda `_controller_configs` 完整结构

Panda 在 `_controller_configs` ([panda.py:76-218]) 中定义了所有控制模式。返回的字典结构为：

```python
dict(
    pd_joint_delta_pos = dict(arm=..., gripper=...),
    pd_joint_pos       = dict(arm=..., gripper=...),
    pd_ee_delta_pos    = dict(arm=..., gripper=...),
    pd_ee_delta_pose   = dict(arm=..., gripper=...),   # <-- 我们关心的
    pd_ee_pose         = dict(arm=..., gripper=...),
    pd_joint_vel       = dict(arm=..., gripper=...),
    # ... 等
)
```

其中 `pd_ee_delta_pose` 的 arm 控制器：

```python
arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
    joint_names=self.arm_joint_names,
    pos_lower=-0.1,
    pos_upper=0.1,
    rot_lower=-0.1,
    rot_upper=0.1,
    stiffness=self.arm_stiffness,    # 1e3
    damping=self.arm_damping,        # 1e2
    force_limit=self.arm_force_limit, # 100
    ee_link=self.ee_link_name,       # "panda_hand_tcp"
    urdf_path=self.urdf_path,
)
```

返回前做了 `deepcopy_dict(controller_configs)`，防止用户修改影响原始配置。

---

## 5. 在你的项目中自定义机械臂

以下给出三种常见场景的实现方法。

### 场景一：仅修改 `pd_ee_delta_pose` 的旋转上限

继承 `PandaWristCam`，只覆盖 `_controller_configs`：

```python
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.registration import register_agent


@register_agent()
class PandaWristCamCustomRot(PandaWristCam):
    """Panda 腕部相机机械臂，增大旋转步幅"""

    uid = "panda_wristcam_custom_rot"
    urdf_path = PandaWristCam.urdf_path

    @property
    def _sensor_configs(self):
        return super()._sensor_configs

    @property
    def _controller_configs(self):
        configs = super()._controller_configs

        # 修改 pd_ee_delta_pose
        configs["pd_ee_delta_pose"]["arm"].rot_lower = -0.5
        configs["pd_ee_delta_pose"]["arm"].rot_upper = 0.5

        # 同步修改 target 版本（如果使用）
        configs["pd_ee_target_delta_pose"]["arm"].rot_lower = -0.5
        configs["pd_ee_target_delta_pose"]["arm"].rot_upper = 0.5

        return configs
```

**使用**：

```python
import gymnasium as gym

# 方式 1：通过类引用
env = gym.make("PickCube-v1", robot_uids=PandaWristCamCustomRot)

# 方式 2：通过 uid 字符串（需要 @register_agent）
env = gym.make("PickCube-v1", robot_uids="panda_wristcam_custom_rot")
```

### 场景二：同时修改位置和旋转范围 + 调整 PD 参数

```python
class PandaWristCamAggressive(PandaWristCam):
    uid = "panda_wristcam_aggressive"

    @property
    def _controller_configs(self):
        configs = super()._controller_configs

        arm = configs["pd_ee_delta_pose"]["arm"]
        arm.pos_lower = -0.2
        arm.pos_upper = 0.2
        arm.rot_lower = -1.0
        arm.rot_upper = 1.0
        arm.stiffness = 2000
        arm.damping = 200

        return configs
```

### 场景三：完全自定义，不继承 Panda，从 BaseAgent 构建

```python
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class MyCustomRobot(BaseAgent):
    uid = "my_custom_robot"
    urdf_path = "/path/to/my_robot.urdf"

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
    ee_link_name = "ee_link"

    @property
    def _controller_configs(self):
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.5,     # 自定义旋转范围
            rot_upper=0.5,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        return dict(
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
        )
```

---

## 6. 控制器配置为 dataclass，可直接修改属性

所有 ControllerConfig 都是 `@dataclass`，继承自 `ControllerConfig` ([base_controller.py])。这意味着：

```python
# 直接赋值即可
configs["pd_ee_delta_pose"]["arm"].rot_lower = -1.0

# 也支持 list 形式，分别为 X/Y/Z 轴设置不同范围
configs["pd_ee_delta_pose"]["arm"].rot_lower = [-0.5, -0.5, -0.2]
configs["pd_ee_delta_pose"]["arm"].rot_upper = [0.5, 0.5, 0.2]
```

---

## 7. 使用方式总结

| 使用方式 | 是否需要 `@register_agent` | 示例 |
|---|---|---|
| 直接传入类 | 否 | `gym.make(..., robot_uids=MyPanda)` |
| 通过 uid 字符串 | 是 | `gym.make(..., robot_uids="my_panda")` |
| 显式实例化 | 否 | `MyPanda(scene, control_freq)` |

其中 `BaseEnv._load_agent()` 会检测 `robot_uids` 是字符串还是类——如果是类，直接实例化；如果是字符串，从 `REGISTERED_AGENTS` 中查找。

---

## 8. 关键源码索引

| 文件 | 内容 |
|---|---|
| [base_agent.py](../mani_skill/agents/base_agent.py) | `BaseAgent` 基类，controller 实例化逻辑 |
| [panda.py](../mani_skill/agents/robots/panda/panda.py) | Panda 完整配置，含所有控制模式 |
| [panda_wristcam.py](../mani_skill/agents/robots/panda/panda_wristcam.py) | PandaWristCam，仅添加腕部摄像头 |
| [pd_ee_pose.py](../mani_skill/agents/controllers/pd_ee_pose.py) | `PDEEPoseController` + `PDEEPoseControllerConfig` |
| [base_controller.py](../mani_skill/agents/controllers/base_controller.py) | `ControllerConfig` 基类，`CombinedController` |
| [registration.py](../mani_skill/agents/registration.py) | `@register_agent` 装饰器，`REGISTERED_AGENTS` |
