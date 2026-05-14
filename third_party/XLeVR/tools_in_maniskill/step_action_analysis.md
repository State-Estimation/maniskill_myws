# `_step_action` 动作处理逻辑详解

## 1. 整体流程概览

```
env.step(action)
  └─ _step_action(action)               # 解析/校验 action 类型，驱动仿真
       ├─ 判断 action 类型（None / Tensor / ndarray / dict）
       ├─ 可选：动态切换 control_mode
       ├─ agent.set_action(action)       # 交由控制器设置驱动目标
       ├─ GPU 刷新驱动目标到 PhysX
       └─ 执行 sim_steps_per_control 次物理步进
```

## 2. 核心问题：step() 怎么知道用哪种控制模式？

### 2.1 控制模式不是在 `step()` 时推断的，而是在 `gym.make()` 时就确定了

```python
# 控制模式在这里就确定了！
env = gym.make("PickCube-v1", control_mode="pd_joint_delta_pos")

# 之后 step 时，仿真器早就知道该用什么模式来解释你的 action
obs, reward, term, trunc, info = env.step(action)
```

仿真器**不是**在 `step()` 时根据 action 的形状去"猜测"你想要哪种控制模式，而是 `step()` 之前 `env.agent.controller` 已经指向了一个确定的控制器实例。该实例的内部逻辑（`set_action` 方法）知道如何解释传入的数组。

打个比方：微波炉的操作面板在出厂时就定了。你按"启动"时传入一个数字（时间），微波炉不需要猜测你要用什么模式——你已经在按启动之前通过按"微波"/"烧烤"/"解冻"按钮选好了。`step(action)` 里的 `action` 只是那个数字，模式已经在 `gym.make()` 时选好了。

### 2.2 如何确定应该传入什么形式的 action

```python
# 1. 直接看 action_space 的 shape 和 bounds
print(env.action_space)               # Box(low=[...], high=[...])
print(env.single_action_space.shape)  # e.g. (7,) for 7-DOF robot

# 2. 用 action_space.sample() 看样例
sample = env.action_space.sample()
print(sample.shape)   # (num_envs, action_dim)

# 3. 看控制模式名字推断含义
print(env.control_mode)  # "pd_joint_delta_pos" → 增量关节位置
```

不同控制模式决定了 action 的含义和形状：

| 控制模式 | action 含义 | shape |
|---|---|---|
| `pd_joint_pos` | 目标关节角度（绝对值，弧度） | `(n_joints,)` |
| `pd_joint_delta_pos` | 关节角度增量（弧度） | `(n_joints,)` |
| `pd_joint_vel` | 目标关节速度（弧度/秒） | `(n_joints,)` |
| `pd_ee_delta_pos` | 末端位移 `[dx, dy, dz]` | `(3,)` |
| `pd_ee_delta_pose` | 末端位姿增量 `[dx, dy, dz, dr, dp, dy]` | `(6,)` |

**关键点**：`gym.make()` 时 `control_mode` 参数决定了哪个控制器被激活，控制器内部定义了 `action_space`（shape + bounds）和 `set_action()` 的解释逻辑。所以不需要在 `step()` 时告诉仿真器用什么模式——它早就知道了。

## 3. `_step_action` 对 action 类型的判断逻辑

源码位于 [sapien_env.py:1073-1132](mani_skill/envs/sapien_env.py#L1073-L1132)，核心是一个 `if-elif-else` 分支：

| action 类型 | 条件 | 行为 |
|---|---|---|
| **`None`** | `action is None` | 不做任何控制，纯物理仿真前进一步 |
| **`np.ndarray` / `torch.Tensor`** | `isinstance(action, ...)` | 转为 tensor，判断是否 unbatched（shape 与 `single_action_space` 一致），标记 `set_action=True` |
| **`dict` 含 `"control_mode"` 键** | `"control_mode" in action` | 若控制模式与当前不同，则切换控制器并 reset；从 `action["action"]` 提取实际动作 |
| **`dict` 不含 `"control_mode"` 键** | else | 认为是多智能体（MultiAgent）动作，key 为智能体名，value 为各自动作 |
| **其他类型** | else | 抛出 `TypeError` |

### 3.1 `None` —— 纯物理仿真

```python
if action is None:
    pass  # 不发送控制信号，仅执行物理步进
```

适用场景：不施加控制，仅观察物体在重力/摩擦力作用下的自由运动。

### 3.2 `np.ndarray` / `torch.Tensor` —— 标准控制

```python
elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
    action = common.to_tensor(action, device=self.device)
    if action.shape == self._orig_single_action_space.shape:
        action_is_unbatched = True    # 表示传入的是单个环境的动作
    set_action = True
```

当 `num_envs == 1` 时，可以传入未批处理的动作（shape 为 `(action_dim,)`），方法会自动调用 `common.batch(action)` 扩展为 `(1, action_dim)`。

### 3.3 `dict` 含 `"control_mode"` —— 动态切换控制模式

```python
elif isinstance(action, dict):
    if "control_mode" in action:
        if action["control_mode"] != self.agent.control_mode:
            self.agent.set_control_mode(action["control_mode"])
            self.agent.controller.reset()
        action = common.to_tensor(action["action"], device=self.device)
```

**用法示例**：在同一个 episode 内，可以从位置控制切换到速度控制或 EE 控制：
```python
env.step({"control_mode": "pd_joint_pos", "action": joint_pos_action})
env.step({"control_mode": "pd_ee_delta_pos", "action": ee_delta_action})
```

### 3.4 `dict` 不含 `"control_mode"` —— 多智能体

```python
else:
    assert isinstance(self.agent, MultiAgent)
    action = common.to_tensor(action, device=self.device)
    # action 是一个 dict，key 为 agent 名称
```

## 4. 控制模式与动作空间的绑定

### 4.1 控制模式 → 控制器 → 动作空间

链路如下：

```
control_mode 字符串
  └─ agent._controller_configs[control_mode]
       └─ 实例化对应 Controller
            └─ controller.action_space        # gymnasium.spaces.Box
            └─ controller.set_action(action)  # 将 action 值解释为驱动目标
```

### 4.2 `_controller_configs` 如何定义

每个机器人类覆盖 `_controller_configs` 属性，将控制模式名映射到控制器配置。两种形式：

**简单机器人（如 SO-100）**—— 一个控制器控制全部关节：
```python
# mani_skill/agents/robots/so100/so_100.py
controller_configs = dict(
    pd_joint_delta_pos=pd_joint_delta_pos,   # PDJointPosController, use_delta=True
    pd_joint_pos=pd_joint_pos,               # PDJointPosController, absolute
)
```

**复杂机器人（如 Fetch）**—— 多个子控制器组合（CombinedController）：
```python
# mani_skill/agents/robots/fetch/fetch.py
controller_configs = dict(
    pd_joint_delta_pos=dict(
        arm=ArmControllerConfig(...),        # PDJointPosController
        gripper=GripperControllerConfig(...), # PDJointPosController
        body=BodyControllerConfig(...),       # PDJointPosController
    ),
    pd_ee_delta_pos=dict(
        arm=ArmEEControllerConfig(...),       # PDEEPosController
        gripper=...,
    ),
)
```

### 4.3 各控制器定义的 action 含义

| 控制器 | 动作空间 shape | 动作含义 | `sets_target_qpos` | `sets_target_qvel` |
|---|---|---|---|---|
| `PDJointPosController` | `(n_joints,)` | 目标关节角度（absolute 或 delta） | ✅ | ❌ |
| `PDJointVelController` | `(n_joints,)` | 目标关节速度 | ❌ | ✅ |
| `PDJointPosVelController` | `(2*n_joints,)` | 前 n 个为目标位置，后 n 个为目标速度 | ✅ | ✅ |
| `PDEEPosController` | `(3,)` | EE 位移 `[dx, dy, dz]` （或绝对位置） | ✅ | ❌ |
| `PDEEPoseController` | `(6,)` | `[dx, dy, dz, droll, dpitch, dyaw]` | ✅ | ❌ |
| `PassiveController` | `(0,)` | 无动作 | ❌ | ❌ |
| `PDBaseVelController` | `(3,)` | `[vx, vy, omega]` 本体速度 | ❌ | ✅ |

### 4.4 `set_action` 内部如何解释 action

以 `PDJointPosController` 为例（[pd_joint_pos.py:76-93](mani_skill/agents/controllers/pd_joint_pos.py#L76-L93)）：

```python
def set_action(self, action):
    action = self._preprocess_action(action)    # shape 校验, 可选 denormalize
    self._start_qpos = self.qpos                # 记录当前关节位置
    if self.config.use_delta:
        # action 是增量: target = current + delta
        self._target_qpos = self._start_qpos + action
    else:
        # action 是绝对目标角度
        self._target_qpos = action
    if self.config.interpolate:
        self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
    else:
        self.set_drive_targets(self._target_qpos)  # 直接设置驱动目标
```

**关键配置选项**：
- `use_delta=True`：action 是对当前位置的偏移，而非绝对目标
- `interpolate=True`：目标位置会被插值到多个物理子步中平滑过渡
- `normalize_action=True`：动作空间为 `Box(-1, 1, ...)`，内部自动缩放到关节限位

### 4.5 从 env 查询当前动作空间

```python
# 查看当前控制模式
env.control_mode                         # "pd_joint_delta_pos"

# 查看动作空间的上下界
env.action_space                         # Box(low=[...], high=[...])
env.single_action_space                  # 单个环境的 action space

# 查看当前控制器的配置
env.agent.controller.config              # ControllerConfig 对象
```

## 5. 如何使用 `env.step()`

### 5.1 基本用法（最简单）

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1", num_envs=1, control_mode="pd_joint_delta_pos")
obs, _ = env.reset()

# 方式 1：直接传 tensor/ndarray（shape 必须匹配 single_action_space）
action = env.action_space.sample()       # shape: (1, action_dim) for num_envs=1, batched
obs, reward, terminated, truncated, info = env.step(action)

# 方式 2：单环境可传 unbatched action
action = env.single_action_space.sample()  # shape: (action_dim,)
obs, reward, terminated, truncated, info = env.step(action)
```

### 5.2 无控制仿真

```python
# action=None —— 不发送控制信号，纯物理仿真
obs, reward, terminated, truncated, info = env.step(None)
```

### 5.3 动态切换控制模式

```python
# 先用关节位置控制
obs, _, _, _, _ = env.step({"control_mode": "pd_joint_pos", "action": joint_pos})

# 切换到 EE delta 控制
obs, _, _, _, _ = env.step({"control_mode": "pd_ee_delta_pos", "action": ee_delta})
```

### 5.4 多环境并行（num_envs > 1）

```python
env = gym.make("PickCube-v1", num_envs=4, control_mode="pd_joint_delta_pos")
obs, _ = env.reset()

# action 必须批处理
action = env.action_space.sample()       # shape: (4, action_dim)
obs, reward, terminated, truncated, info = env.step(action)

# 对某些环境施加零动作
action = torch.zeros(4, env.single_action_space.shape[0])
obs, reward, terminated, truncated, info = env.step(action)
```

### 5.5 多智能体

```python
# num_envs=1 时可以传 dict
action = {"agent_0": agent0_action, "agent_1": agent1_action}
obs, reward, terminated, truncated, info = env.step(action)
```

### 5.6 返回值说明

```python
obs, reward, terminated, truncated, info = env.step(action)

# obs: 观测（根据 obs_mode 不同格式不同）
#   - "state": 扁平 1D tensor
#   - "state_dict": dict{"agent": {"qpos": ..., "qvel": ...}, "extra": ...}
#   - "rgb" / "rgbd" / "pointcloud": 含 sensor_data 的 dict

# reward: tensor (num_envs,)
# terminated: tensor (num_envs,), bool
# truncated: tensor (num_envs,), bool （ManiSkill 始终为 False）
# info: dict, 包含 "success" / "fail" 等任务相关键
```

### 5.7 常用控制模式命名约定

| 模式名 | 含义 |
|---|---|
| `pd_joint_pos` | PD 关节位置（绝对） |
| `pd_joint_delta_pos` | PD 关节位置（增量） |
| `pd_joint_target_delta_pos` | PD 关节位置（增量，相对上一帧目标） |
| `pd_joint_vel` | PD 关节速度 |
| `pd_ee_delta_pos` | PD 末端位置（增量） |
| `pd_ee_delta_pose` | PD 末端位姿（增量，含旋转） |
| `pd_ee_pose` | PD 末端位姿（绝对） |
| `passive` | 无力控制 |

## 6. 完整链路总结

```
用户调用 env.step(action)
  │
  ├─ _step_action 判断 action 类型
  │   ├─ None → 跳过控制
  │   ├─ Tensor/ndarray → 使用当前 control_mode
  │   └─ dict{"control_mode": ..., "action": ...} → 先切换控制模式
  │
  ├─ agent.set_action(action)
  │   └─ controller.set_action(action)
  │       ├─ _preprocess_action()    校验 shape, 可选 denormalize
  │       └─ set_drive_targets()     设置关节驱动目标到 PhysX
  │
  ├─ GPU: gpu_apply_articulation_target_position/velocity
  │
  └─ for _ in range(sim_steps_per_control):
       ├─ controller.before_simulation_step()   插值步进
       └─ scene.step()                          PhysX 物理步进
```

## 7. 关键源码索引

| 文件 | 行号 | 内容 |
|---|---|---|
| [sapien_env.py](mani_skill/envs/sapien_env.py) | 1042-1071 | `step()` 方法 |
| [sapien_env.py](mani_skill/envs/sapien_env.py) | 1073-1132 | `_step_action()` 方法 |
| [sapien_env.py](mani_skill/envs/sapien_env.py) | 334-338 | `action_space` 从 agent 继承 |
| [base_agent.py](mani_skill/agents/base_agent.py) | 245-296 | `control_mode`, `set_control_mode()`, `controller` property |
| [base_agent.py](mani_skill/agents/base_agent.py) | 298-308 | `action_space` property（委托给 controller） |
| [base_controller.py](mani_skill/agents/controllers/base_controller.py) | 67-77 | Controller 初始化 action_space |
| [base_controller.py](mani_skill/agents/controllers/base_controller.py) | 125-138 | `_preprocess_action()` |
| [pd_joint_pos.py](mani_skill/agents/controllers/pd_joint_pos.py) | 33-36 | PDJointPosController action_space |
| [pd_joint_pos.py](mani_skill/agents/controllers/pd_joint_pos.py) | 76-93 | PDJointPosController set_action |
| [pd_joint_vel.py](mani_skill/agents/controllers/pd_joint_vel.py) | 39-43 | PDJointVelController set_action |
| [pd_ee_pose.py](mani_skill/agents/controllers/pd_ee_pose.py) | 57-60 | PDEEPosController action_space |
