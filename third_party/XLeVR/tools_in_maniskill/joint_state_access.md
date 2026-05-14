# ManiSkill 获取机器人关节状态

## 最常用的方式

```python
# 1. env.step() 直接返回（需使用 state_dict 模式的 obs_mode）
obs, reward, terminated, truncated, info = env.step(action)
# obs["agent"]["qpos"]  -> tensor (num_envs, dof)
# obs["agent"]["qvel"]  -> tensor (num_envs, dof)

# 2. 通过 env.agent 直接访问（推荐，不依赖 obs_mode）
qpos = env.agent.robot.get_qpos()    # tensor (num_envs, dof)
qvel = env.agent.robot.get_qvel()    # tensor (num_envs, dof)

# 3. 获取完整的 proprioception 字典
proprio = env.agent.get_proprioception()
# {"qpos": ..., "qvel": ..., "controller": ...}
```

## 更细粒度的访问

```python
# 4. 按关节名称访问
joint = env.agent.robot.active_joints_map["joint_name"]
joint.qpos    # 该关节位置
joint.qvel    # 该关节速度

# 5. 获取控制器所控制的关节（过滤后）
env.agent.controller.qpos    # 仅当前控制器控制的关节
env.agent.controller.qvel
```

## 获取完整状态

```python
# 6. 获取完整状态（包含基座 pose/速度等）
state = env.agent.get_state()
# {"robot_root_pose": ..., "robot_root_vel": ...,
#  "robot_qpos": ..., "robot_qvel": ...}
```

## 关键点

- `qpos` 和 `qvel` 返回的都是 `torch.Tensor`，形状为 `(num_envs, dof)`，CPU/GPU 仿真均一致
- `env.agent.robot` 是 `Articulation` 对象，核心 API：`.qpos` / `.qvel` 属性、`.get_qpos()` / `.get_qvel()` 方法
- 所有关节（含 fixed）在 `env.agent.robot.joints`，可动关节在 `env.agent.robot.active_joints`
- 相关源码：
  - `mani_skill/envs/sapien_env.py` — `step()`, `get_obs()`, `_get_obs_agent()`
  - `mani_skill/agents/base_agent.py` — `get_proprioception()`, `get_state()`
  - `mani_skill/utils/structs/articulation.py` — `Articulation.qpos`, `Articulation.qvel`
  - `mani_skill/utils/structs/articulation_joint.py` — `ArticulationJoint.qpos`, `ArticulationJoint.qvel`
  - `mani_skill/agents/controllers/base_controller.py` — `Controller.qpos`, `Controller.qvel`
