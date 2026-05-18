# VR 遥操作旋转对齐问题排查记录

## 问题背景

三套 Panda 遥操作脚本共用 `compute_target_ee_pose` 函数，把 VR 手柄姿态映射为机械臂末端目标姿态：

| 脚本 | 控制模式 | 旋转动作格式 | `angle=-angle` |
|------|---------|------------|---------------|
| `vr_controller_panda_simple2.py` | `pd_ee_delta_pose` | axis * angle (delta) | **需要** |
| `vr_controller_panda_pd_ee_pose.py` | `pd_ee_pose` | Euler XYZ (绝对) | **需要** |
| `vr_controller_panda_pink.py` | `pd_joint_pos` + Pink IK | pin.SE3 (绝对) | **不需要** |

现象：`simple2` 工作正常，Pink 工作正常，但 `pd_ee_pose` 末端旋转不对齐（尤其手柄竖直左右摆动时方向反了），且 IK 求解器未报告失败。

## 根因 1：`angle = -angle`（旋转方向修正）

### 问题定位

VR 坐标系到机器人坐标系的映射矩阵：

```python
coord_transform = np.array([
    [0, 0, -1],   # VR Z → Robot -X
    [-1, 0, 0],   # VR X → Robot -Y
    [0, 1, 0],    # VR Y → Robot Z
])
```

`compute_target_ee_pose` 中把 VR 姿态差 `q_diff` 转为机器人坐标系下的 delta 四元数 `q_delta_sim`，然后 LEFT-multiply 到参考姿态上：

```python
target_quat = tf_quat.qmult(q_delta_sim, sim_target_quat)
```

`pd_ee_pose` 控制器和 `simple2` 的 `pd_ee_delta_pose` 控制器都走 `kinematics.compute_ik`，其内部把 target 转 delta 的方式是：

```python
# kinematics.py 第 228-231 行
translation = pose.p - current_pose.p
quaternion = rotation_conversions.quaternion_multiply(
    pose.q, rotation_conversions.quaternion_invert(current_pose.q)
)
# = q_target * q_current^{-1}   (全局帧右减)
```

这个全局帧 quaternion 差在 `pd_ee_pose` 系列控制器内部被转为 Euler 角增量后，与雅可比矩阵的符号约定之间存在一个方向反转。需要显式 `angle = -angle` 来修正。

### Pink 为什么不需要

Pink 的 `FrameTask.compute_error` 使用 `pin.log(T_current^{-1} * T_target)` 从机体帧计算误差，内部符号约定自洽，不需要这个修正。

### 关键代码位置

- `third_party/XLeVR/vr_controller_panda_pd_ee_pose.py` 第 186 行：`angle > np.pi` 分支之后应加 `angle = -angle`
- `third_party/XLeVR/vr_controller_panda_simple2.py` 第 197 行：已有 `angle = -angle`
- `third_party/XLeVR/vr_controller_panda_pink.py` 第 318 行：`#angle = -angle` 被注释掉（正确）

## 根因 2：Euler 角约定不匹配（`axes='sxyz'` vs `axes='rxyz'`）

### 问题定位

`pd_ee_pose` 脚本把目标四元数转为 Euler 角写入动作：

```python
# vr_controller_panda_pd_ee_pose.py 第 402 行
target_euler = tf_euler.quat2euler(last_target_quat, axes='sxyz')
action[3:6] = target_euler
```

控制器内部把 Euler 角重建为姿态四元数：

```python
# pd_ee_pose.py controller (absolute mode)
target_quat = matrix_to_quaternion(
    euler_angles_to_matrix(target_rot, "XYZ")
)
# R = Rx(e0) @ Ry(e1) @ Rz(e2)   ← XYZ 外旋
```

**两种约定的乘法顺序相反：**

```
transforms3d 'sxyz':  R = Rz(ez) @ Ry(ey) @ Rx(ex)    ← ZYX 外旋（先绕 X 再绕 Y 再绕 Z）
Controller   'XYZ':   R = Rx(ex) @ Ry(ey) @ Rz(ez)    ← XYZ 外旋（先绕 X 再绕 Y 再绕 Z）

transforms3d 'rxyz':  R = Rx(ex) @ Ry(ey) @ Rz(ez)    ← XYZ 内旋 = 匹配控制器！
```

对于单轴旋转（如纯绕 Z），两种约定结果相同，所以简单测试看不出来。但当四元数涉及多轴旋转（如手柄竖直 + 左右摆动 + 末端起始姿态非 identity），Euler 角就会被错误分解。

### 实测验证

测试条件：手柄竖直（绕 VR X +90° 指向上方），然后向左摆动（绕 VR Z +30°）。末端初始姿态为绕 Robot Z +45°，`angle=-angle` 已启用。

```python
# 测试脚本输出
quat2euler(sxyz):  euler = [22.2, -20.7, 40.9]
控制器重建后 IK delta = [22.2, -20.7, -4.1]    ← 错误！旋转分量泄漏到三根轴上

quat2euler(rxyz):  euler = [30, 0, 45]
控制器重建后 IK delta = [30, 0, 0]              ← 正确！纯绕 X 轴
```

### 为什么 simple2 和 Pink 不受影响

- **simple2**：直接把 `axis_robot * angle`（axis-angle 表示）塞进动作数组，不经过 `quat2euler`。对于小角度，Euler XYZ ≈ axis-angle，控制器内部 `euler_angles_to_matrix` → `matrix_to_quaternion` 的来回转换近似保真。
- **Pink**：直接构造 `pin.SE3(rot_matrix, pos)` 给 `FrameTask.set_target`，完全不经过 Euler 角表示。

### 修复方式

```python
# vr_controller_panda_pd_ee_pose.py 第 402 行
# 改前
target_euler = tf_euler.quat2euler(last_target_quat, axes='sxyz')
# 改后
target_euler = tf_euler.quat2euler(last_target_quat, axes='rxyz')
```

## 关键测试脚本

### 1. 验证 Euler 约定不匹配

```python
import numpy as np
import torch
from mani_skill.utils.geometry import rotation_conversions as rc
import transforms3d.euler as tfe
import transforms3d.quaternions as tfq

# 模拟手柄竖直 + 左摆 + 末端初始 RotZ(45°)
vr_ref_quat = tfq.axangle2quat([1., 0., 0.], np.deg2rad(90))
q_swing = tfq.axangle2quat([0., 0., 1.], np.deg2rad(30))
curr_vr_quat = tfq.qmult(q_swing, vr_ref_quat)

q_diff = tfq.qmult(curr_vr_quat, tfq.qinverse(vr_ref_quat))
axis_diff, angle_diff = tfq.quat2axangle(q_diff)
angle_neg = -angle_diff  # angle=-angle 修正

coord_transform = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
axis_sim = coord_transform @ axis_diff
q_delta_sim = tfq.axangle2quat(axis_sim, angle_neg)

sim_target_q = np.array(tfq.axangle2quat([0., 0., 1.], np.deg2rad(45)))
target_quat = tfq.qmult(q_delta_sim, sim_target_q)

for axes in ['sxyz', 'rxyz']:
    euler = np.array(tfe.quat2euler(target_quat, axes=axes), dtype=np.float32)
    t = torch.from_numpy(euler).unsqueeze(0)
    R = rc.euler_angles_to_matrix(t, 'XYZ')
    q_rt = rc.matrix_to_quaternion(R)
    current_q = torch.from_numpy(sim_target_q.astype(np.float32)).unsqueeze(0)
    delta_q = rc.quaternion_multiply(q_rt, rc.quaternion_invert(current_q))
    delta_e = rc.matrix_to_euler_angles(rc.quaternion_to_matrix(delta_q), 'XYZ')
    print(f"{axes}: IK delta = {torch.rad2deg(delta_e)}")
    # sxyz: IK delta = [22.2, -20.7, -4.1]  ← 错误
    # rxyz: IK delta = [30, 0, 0]           ← 正确
```

### 2. 验证所有 transforms3d 约定与 Controller 的兼容性

```python
# 遍历所有 transforms3d axes 参数，检查哪个与 controller 的 "XYZ" 匹配
q = tfq.qmult(
    tfq.axangle2quat([1,0,0], np.deg2rad(30)),
    tfq.axangle2quat([0,0,1], np.deg2rad(45))
)

for axes in ['sxyz', 'szyx', 'szxy', 'syxz', 'syzx', 'sxzy',
             'rxyz', 'rzyx', 'rzxy', 'ryxz', 'ryzx', 'rxzy']:
    euler = np.array(tfe.quat2euler(q, axes=axes), dtype=np.float32)
    t = torch.from_numpy(euler).unsqueeze(0)
    R = rc.euler_angles_to_matrix(t, 'XYZ')
    q_rt = rc.matrix_to_quaternion(R)
    dot = torch.abs(torch.sum(q_rt * torch.from_numpy(q.astype(np.float32)), dim=-1))
    if dot > 0.9999:
        print(f"MATCH: axes='{axes}', euler={np.rad2deg(euler)}")
# 输出: MATCH: axes='rxyz', euler=[30, 0, 45]
```

### 3. 验证简单 delta 管线的正确性（simple2 走的路）

```python
# 测试 axis_robot * angle 经过 controller Euler 解析后的保真度
axis_vr = np.array([0., 1., 0.])  # 绕 VR Y 旋转
angle = np.deg2rad(30)
axis_sim = coord_transform @ axis_vr  # → [0, 0, 1]
action_rot = axis_sim * (-angle)      # angle=-angle

t = torch.tensor([action_rot], dtype=torch.float32)
delta_quat = rc.matrix_to_quaternion(rc.euler_angles_to_matrix(t, 'XYZ'))
delta_euler = rc.matrix_to_euler_angles(rc.quaternion_to_matrix(delta_quat), 'XYZ')
# 小角度下 delta_euler ≈ action_rot（axis-angle ≈ Euler XYZ）
```

## 经验总结

1. **不要混用 Euler 角约定**：`transforms3d` 和 `pytorch3d`（mani_skill 使用的）虽然参数名相似，但 `axes='sxyz'` ≠ `convention="XYZ"`。前者是 extrinsic ZYX，后者是 extrinsic XYZ。等价约定是 `axes='rxyz'`（intrinsic XYZ）。

2. **delta 模式有天然鲁棒性**：`pd_ee_delta_pose` 每帧只传小角度增量，axis-angle ≈ Euler XYZ 的近似对 Euler 约定不敏感。绝对姿态模式则会暴露约定差异。

3. **坐标变换矩阵同时用于位置和旋转轴时要小心**：`coord_transform @ pos_vector` 和 `coord_transform @ rotation_axis` 在纯旋转矩阵（det=+1）下都是正确的，但如果矩阵包含反射（det=-1）则需要额外处理。

4. **排查这类问题时，先验证 roundtrip**：`target_quat → Euler → controller → quat → IK delta` 这条链路上每一步的保真度都要测试，尤其用非平凡的多轴旋转用例。
