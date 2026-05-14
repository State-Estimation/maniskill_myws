# Sweep Solar Panel v2 — Task Design Documentation

> 用于论文“任务设计”部分的撰写参考。

---

## 1. 任务概述

**任务名称**: `SolarPanelStatic-v2`  
**任务目标**: 控制 Panda 机械臂抓取刷子，在太阳能板表面的指定区域上完成擦拭（清洁）任务。  
**成功条件**: 清洁区域内 ≥ 60% 的网格单元（cell）被刷子覆盖过。  
**最大步数**: 600 步（sim_freq=200Hz, control_freq=20Hz → 30 秒）。  
**奖励模式**: 稀疏奖励（sparse），成功为 1，否则为 0。

### 1.1 场景布局（俯视）

```
  ← 机械臂 (base at x=-0.615)
         |
    [刷子]  →  太阳能板 (center at x≈0.25)
                  面朝向机械臂（world -X, +Z 方向）
```

---

## 2. 场景构建

### 2.1 基础场景

- **桌子**: `TableSceneBuilder` 自动生成
- **重力**: `(0, 0, -0.00098)` — 微重力环境
- **仿真频率**: sim_freq=200Hz, control_freq=20Hz

### 2.2 机械臂

- 型号: Franka Panda (支持 `panda` 和 `panda_wristcam`)
- 基座位置: `(-0.615, 0, 0)`（桌面左后方）
- 初始关节角 (home qpos):

```python
[0.008, 0.105, 0.029, -2.747, 0.002, 2.772, ±0.870, 0.04, 0.04]
# 关节 0-6 为臂部，7-8 为夹爪
```

- 初始化噪声: ±0.02 rad（每关节独立采样）

### 2.3 太阳能板

- 加载方式: OBJ 网格文件 (`solar_panel/mesh/solar_panel.obj`)，含 non-convex 三角网格碰撞
- 构建类型: `build_static`（静态刚体，不可移动）
- 面板局部 BBox（OBJ 原始坐标）:

| 轴 | 最小值 | 最大值 | 说明 |
|---|---|---|---|
| X | -0.330 | 0.330 | 水平方向（面板宽度） |
| Y | -0.023 | 0.471 | 面板厚度 / 面法线方向 |
| Z | -0.850 | 0.850 | 竖直方向（面板高度） |

- `PANEL_BASE_Q = (√2/2, √2/2, 0, 0)` — 绕 X 轴 +90°，使局部 Y 指向世界 +Z
- 放置 Z: `PANEL_TABLE_CLEARANCE(0.005) - PANEL_LOCAL_MIN_Y(-0.023) ≈ 0.028m`（底面刚好贴桌）
- 位置随机化:
  - 世界 X: `0.25 ± 0.035`
  - 世界 Y: `0.0 ± 0.03`
  - 偏航角: ±5°

### 2.4 刷子

- 加载方式: URDF 文件 (`brush/urdf/brush.urdf`)
  - 单 link (base_link)，无关节 → SAPIEN 解析为 free Actor（可被抓取）
  - 视觉和碰撞网格 rpy 均为 `rpy="-1.5708 0 0"`（绕 X 轴 -90°）
- 碰撞: 非凸三角网格（glb 分解文件）
- 初始朝向 `BRUSH_BASE_Q = (0.5, 0.5, 0.5, -0.5)`:
  - 使刷子趴放（清洁面朝桌面），刷头指向 world +Y
  - 由 `R_z(-90°) * R_y(-90°)` 组合得到
- 放置 Z: `0.083m`（刷子最低点高于桌面 5mm 间隙）
- 位置随机化:
  - 世界 X: `-0.10 ± 0.03`
  - 世界 Y: `-0.20 ± 0.05`
  - 偏航角: ±10°
  - X 钳制: 刷子右侧不穿透面板左侧 `panel_min_x - brush_x_radius - clearance`

---

## 3. 面板表面平面方程

### 3.1 问题

面板 OBJ 表面不是平行于面板局部 X-Z 平面的——它是一个倾斜面。面板四角在面板局部坐标系中:

```
A (顶部, +Z): ( 0.316,  0.470, -0.812)
B (底部, +Z): (-0.329,  0.087, -0.812)
C (底部, -Z): (-0.329,  0.087,  0.812)
D (顶部, -Z): ( 0.316,  0.470,  0.812)
```

可见 Z 恒定时 Y 随 X 变化 → 面板面在局部 X-Y 平面内倾斜。

### 3.2 平面方程

法向量（朝向机械臂，即 world -X/+Z 方向）:

```
n_outward = (-0.511, 0.860, 0)   # |n| = 1
```

平面方程（面板局部坐标系）:

```
-0.511·x + 0.860·y - 0.243 = 0
```

等价形式:

```
y(x) = 0.594·x + 0.283   # 面板面 Y 与 X 的线性关系
```

### 3.3 点到面的垂直距离

```python
dist = |a*x + b*y + d|   # a=-0.511, b=0.860, d=-0.243
```

由于法向量已归一化，dist 即为点到面的垂直距离（米）。

---

## 4. 清洁检测逻辑

### 4.1 刷面表示

刷子清洁面由四个角点表示（在 link frame 中）:

```python
BRUSH_FACE_PTS_LINK = [
    (0.0552,  0.0811, -0.1391),  # P1: 顶-窄边
    (0.0778,  0.0275, -0.1382),  # P2: 底-窄边
    (0.0778,  0.0271,  0.1398),  # P3: 底-宽边
    (0.0555,  0.0808,  0.1398),  # P4: 顶-宽边
]
# 测量自 brush2.STL (Blender 对齐)，经 URDF rpy="-1.5708 0 0" 转换至 link frame
# 面积 ≈ 0.0162 m²
```

每帧将四角点坐标变换到面板局部坐标系:

```
face_world = brush_pose × face_link          # (B, 4, 3)
face_panel = panel_pose⁻¹ × face_world       # (B, 4, 3)
```

### 4.2 接触判定

四个角点**全部**满足距面板面的垂直距离 ≤ `clean_surface_tolerance`（默认 0.03m）:

```python
all_near = all_i( |a*x_i + b*y_i + d| ≤ 0.03 )  # i ∈ {0,1,2,3}
```

这替代了原来的单点刷头检测，能更准确地判断刷子清洁面是否贴合面板表面。

### 4.3 覆盖范围判定

将四个角点在面板面上的**垂足**投影到面板局部 X-Z 坐标:

```python
foot_x = x - a * signed_dist    # 垂足在面板面上的 X 坐标
foot_z = z                       # Z 不变（法线 n_z = 0）
```

在 X-Z 平面上取四个垂足点的包围盒。所有中心位于包围盒内的 cell 被标记为已清洁:

```python
cell_covered = (x_min ≤ cell_x ≤ x_max) ∧ (z_min ≤ cell_z ≤ z_max)
newly_cleaned = all_near ∧ cell_covered
```

### 4.4 清洁累计与成功判定

- 清洁状态**累计**: `cleaned_cells |= newly_cleaned` — 已经清洁的 cell 不会回退
- 清洁覆盖率: `clean_coverage = mean(cleaned_cells)`
- 成功: `clean_coverage ≥ clean_success_ratio`（默认 0.6，即 60%）

---

## 5. 检测区域与网格划分

### 5.1 清洁区域

检测区域为面板表面上的一个子矩形，定义在面板局部 X-Z 平面:

```python
CLEAN_REGION_MIN_X = PANEL_LOCAL_MIN_X                    # ≈ -0.330
CLEAN_REGION_MAX_X = PANEL_LOCAL_MIN_X + panel_len / 2.5  # ≈ -0.066
CLEAN_REGION_MIN_Z = -panel_half_z / 3                    # ≈ -0.283
CLEAN_REGION_MAX_Z =  panel_half_z / 3                    # ≈  0.283
```

### 5.2 网格

```python
clean_grid_x = 5   # X 方向分 5 列
clean_grid_y = 7   # Z 方向分 7 行
# 共 35 个 cell
```

Cell 中心坐标均匀分布在 `CLEAN_REGION` 内。

---

## 6. 清洁标记（Clean Marker）可视化

### 6.1 构建

- 类型: `build_kinematic`（纯视觉，无碰撞，可 set_pose）
- 形状: 薄盒 (half_size = `[cell_spacing_X*0.47, 0.0015, cell_spacing_Z*0.47]`)
- 材质: 半透明青色 `(0.78, 0.96, 1.0, 0.62)`
- 尺寸 ×0.47 确保相邻 cell 间有约 6% 的间隙

### 6.2 朝向

Markers 需与面板面法线对齐（而非 link frame Y 轴）。panel link Y 与表面法线的夹角为 30.7°，需施加偏移四元数:

```python
q_offset = (0.9643, 0, 0, 0.2649)  # 绕 Z 轴 30.7°
marker_q = panel_q * q_offset
```

### 6.3 显示/隐藏

- **可见**: 面板面上方 4mm（沿外向法线 `(-0.511, 0.860, 0)` 偏移）
- **隐藏**: 面板面后方 10m（沿外向法线反向偏移，推到面板背后深处）

---

## 7. 观测空间

除 ManiSkill 默认观测外，额外提供:

| 字段 | 形状 | 说明 |
|---|---|---|
| `tcp_pose` | (7,) | 末端执行器位姿 (x,y,z,qw,qx,qy,qz) |
| `brush_pose` | (7,) | 刷子位姿 |
| `panel_pose` | (7,) | 面板位姿 |
| `clean_coverage` | (B,) | 当前清洁覆盖率 [0, 1] |
| `cleaning_contact` | (B,) | 刷子是否贴合面板 (bool) |
| `brush_face_panel` | (B,4,3) | 刷面四角在面板局部的坐标 |

---

## 8. 域随机化（Domain Randomization）

| 参数 | 分布 | 范围 |
|---|---|---|
| 刷子 XY 位置 | Uniform | ±0.03m / ±0.05m |
| 刷子偏航角 | Uniform | ±10° |
| 面板 XY 位置 | Uniform | ±0.035m / ±0.03m |
| 面板偏航角 | Uniform | ±5° |
| 机器人初始 qpos | Normal(0, 0.02) | 每关节独立 |
| 刷子初始 X 钳制 | — | 不穿透面板左侧 |

---

## 9. 关键设计要点总结

1. **倾斜表面处理**: 面板面不平行于任何坐标平面。所有接触检测、标记放置必须基于面板表面平面方程 `-0.511x + 0.860y - 0.243 = 0`，而非简单的常数坐标比较。

2. **刷面四角接触判定**: 用刷刷清洁面的四个物理角点替代单点刷头中心，判定更准确——要求全部四个角点都贴近面板表面才算有效接触。

3. **垂足投影覆盖**: 清洁覆盖范围由四角点垂足在面板面上的包围盒决定，自动适配刷子的实际朝向和位姿。

4. **视觉标记无碰撞**: clean marker 使用 `build_kinematic` + 仅 `add_box_visual`（无碰撞几何），确保纯可视化不影响物理交互。

5. **朝向对齐**: marker 需要通过 `panel_q * q_offset` 使厚度方向对齐面板表面法线，而非简单沿用面板 link 朝向。

6. **坐标系转换链**:

```
STL/Mesh 坐标 → (URDF rpy) → Link Frame → (brush.pose) → World → (panel.pose⁻¹) → Panel Local
```
