# CoACD 凸分解操作指南

## 1. 使用 Blender Python 环境运行 CoACD

CoACD 需要 `coacd` 和 `trimesh` 库。使用 Blender 自带的 Python 环境：

```bash
/home/firedust/app/blender/blender-5.0.0-linux-x64/5.0/python/bin/python3.11 -m pip install coacd trimesh
```

## 2. 凸分解单个网格

`CoACD.py` 支持命令行参数，无需编辑代码：

```bash
cd /path/to/assets
/home/firedust/app/blender/blender-5.0.0-linux-x64/5.0/python/bin/python3.11 CoACD.py <dir_path> <pre_fix>
```

**参数说明：**
- `dir_path`: STL 文件所在目录（如 `door2/meshes`）
- `pre_fix`: STL 文件名前缀（不含 `.STL`，如 `handle_link`）

**示例：**
```bash
# 分解 door2/meshes/handle_link.STL，输出到 door2/meshes/handle_link/
/home/firedust/app/blender/blender-5.0.0-linux-x64/5.0/python/bin/python3.11 CoACD.py door2/meshes handle_link

# 分解 door2/meshes/door_frame.STL
/home/firedust/app/blender/blender-5.0.0-linux-x64/5.0/python/bin/python3.11 CoACD.py door2/meshes door_frame
```

**CoACD 参数（可在脚本中修改）：**
```python
parts = coacd.run_coacd(
    mesh,
    threshold=0.05,     # 越小越精细（关键参数）
    max_convex_hull=15, # 最大分块数
    resolution=1000     # 分辨率（越大越精细但更慢）
)
```

## 3. URDF collision 替换为 GLB 文件

凸分解后会得到 `part_0.glb`, `part_1.glb`, ... 等文件。需要将这些文件添加到 URDF 对应 link 的 collision 中。

### 替换规则

将 URDF 中的：
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename="package://door2/meshes/door_frame.STL" />
  </geometry>
</collision>
```

替换为多个 collision（每个 part 一个）：
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename="package://door2/meshes/door_frame/part_0.glb" />
  </geometry>
</collision>
<collision>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename="package://door2/meshes/door_frame/part_1.glb" />
  </geometry>
</collision>
<!-- ... part_2, part_3, ... -->
```

### 注意事项

- 路径前缀保留 `package://`（如 `package://door2/meshes/`）
- 每个 part 都需要一个独立的 `<collision>` 标签
- collision 应放在对应 `<link>` 标签内，与 `<inertial>`、`<visual>` 平级
- mesh 文件路径相对于 URDF 文件所在目录

### 示例：door2/urdf/door.urdf

| Link | GLB 文件 | 数量 |
|------|----------|------|
| door_frame | part_0.glb ~ part_11.glb | 12 |
| door_link | part_0.glb ~ part_8.glb | 9 |
| button_link | part_0.glb | 1 |
| handle_link | part_0.glb ~ part_2.glb | 3 |
