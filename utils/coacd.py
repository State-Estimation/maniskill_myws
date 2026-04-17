####################
# 由于numpy版本问题, 次脚本无法在当前环境运行, 
# 功能是将一个复杂的网格模型分割成多个凸块, 
# 以便在物理引擎中更高效地进行碰撞检测和物理模拟。
# 使用了coacd库来实现这个功能, 并且使用trimesh库来处理网格数据。
# 分割后的凸块被保存为新的glb文件, 可以在后续的仿真中使用。
####################
import coacd
import trimesh

dir_path = "/home/firedust/mimic/maniskill_myws/src/maniskill_myws/assets/brush/meshes"
mesh = trimesh.load(f"{dir_path}/brush.glb", force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)

parts = coacd.run_coacd(
    mesh,
    threshold=0.05,     # 越小越精细（关键参数）
    max_convex_hull=20, # 最大分块数
    resolution=1000     # 分辨率（越大越精细但更慢）
)

mesh_parts = []
for vs, fs in parts:
    mesh_parts.append(trimesh.Trimesh(vs, fs))

for i, p in enumerate(mesh_parts):
    p.export(f"{dir_path}/brush_part_{i}.glb")