import coacd
import trimesh
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python CoACD.py <dir_path> <pre_fix>")
    print("Example: python CoACD.py door2/meshes handle_link")
    sys.exit(1)

dir_path = sys.argv[1]
pre_fix = sys.argv[2]
out_dir = f"{dir_path}/{pre_fix}"
os.makedirs(out_dir, exist_ok=True)
mesh = trimesh.load(f"{dir_path}/{pre_fix}.glb", force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)

parts = coacd.run_coacd(
    mesh,
    threshold=0.05,     # 越小越精细（关键参数）
    max_convex_hull=15, # 最大分块数
    resolution=1000     # 分辨率（越大越精细但更慢）
)

mesh_parts = []
for vs, fs in parts:
    mesh_parts.append(trimesh.Trimesh(vs, fs))

for i, p in enumerate(mesh_parts):
    p.export(f"{out_dir}/part_{i}.glb")