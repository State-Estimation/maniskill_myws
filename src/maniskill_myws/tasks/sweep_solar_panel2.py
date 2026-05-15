import math
from typing import Any
import sapien

from mani_skill.agents.robots import Panda, PandaWristCam

from mani_skill.sensors.camera import CameraConfig, Union
from mani_skill.utils import sapien_utils
from mani_skill.envs.utils import randomization

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig, SceneConfig
from mani_skill.utils.registration import register_env
import importlib.resources as importlib_resources
import torch
import numpy as np

from maniskill_myws.task_prompts import TASK_PROMPTS


@register_env("SolarPanelStatic-v2", max_episode_steps=600)
class SolarPanelStaticEnv2(BaseEnv):

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    DEFAULT_TASK_PROMPT = TASK_PROMPTS["SolarPanelStatic-v2"]
    # The OBJ is placed with a 90 deg X rotation, so local y becomes world z.
    # These bounds are from assets/solar_panel/mesh/solar_panel.obj.
    PANEL_LOCAL_MIN_X: float = -0.329515
    PANEL_LOCAL_MAX_X: float = 0.329515
    PANEL_LOCAL_MIN_Y: float = -0.023163
    PANEL_LOCAL_MAX_Y: float = 0.470819
    PANEL_LOCAL_MIN_Z: float = -0.850000
    PANEL_LOCAL_MAX_Z: float = 0.850003
    PANEL_TABLE_CLEARANCE: float = 0.005
    PANEL_BASE_Q: tuple[float, float, float, float] = (
        math.sqrt(2.0) / 2.0,
        math.sqrt(2.0) / 2.0,
        0.0,
        0.0,
    )

    # Detection region on the panel surface (panel-local X-Z):
    #   bottom edge midpoint → up half panel height, ±1/6 panel length sideways
    _panel_len_x = PANEL_LOCAL_MAX_X - PANEL_LOCAL_MIN_X  # ≈ 0.659
    _panel_half_z = (PANEL_LOCAL_MAX_Z - PANEL_LOCAL_MIN_Z) / 2  # ≈ 0.85
    CLEAN_REGION_MIN_X: float = PANEL_LOCAL_MIN_X  
    CLEAN_REGION_MAX_X: float = PANEL_LOCAL_MIN_X + _panel_len_x / 2.5
    CLEAN_REGION_MIN_Z: float = -_panel_half_z / 3  # = -0.85/3
    CLEAN_REGION_MAX_Z: float = _panel_half_z / 3  # = 0.85/3

    # Brush: loaded from assets/brush/urdf/brush.urdf (base_link with STL mesh).
    #   Collision mesh frame:  X=thin(0.044) Y=wide(0.140) Z=long/handle(0.163)
    #   BRUSH_BASE_Q rotates collision frame so the brush lies flat (face down)
    #   with the head pointing +Y:  mesh Z→world+Y, mesh X→world-Z, mesh Y→world+X.
    BRUSH_BASE_Q: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    # Four corners of the brush cleaning face in the LINK frame.
    # Measured on brush2.STL (Blender-aligned), then rotated by rpy="-1.5708 0 0".
    BRUSH_FACE_PTS_LINK: tuple = (
        ( 0.0552,  0.0811, -0.1391),
        ( 0.0778,  0.0275, -0.1382),
        ( 0.0778,  0.0271,  0.1398),
        ( 0.0555,  0.0808,  0.1398),
    )
    BRUSH_TABLE_CLEARANCE: float = 0.005
    BRUSH_PANEL_CLEARANCE: float = 0.04
    BRUSH_MAX_WORLD_X_RADIUS: float = 0.15  # ≈ half mesh Y after rotation
    BRUSH_REST_Z: float = 0.083  # = |mesh X min after rot| + clearance ≈ 0.0775 + 0.005 ≈ 0.083

    ROBOT_HOME_QPOS_PANDA = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )
    ROBOT_HOME_QPOS_PANDA_WRISTCAM = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )

    # Panel surface plane in panel-local coords: a*x + b*y + d = 0  (c=0)
    #   Surface y(x) = slope * x + intercept
    #   Outward normal (toward robot): (nx, ny, 0)
    PANEL_SURFACE_A: float = -0.511
    PANEL_SURFACE_B: float = 0.860
    PANEL_SURFACE_D: float = -0.243
    PANEL_SURFACE_SLOPE: float = 0.594
    PANEL_SURFACE_INTERCEPT: float = 0.283
    PANEL_SURFACE_NX: float = -0.511
    PANEL_SURFACE_NY: float = 0.860
    # Rotation from link Y to surface normal (around panel-local Z): (w, 0, 0, z)
    PANEL_SURFACE_Q_OFFSET: tuple = (0.9643, 0.0, 0.0, 0.2649)
    CLEAN_MARKER_SURFACE_OFFSET: float = 0.004
    CLEAN_MARKER_THICKNESS: float = 0.0015
    CLEAN_MARKER_HIDE_DIST: float = 10.0  # push marker this far behind surface

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_base_x: float = -0.615,
        robot_base_y: float = 0.0,
        robot_init_qpos_noise: float = 0.02,
        brush_spawn_center_x: float = -0.10,
        brush_spawn_center_y: float = -0.2,
        brush_xy_noise: float = 0.03,
        brush_y_noise: float = 0.05,
        brush_z: float | None = None,
        brush_yaw_noise: float = math.radians(10),
        panel_spawn_center_x: float = 0.25,
        panel_spawn_center_y: float = 0.0,
        panel_spawn_half_size_x: float = 0.035,
        panel_spawn_half_size_y: float = 0.03,
        panel_yaw_noise: float = math.radians(5),
        clean_grid_x: int = 5,
        clean_grid_y: int = 7,
        clean_radius: float = 0.1,
        clean_surface_tolerance: float = 0.03,
        clean_success_ratio: float = 0.6,
        **kwargs,
    ):
        self.robot_base_x = float(robot_base_x)
        self.robot_base_y = float(robot_base_y)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.brush_spawn_center_x = float(brush_spawn_center_x)
        self.brush_spawn_center_y = float(brush_spawn_center_y)
        self.brush_xy_noise = brush_xy_noise
        self.brush_y_noise = brush_y_noise
        self.brush_z = self.BRUSH_REST_Z if brush_z is None else float(brush_z)
        self.brush_yaw_noise = float(brush_yaw_noise)
        self.panel_spawn_center_x = float(panel_spawn_center_x)
        self.panel_spawn_center_y = float(panel_spawn_center_y)
        self.panel_spawn_half_size_x = float(panel_spawn_half_size_x)
        self.panel_spawn_half_size_y = float(panel_spawn_half_size_y)
        self.panel_yaw_noise = float(panel_yaw_noise)
        self.clean_grid_x = int(clean_grid_x)
        self.clean_grid_y = int(clean_grid_y)
        self.clean_radius = float(clean_radius)
        self.clean_surface_tolerance = float(clean_surface_tolerance)
        self.clean_success_ratio = float(clean_success_ratio)
        self._cleaned_cells: torch.Tensor | None = None
        self._clean_cell_centers: torch.Tensor | None = None
        self.clean_markers = []
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=20,scene_config=SceneConfig(gravity=[0, 0, -0.00098]))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.7, -0.6, 0.9], [0.0, 0.0, 0.2])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=1)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 0.4, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[self.robot_base_x, self.robot_base_y, 0])
        )

    def _load_scene(self, options):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        assets_dir = importlib_resources.files("maniskill_myws").joinpath("assets")
        with importlib_resources.as_file(assets_dir) as d:

            # =========================
            # 静态物体：太阳能板
            # =========================
            panel_builder = self.scene.create_actor_builder()
            panel_builder.add_visual_from_file(
                str(d / "solar_panel/mesh/solar_panel.obj")
            )
            panel_builder.add_nonconvex_collision_from_file(
                str(d / "solar_panel/mesh/solar_panel.obj")
            )
            panel_pose = sapien.Pose(
                p=[
                    self.panel_spawn_center_x,
                    self.panel_spawn_center_y,
                    self.PANEL_TABLE_CLEARANCE - self.PANEL_LOCAL_MIN_Y,
                ],
                q=list(self.PANEL_BASE_Q),
            )
            panel_builder.initial_pose = panel_pose
            self.panel = panel_builder.build_static(name="solar_panel")
            self.panel.set_pose(panel_pose)

            # =========================
            # 刷子：通过 URDF 导入
            # =========================
            loader = self.scene.create_urdf_loader()
            loader.name = "brush"
            loader.fix_root_link = False
            loader.disable_self_collisions = True

            urdf_path = d / "brush/urdf/brush.urdf"
            parsed = loader.parse(str(urdf_path), package_dir=str(d))
            articulation_builders = parsed["articulation_builders"]
            actor_builders = parsed["actor_builders"]
            if actor_builders:
                brush_builder = actor_builders[0]
            elif len(articulation_builders) == 1:
                brush_builder = articulation_builders[0]
            else:
                raise RuntimeError(
                    "Expected brush URDF to contain exactly one actor or articulation."
                )
            brush_builder.set_scene_idxs(torch.arange(self.num_envs, dtype=torch.int32))
            brush_builder.initial_pose = sapien.Pose(
                p=[self.brush_spawn_center_x, self.brush_spawn_center_y, self.brush_z]
            )
            self.brush = brush_builder.build(name="brush")

            # =========================
            # 清洁标记
            # =========================
            self.clean_markers = []
            marker_mat = sapien.render.RenderMaterial(
                base_color=[0.78, 0.96, 1.0, 0.62]
            )
            region_len_x = self.CLEAN_REGION_MAX_X - self.CLEAN_REGION_MIN_X
            region_len_z = self.CLEAN_REGION_MAX_Z - self.CLEAN_REGION_MIN_Z
            cell_half_x = region_len_x / self.clean_grid_x * 0.47
            cell_half_z = region_len_z / self.clean_grid_y * 0.47
            for i in range(self.clean_grid_x * self.clean_grid_y):
                marker_builder = self.scene.create_actor_builder()
                marker_builder.add_box_visual(
                    half_size=[
                        cell_half_x,
                        self.CLEAN_MARKER_THICKNESS,
                        cell_half_z,
                    ],
                    material=marker_mat,
                )
                marker_builder.initial_pose = sapien.Pose(
                    p=[
                        self.panel_spawn_center_x,
                        self.panel_spawn_center_y,
                        -self.CLEAN_MARKER_HIDE_DIST,
                    ],
                    q=list(self.PANEL_BASE_Q),
                )
                self.clean_markers.append(
                    marker_builder.build_kinematic(name=f"clean_marker_{i}")
                )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 保留桌面与机器人初始化
        self.scene_builder.initialize(env_idx)
        self._reset_robot_retracted_qpos(env_idx)
        self.agent.robot.set_pose(
            sapien.Pose([self.robot_base_x, self.robot_base_y, 0])
        )

        # 随机放置刷子位置（每个艺术体单环境）
        # 如果 num_envs=1，这里也正常
        b = len(env_idx)
        p = torch.zeros((b, 3), device=self.device)
        p[:, 0] = self.brush_spawn_center_x + randomization.uniform(
            -self.brush_xy_noise, self.brush_xy_noise, size=(b,), device=self.device
        )
        p[:, 1] = self.brush_spawn_center_y + randomization.uniform(
            -self.brush_y_noise, self.brush_y_noise, size=(b,), device=self.device
        )
        p[:, 2] = self.brush_z

        yaw = randomization.uniform(
            -self.brush_yaw_noise,
            self.brush_yaw_noise,
            size=(b,),
            device=self.device,
        )
        half_yaw = yaw * 0.5
        q_yaw = torch.zeros((b, 4), device=self.device)
        q_yaw[:, 0] = torch.cos(half_yaw)
        q_yaw[:, 3] = torch.sin(half_yaw)
        q_base = torch.zeros((b, 4), device=self.device)
        q_base[:, 0] = self.BRUSH_BASE_Q[0]
        q_base[:, 1] = self.BRUSH_BASE_Q[1]
        q_base[:, 2] = self.BRUSH_BASE_Q[2]
        q_base[:, 3] = self.BRUSH_BASE_Q[3]
        q = self._quat_mul(q_yaw, q_base)

        panel_p = torch.zeros((b, 3), device=self.device)
        panel_p[:, 0] = self.panel_spawn_center_x + randomization.uniform(
            -self.panel_spawn_half_size_x,
            self.panel_spawn_half_size_x,
            size=(b,),
            device=self.device,
        )
        panel_p[:, 1] = self.panel_spawn_center_y + randomization.uniform(
            -self.panel_spawn_half_size_y,
            self.panel_spawn_half_size_y,
            size=(b,),
            device=self.device,
        )
        # Keep the lowest mesh vertex slightly above the table, even with yaw noise.
        panel_p[:, 2] = self.PANEL_TABLE_CLEARANCE - self.PANEL_LOCAL_MIN_Y

        yaw = randomization.uniform(
            -self.panel_yaw_noise,
            self.panel_yaw_noise,
            size=(b,),
            device=self.device,
        )
        half_yaw = yaw * 0.5
        q_yaw = torch.zeros((b, 4), device=self.device)
        q_yaw[:, 0] = torch.cos(half_yaw)
        q_yaw[:, 3] = torch.sin(half_yaw)
        q_base = torch.zeros((b, 4), device=self.device)
        q_base[:, 0] = self.PANEL_BASE_Q[0]
        q_base[:, 1] = self.PANEL_BASE_Q[1]
        panel_q = self._quat_mul(q_yaw, q_base)
        self.panel.set_pose(Pose.create_from_pq(panel_p, panel_q))

        panel_min_x = panel_p[:, 0] + self.PANEL_LOCAL_MIN_X
        max_brush_x = (
            panel_min_x - self.BRUSH_MAX_WORLD_X_RADIUS - self.BRUSH_PANEL_CLEARANCE
        )
        p[:, 0] = torch.minimum(p[:, 0], max_brush_x)
        self.brush.set_pose(Pose.create_from_pq(p, q))

        self._ensure_clean_state()
        self._cleaned_cells[env_idx] = False
        self._update_clean_marker_poses()

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _reset_robot_retracted_qpos(self, env_idx: torch.Tensor):
        b = len(env_idx)
        if self.robot_uids == "panda":
            base_qpos = self.ROBOT_HOME_QPOS_PANDA
        else:
            base_qpos = self.ROBOT_HOME_QPOS_PANDA_WRISTCAM
        qpos = np.repeat(base_qpos[None, :], b, axis=0)

        if self._enhanced_determinism:
            qpos = (
                self._batched_episode_rng[env_idx].normal(
                    0, self.robot_init_qpos_noise, len(base_qpos)
                )
                + qpos
            )
        else:
            qpos = self._episode_rng.normal(
                0, self.robot_init_qpos_noise, qpos.shape
            ) + qpos

        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)

    def _ensure_clean_state(self):
        num_cells = self.clean_grid_x * self.clean_grid_y
        if self._cleaned_cells is None or self._cleaned_cells.shape != (self.num_envs, num_cells):
            self._cleaned_cells = torch.zeros(
                (self.num_envs, num_cells), dtype=torch.bool, device=self.device
            )

        if (
            self._clean_cell_centers is not None
            and self._clean_cell_centers.device == torch.device(self.device)
            and self._clean_cell_centers.shape == (num_cells, 2)
        ):
            return

        x_edges = torch.linspace(
            self.CLEAN_REGION_MIN_X,
            self.CLEAN_REGION_MAX_X,
            self.clean_grid_x + 1,
            device=self.device,
        )
        z_edges = torch.linspace(
            self.CLEAN_REGION_MIN_Z,
            self.CLEAN_REGION_MAX_Z,
            self.clean_grid_y + 1,
            device=self.device,
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        grid_x, grid_z = torch.meshgrid(x_centers, z_centers, indexing="ij")
        self._clean_cell_centers = torch.stack([grid_x, grid_z], dim=-1).reshape(-1, 2)

    def _update_clean_marker_poses(self):
        if len(self.clean_markers) == 0 or self._cleaned_cells is None:
            return
        self._ensure_clean_state()

        panel_t = self.panel.pose.to_transformation_matrix()
        if panel_t.ndim == 2:
            panel_t = panel_t.unsqueeze(0)
        b = panel_t.shape[0]
        panel_raw_pose = self.panel.pose.raw_pose
        panel_q = (
            panel_raw_pose[3:7].unsqueeze(0)
            if panel_raw_pose.ndim == 1
            else panel_raw_pose[:, 3:7]
        )
        # marker_q = panel_q * q_offset (rotate marker Y → surface normal)
        qoff = torch.tensor(self.PANEL_SURFACE_Q_OFFSET, device=self.device).unsqueeze(0).expand(b, -1)
        marker_q = self._quat_mul(panel_q, qoff)

        for cell_idx, marker in enumerate(self.clean_markers):
            center = self._clean_cell_centers[cell_idx]
            cx, cz = center[0], center[1]
            # Surface Y at this X on the tilted panel
            surface_y = self.PANEL_SURFACE_SLOPE * cx + self.PANEL_SURFACE_INTERCEPT
            # Visible: surface + small offset along outward normal (toward robot)
            vis_x = cx + self.CLEAN_MARKER_SURFACE_OFFSET * self.PANEL_SURFACE_NX
            vis_y = surface_y + self.CLEAN_MARKER_SURFACE_OFFSET * self.PANEL_SURFACE_NY
            vis_local = torch.stack([
                torch.as_tensor(vis_x, dtype=panel_t.dtype, device=self.device),
                torch.as_tensor(vis_y, dtype=panel_t.dtype, device=self.device),
                cz,
            ])
            # Hidden: push behind surface (opposite of outward normal)
            hide_x = cx - self.CLEAN_MARKER_HIDE_DIST * self.PANEL_SURFACE_NX
            hide_y = surface_y - self.CLEAN_MARKER_HIDE_DIST * self.PANEL_SURFACE_NY
            hide_local = torch.stack([
                torch.as_tensor(hide_x, dtype=panel_t.dtype, device=self.device),
                torch.as_tensor(hide_y, dtype=panel_t.dtype, device=self.device),
                cz,
            ])

            world_pos = panel_t[:, :3, :3] @ vis_local + panel_t[:, :3, 3]
            hidden_pos = panel_t[:, :3, :3] @ hide_local + panel_t[:, :3, 3]
            visible = self._cleaned_cells[:, cell_idx].unsqueeze(-1)
            marker_pos = torch.where(visible, world_pos, hidden_pos)
            marker.set_pose(Pose.create_from_pq(marker_pos, marker_q))

    def _get_brush_face_panel_local(self) -> torch.Tensor:
        """Returns (B, 4, 3): four brush-face corners in panel-local coords."""
        brush_t = self.brush.pose.to_transformation_matrix()
        if brush_t.ndim == 2:
            brush_t = brush_t.unsqueeze(0)
        B = brush_t.shape[0]

        face_link = torch.tensor(
            self.BRUSH_FACE_PTS_LINK, dtype=brush_t.dtype, device=self.device
        )  # (4, 3)
        # world = brush_pose * link
        face_world = (
            brush_t[:, :3, :3] @ face_link.T + brush_t[:, :3, 3:4]
        ).transpose(1, 2)  # (B, 4, 3)

        # to panel-local
        panel_t = self.panel.pose.to_transformation_matrix()
        if panel_t.ndim == 2:
            panel_t = panel_t.unsqueeze(0)
        flat = face_world.reshape(B * 4, 3)
        ones = torch.ones((B * 4, 1), dtype=flat.dtype, device=self.device)
        pts_h = torch.cat([flat, ones], dim=-1)
        panel_flat = torch.linalg.solve(panel_t, pts_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        return panel_flat.reshape(B, 4, 3)

    def _world_to_panel_local(self, points: torch.Tensor) -> torch.Tensor:
        panel_t = self.panel.pose.to_transformation_matrix()
        if panel_t.ndim == 2:
            panel_t = panel_t.unsqueeze(0)
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=self.device)
        points_h = torch.cat([points, ones], dim=-1)
        return torch.linalg.solve(panel_t, points_h.unsqueeze(-1)).squeeze(-1)[:, :3]

    def evaluate(self):
        self._ensure_clean_state()

        face_panel = self._get_brush_face_panel_local()  # (B, 4, 3)

        a, b, d = self.PANEL_SURFACE_A, self.PANEL_SURFACE_B, self.PANEL_SURFACE_D
        # Signed distance from each corner to the panel surface plane
        signed_dists = a * face_panel[:, :, 0] + b * face_panel[:, :, 1] + d  # (B, 4)
        # All 4 corners must be within tolerance
        all_near = (signed_dists.abs() <= self.clean_surface_tolerance).all(dim=-1)  # (B,)

        # Feet of perpendiculars on the panel surface (X-Z only, since nz=0)
        feet_x = face_panel[:, :, 0] - a * signed_dists  # (B, 4)
        feet_z = face_panel[:, :, 2]                       # (B, 4)

        x_min, x_max = feet_x.min(dim=-1).values, feet_x.max(dim=-1).values  # (B,)
        z_min, z_max = feet_z.min(dim=-1).values, feet_z.max(dim=-1).values  # (B,)

        # Cells whose X-Z center falls inside the feet bounding box
        cell_x = self._clean_cell_centers[:, 0]  # (N,)
        cell_z = self._clean_cell_centers[:, 1]  # (N,)
        covered = (
            (cell_x[None, :] >= x_min[:, None])
            & (cell_x[None, :] <= x_max[:, None])
            & (cell_z[None, :] >= z_min[:, None])
            & (cell_z[None, :] <= z_max[:, None])
        )  # (B, N)

        newly_cleaned = all_near[:, None] & covered
        self._cleaned_cells |= newly_cleaned
        self._update_clean_marker_poses()

        clean_coverage = self._cleaned_cells.to(torch.float32).mean(dim=-1)
        success = clean_coverage >= self.clean_success_ratio

        brush_p = self.brush.pose.p
        brush_height = brush_p[:, 2] if brush_p.ndim == 2 else brush_p[2]

        return {
            "success": success,
            "clean_coverage": clean_coverage,
            "cleaning_contact": all_near,
            "brush_height": brush_height,
            "brush_face_panel": face_panel,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            brush_pose=self.brush.pose.raw_pose,
            panel_pose=self.panel.pose.raw_pose,
        )
        if info.get("clean_coverage") is not None:
            obs["clean_coverage"] = info["clean_coverage"]
        if info.get("cleaning_contact") is not None:
            obs["cleaning_contact"] = info["cleaning_contact"].float()
        if info.get("brush_face_panel") is not None:
            brush_face_panel = info["brush_face_panel"]
            obs["brush_face_panel"] = brush_face_panel.reshape(
                brush_face_panel.shape[0], -1
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info.get("success", torch.tensor(False, device=self.device)).to(torch.float32)
