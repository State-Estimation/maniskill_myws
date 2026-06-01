import math
from typing import Any
import sapien

from mani_skill.agents.robots import Panda

from mani_skill.sensors.camera import CameraConfig
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


@register_env("SolarPanelStatic-v1", max_episode_steps=200)
class SolarPanelStaticEnv(BaseEnv):

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda

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

    BRUSH_LOCAL_MIN_X: float = -0.005642
    BRUSH_HEAD_LOCAL_POS: tuple[float, float, float] = (0.0260, 0.0343, 0.0)
    BRUSH_HANDLE_CENTER: tuple[float, float, float] = (0.0036, -0.0362, 0.0)
    BRUSH_HANDLE_RADIUS: float = 0.0090
    BRUSH_HANDLE_HALF_LENGTH: float = 0.0725
    BRUSH_HEAD_BOX_CENTER: tuple[float, float, float] = (0.0260, 0.0343, 0.0)
    BRUSH_HEAD_BOX_HALF_SIZE: tuple[float, float, float] = (0.0132, 0.0218, 0.0709)
    BRUSH_TABLE_CLEARANCE: float = 0.005
    BRUSH_MAX_WORLD_X_RADIUS: float = 0.075
    BRUSH_PANEL_CLEARANCE: float = 0.04
    BRUSH_LIE_Q: tuple[float, float, float, float] = (
        -math.sqrt(2.0) / 2.0,
        0.0,
        -math.sqrt(2.0) / 2.0,
        0.0,
    )
    BRUSH_REST_Z: float = -BRUSH_LOCAL_MIN_X + BRUSH_TABLE_CLEARANCE

    ROBOT_HOME_QPOS_PANDA = np.array(
        [0.0, 0.55, 0.0, -2.05, 0.0, 2.45, np.pi / 4, 0.04, 0.04],
        dtype=np.float32,
    )
    ROBOT_HOME_QPOS_PANDA_WRISTCAM = np.array(
        [0.0, 0.55, 0.0, -2.05, 0.0, 2.45, -np.pi / 4, 0.04, 0.04],
        dtype=np.float32,
    )

    CLEAN_MARKER_SURFACE_OFFSET: float = 0.004
    CLEAN_MARKER_THICKNESS: float = 0.0015
    CLEAN_MARKER_HIDE_Z: float = -10.0

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_base_x: float = -0.615,
        robot_base_y: float = 0.0,
        robot_init_qpos_noise: float = 0.02,
        brush_spawn_center_x: float = -0.10,
        brush_spawn_center_y: float = 0.0,
        brush_xy_noise: float = 0.01,
        brush_y_noise: float = 0.02,
        brush_z: float | None = None,
        brush_yaw_noise: float = math.radians(3),
        brush_mass: float = 0.35,
        panel_spawn_center_x: float = 0.38,
        panel_spawn_center_y: float = 0.0,
        panel_spawn_half_size_x: float = 0.035,
        panel_spawn_half_size_y: float = 0.03,
        panel_yaw_noise: float = math.radians(5),
        clean_grid_x: int = 5,
        clean_grid_y: int = 7,
        clean_radius: float = 0.18,
        clean_surface_tolerance: float = 0.08,
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
        self.brush_mass = brush_mass
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
        pose = sapien_utils.look_at([-0.7, -0.6, 0.9], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[self.robot_base_x, self.robot_base_y, 0])
        )

    def _load_scene(self, options):
        # 先建基础场景（桌子 + 机器人）
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # 资源路径
        base_dir = importlib_resources.files("maniskill_myws").joinpath("assets/solar_panel")
        with importlib_resources.as_file(base_dir) as d:

            # =========================
            # 静态物体：太阳能板
            # =========================
            panel_builder = self.scene.create_actor_builder()
            panel_builder.add_visual_from_file(
                str(d / "mesh/solar_panel.obj")
            )
            panel_builder.add_nonconvex_collision_from_file(
                str(d / "mesh/solar_panel.obj")
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
            # 设置位置（很重要！不然默认在原点）
            self.panel.set_pose(panel_pose)

            self.clean_markers = []
            marker_mat = sapien.render.RenderMaterial(
                base_color=[0.78, 0.96, 1.0, 0.62]
            )
            cell_half_x = (
                self.PANEL_LOCAL_MAX_X - self.PANEL_LOCAL_MIN_X
            ) / self.clean_grid_x * 0.47
            cell_half_z = (
                self.PANEL_LOCAL_MAX_Z - self.PANEL_LOCAL_MIN_Z
            ) / self.clean_grid_y * 0.47
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
                        self.CLEAN_MARKER_HIDE_Z,
                    ],
                    q=list(self.PANEL_BASE_Q),
                )
                self.clean_markers.append(
                    marker_builder.build_static(name=f"clean_marker_{i}")
                )

            # =========================
            #  可移动刷子（动态，保留 SolidWorks 外观，改用简单碰撞）
            # =========================
            brush_builder = self.scene.create_actor_builder()
            brush_density = self._compute_brush_density()
            base_dir = importlib_resources.files("maniskill_myws").joinpath("assets/brush/meshes")
            with importlib_resources.as_file(base_dir) as brush_mesh_dir:
                brush_builder.add_visual_from_file(str(brush_mesh_dir / "base_link.STL"))
            brush_builder.add_cylinder_collision(
                pose=sapien.Pose(
                    p=list(self.BRUSH_HANDLE_CENTER),
                    q=[math.sqrt(2.0) / 2.0, 0.0, 0.0, math.sqrt(2.0) / 2.0],
                ),
                radius=self.BRUSH_HANDLE_RADIUS,
                half_length=self.BRUSH_HANDLE_HALF_LENGTH,
                density=brush_density,
            )
            brush_builder.add_box_collision(
                pose=sapien.Pose(p=list(self.BRUSH_HEAD_BOX_CENTER)),
                half_size=list(self.BRUSH_HEAD_BOX_HALF_SIZE),
                density=brush_density,
            )
            brush_pose = sapien.Pose(
                p=[
                    self.brush_spawn_center_x,
                    self.brush_spawn_center_y,
                    self.brush_z,
                ],
                q=list(self.BRUSH_LIE_Q),
            )
            brush_builder.initial_pose = brush_pose
            self.brush = brush_builder.build(name="brush")
            self.brush.set_pose(brush_pose)

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
        q_lie = torch.zeros((b, 4), device=self.device)
        q_lie[:, 0] = self.BRUSH_LIE_Q[0]
        q_lie[:, 2] = self.BRUSH_LIE_Q[2]
        q = self._quat_mul(q_yaw, q_lie)

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

    def _compute_brush_density(self) -> float:
        handle_volume = (
            math.pi * self.BRUSH_HANDLE_RADIUS**2 * (2.0 * self.BRUSH_HANDLE_HALF_LENGTH)
        )
        head_half_size = self.BRUSH_HEAD_BOX_HALF_SIZE
        head_volume = 8.0 * head_half_size[0] * head_half_size[1] * head_half_size[2]
        total_volume = handle_volume + head_volume
        return self.brush_mass / max(total_volume, 1e-6)

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
            self.PANEL_LOCAL_MIN_X,
            self.PANEL_LOCAL_MAX_X,
            self.clean_grid_x + 1,
            device=self.device,
        )
        z_edges = torch.linspace(
            self.PANEL_LOCAL_MIN_Z,
            self.PANEL_LOCAL_MAX_Z,
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
        panel_raw_pose = self.panel.pose.raw_pose
        panel_q = (
            panel_raw_pose[3:7].unsqueeze(0)
            if panel_raw_pose.ndim == 1
            else panel_raw_pose[:, 3:7]
        )

        hidden_pos = torch.zeros((self.num_envs, 3), dtype=panel_t.dtype, device=self.device)
        hidden_pos[:, 2] = self.CLEAN_MARKER_HIDE_Z

        for cell_idx, marker in enumerate(self.clean_markers):
            center = self._clean_cell_centers[cell_idx]
            local_pos = torch.stack(
                [
                    center[0],
                    torch.as_tensor(
                        self.PANEL_LOCAL_MAX_Y + self.CLEAN_MARKER_SURFACE_OFFSET,
                        dtype=panel_t.dtype,
                        device=self.device,
                    ),
                    center[1],
                ]
            )
            world_pos = panel_t[:, :3, :3] @ local_pos + panel_t[:, :3, 3]
            visible = self._cleaned_cells[:, cell_idx].unsqueeze(-1)
            marker_pos = torch.where(visible, world_pos, hidden_pos)
            marker.set_pose(Pose.create_from_pq(marker_pos, panel_q))

    def _get_brush_head_world_pos(self) -> torch.Tensor:
        brush_t = self.brush.pose.to_transformation_matrix()
        if brush_t.ndim == 2:
            brush_t = brush_t.unsqueeze(0)
        head_local = torch.tensor(
            self.BRUSH_HEAD_LOCAL_POS, dtype=brush_t.dtype, device=self.device
        )
        return brush_t[:, :3, :3] @ head_local + brush_t[:, :3, 3]

    def _world_to_panel_local(self, points: torch.Tensor) -> torch.Tensor:
        panel_t = self.panel.pose.to_transformation_matrix()
        if panel_t.ndim == 2:
            panel_t = panel_t.unsqueeze(0)
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=self.device)
        points_h = torch.cat([points, ones], dim=-1)
        return torch.linalg.solve(panel_t, points_h.unsqueeze(-1)).squeeze(-1)[:, :3]

    def evaluate(self):
        self._ensure_clean_state()

        head_world = self._get_brush_head_world_pos()
        head_local = self._world_to_panel_local(head_world)

        x_in_bounds = (head_local[:, 0] >= self.PANEL_LOCAL_MIN_X) & (
            head_local[:, 0] <= self.PANEL_LOCAL_MAX_X
        )
        z_in_bounds = (head_local[:, 2] >= self.PANEL_LOCAL_MIN_Z) & (
            head_local[:, 2] <= self.PANEL_LOCAL_MAX_Z
        )
        near_surface = (
            torch.abs(head_local[:, 1] - self.PANEL_LOCAL_MAX_Y)
            <= self.clean_surface_tolerance
        )
        cleaning_contact = x_in_bounds & z_in_bounds & near_surface

        brush_axis = torch.stack([head_local[:, 0], head_local[:, 2]], dim=-1)
        dists = torch.norm(
            brush_axis[:, None, :] - self._clean_cell_centers[None, :, :], dim=-1
        )
        newly_cleaned = cleaning_contact[:, None] & (dists <= self.clean_radius)
        self._cleaned_cells |= newly_cleaned
        self._update_clean_marker_poses()

        clean_coverage = self._cleaned_cells.to(torch.float32).mean(dim=-1)
        success = clean_coverage >= self.clean_success_ratio

        brush_p = self.brush.pose.p
        brush_height = brush_p[:, 2] if brush_p.ndim == 2 else brush_p[2]

        return {
            "success": success,
            "clean_coverage": clean_coverage,
            "cleaning_contact": cleaning_contact,
            "brush_height": brush_height,
            "brush_head_local": head_local,
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
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info.get("success", torch.tensor(False, device=self.device)).to(torch.float32)



            

    
