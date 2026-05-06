from typing import Any

import importlib.resources as importlib_resources
import math
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig, SceneConfig
from maniskill_myws.task_prompts import TASK_PROMPTS


_Q_BEAM_Y: list[float] = [math.sqrt(2.0) / 2.0, 0.0, 0.0, math.sqrt(2.0) / 2.0]


@register_env("TakeSafetyHook-v1", max_episode_steps=200)
class TakeSafetyHookEnv(BaseEnv):
    """
    Tabletop task: take a safety hook from a fixed horizontal rod.

    Scene:
      - Panda mounted at a fixed pose relative to the table
      - A horizontal rod is placed in front of the robot with small pose randomization
      - Safety hook is loaded as a free-root articulation and randomized along
        different positions on the rod

    Success:
      - Current proxy keeps the existing gate-opening progress criterion
    """

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["TakeSafetyHook-v1"]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        hook_init_qpos_noise: float = 0.1,
        success_threshold: float = np.pi / 4,
        beam_length: float = 0.7,
        beam_radius: float = 0.015,
        beam_center_x: float = 0.05,
        beam_center_y: float = 0.0,
        beam_center_z: float = 0.5,
        beam_center_x_noise: float = 0.025,
        beam_center_y_noise: float = 0.015,
        beam_center_z_noise: float = 0.02,
        beam_yaw_noise: float = math.radians(5),
        hook_on_beam_y_range: float = 0.18,
        hook_rod_local_x: float = -0.035,
        hook_top_local_z: float = 0.085,
        hook_rod_clearance: float = 0.002,
        hook_xy_noise: float | None = None,
        hook_yaw_noise: float | None = None,
        beam_x_range: tuple[float, float] | None = None,
        beam_y_range: tuple[float, float] | None = None,
        beam_z_range: tuple[float, float] | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.hook_init_qpos_noise = hook_init_qpos_noise
        self.success_threshold = float(success_threshold)

        if beam_x_range is not None:
            beam_center_x = sum(beam_x_range) / 2.0
            beam_center_x_noise = abs(beam_x_range[1] - beam_x_range[0]) / 2.0
        if beam_y_range is not None:
            beam_center_y = sum(beam_y_range) / 2.0
            beam_center_y_noise = abs(beam_y_range[1] - beam_y_range[0]) / 2.0
        if beam_z_range is not None:
            beam_center_z = sum(beam_z_range) / 2.0
            beam_center_z_noise = abs(beam_z_range[1] - beam_z_range[0]) / 2.0
        if hook_xy_noise is not None:
            hook_on_beam_y_range = hook_xy_noise
        if hook_yaw_noise is not None:
            beam_yaw_noise = hook_yaw_noise

        self.beam_length = float(beam_length)
        self.beam_radius = float(beam_radius)
        self.beam_center_x = float(beam_center_x)
        self.beam_center_y = float(beam_center_y)
        self.beam_center_z = float(beam_center_z)
        self.beam_center_x_noise = abs(float(beam_center_x_noise))
        self.beam_center_y_noise = abs(float(beam_center_y_noise))
        self.beam_center_z_noise = abs(float(beam_center_z_noise))
        self.beam_yaw_noise = abs(float(beam_yaw_noise))
        usable_half_length = max(self.beam_length / 2.0 - 0.04, 0.0)
        self.hook_on_beam_y_range = min(
            abs(float(hook_on_beam_y_range)), usable_half_length
        )
        self.hook_rod_local_x = float(hook_rod_local_x)
        self.hook_top_local_z = float(hook_top_local_z)
        self.hook_rod_clearance = float(hook_rod_clearance)

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
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        loader = self.scene.create_urdf_loader()
        loader.name = "safety_hook"
        loader.fix_root_link = False
        loader.disable_self_collisions = True

        # Use workspace-shipped asset (works even when ManiSkill is installed via pip).
        assets_dir = importlib_resources.files("maniskill_myws").joinpath("assets")
        with importlib_resources.as_file(assets_dir) as assets_path:
            urdf_path = assets_path / "safety_hook2/urdf/safety_hook.urdf"
            parsed = loader.parse(str(urdf_path), package_dir=str(assets_path))
            articulation_builders = parsed["articulation_builders"]
            actor_builders = parsed["actor_builders"]
            if len(articulation_builders) != 1 or actor_builders:
                raise RuntimeError(
                    "Expected safety_hook.urdf to contain exactly one articulation and no loose actors."
                )
            hook_builder = articulation_builders[0]
            hook_builder.set_scene_idxs(torch.arange(self.num_envs, dtype=torch.int32))
            hook_builder.disable_self_collisions = loader.disable_self_collisions
            # Keep the temporary build pose close to the later randomized spawn to avoid startup collisions.
            hook_builder.initial_pose = sapien.Pose(
                p=[self._hook_origin_x, self.beam_center_y, self._hook_origin_z]
            )
            self.hook: Articulation = hook_builder.build()

        self.gate_joint = self.hook.active_joints_map["joint_bar"]
        self.gate_joint.set_drive_properties(150.0, 15.0)
        self.gate_joint.set_drive_target(-0.02)
        self.gate_joint.set_friction(0.1)

        # beam: thin horizontal cylinder in mid-air, static/kinematic
        beam_builder = self.scene.create_actor_builder()
        beam_builder.initial_pose = sapien.Pose(
            p=[self.beam_center_x, self.beam_center_y, self.beam_center_z]
        )
        beam_builder.add_cylinder_collision(
            pose=sapien.Pose(q=_Q_BEAM_Y),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
        )
        beam_builder.add_cylinder_visual(
            pose=sapien.Pose(q=_Q_BEAM_Y),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
            material=sapien.render.RenderMaterial(base_color=[0.12, 0.36, 0.85, 1.0]),
        )
        self.beam = beam_builder.build_kinematic(name="beam")

    @property
    def _hook_origin_x(self) -> float:
        return self.beam_center_x - self.hook_rod_local_x

    @property
    def _hook_origin_z(self) -> float:
        return (
            self.beam_center_z
            - self.hook_top_local_z
            + self.beam_radius
            + self.hook_rod_clearance
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)

            # Beam stays in front of the robot with small pose domain randomization.
            beam_p = torch.zeros((b, 3), device=self.device)
            beam_p[:, 0] = self.beam_center_x + randomization.uniform(
                -self.beam_center_x_noise,
                self.beam_center_x_noise,
                size=(b,),
                device=self.device,
            )
            beam_p[:, 1] = self.beam_center_y + randomization.uniform(
                -self.beam_center_y_noise,
                self.beam_center_y_noise,
                size=(b,),
                device=self.device,
            )
            beam_p[:, 2] = self.beam_center_z + randomization.uniform(
                -self.beam_center_z_noise,
                self.beam_center_z_noise,
                size=(b,),
                device=self.device,
            )
            beam_yaw = randomization.uniform(
                -self.beam_yaw_noise,
                self.beam_yaw_noise,
                size=(b,),
                device=self.device,
            )
            half_yaw = beam_yaw * 0.5
            beam_q = torch.zeros((b, 4), dtype=torch.float32, device=self.device)
            beam_q[:, 0] = torch.cos(half_yaw)
            beam_q[:, 3] = torch.sin(half_yaw)
            self.beam.set_pose(Pose.create_from_pq(beam_p, beam_q))

            # Randomize the hook at different positions along the randomized rod.
            rod_offset = randomization.uniform(
                -self.hook_on_beam_y_range,
                self.hook_on_beam_y_range,
                size=(b,),
                device=self.device,
            )
            cos_yaw = torch.cos(beam_yaw)
            sin_yaw = torch.sin(beam_yaw)
            rod_axis_x = -sin_yaw
            rod_axis_y = cos_yaw
            rod_target_x = beam_p[:, 0] + rod_axis_x * rod_offset
            rod_target_y = beam_p[:, 1] + rod_axis_y * rod_offset
            rod_target_z = beam_p[:, 2] + self.beam_radius + self.hook_rod_clearance

            hook_p = torch.zeros((b, 3), device=self.device)
            hook_p[:, 0] = rod_target_x - cos_yaw * self.hook_rod_local_x
            hook_p[:, 1] = rod_target_y - sin_yaw * self.hook_rod_local_x
            hook_p[:, 2] = rod_target_z - self.hook_top_local_z
            hook_q = beam_q
            self.hook.set_pose(Pose.create_from_pq(hook_p, hook_q))

            qpos0 = torch.full((b, 1), -0.02, device=self.device)
            self.hook.set_qpos(qpos0)
            self.hook.set_qvel(torch.zeros((b, 1), device=self.device))

            self._hook_qpos_prev = qpos0[:, 0].clone()

    def evaluate(self):
        gate_qpos = self.gate_joint.qpos
        # We do a simple proxy for now; later change to beam-hang detection by keypoints.
        progress = torch.clamp((gate_qpos - 1.57) / (2.35 - 1.57), 0.0, 1.0)
        success = progress >= torch.tensor(0.75, device=self.device)

        return {
            "success": success,
            "progress": progress,
            "hook_qpos": gate_qpos,
            "beam_pose": self.beam.pose.raw_pose,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            hook_qpos=self.gate_joint.qpos,
            gate_progress=info.get("progress"),
        )
        if "state" in self.obs_mode:
            obs["hook_pose"] = self.hook.pose.raw_pose
            obs["beam_pose"] = self.beam.pose.raw_pose
            obs["hook_qpos"] = self.gate_joint.qpos
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].to(torch.float32)
