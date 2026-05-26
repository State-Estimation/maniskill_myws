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


_Q_BEAM_X: list[float] = [1.0, 0.0, 0.0, 0.0]
_Q_HOOK_NARROW_DOWN_ON_X_BEAM: list[float] = [
    math.sqrt(2.0) / 2.0,
    0.0,
    0.0,
    math.sqrt(2.0) / 2.0,
]


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

    ROBOT_HOME_QPOS_PANDA = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )
    ROBOT_HOME_QPOS_PANDA_WRISTCAM = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        hook_init_qpos_noise: float = 0.1,
        success_threshold: float = np.pi / 4,
        beam_length: float = 0.7,
        beam_robot_side_length: float = 0.06,
        beam_far_side_length: float | None = None,
        beam_end_margin: float = 0.04,
        beam_radius: float = 0.005,
        beam_center_x: float = -0.03,
        beam_center_y: float = 0.0,
        beam_center_z: float = 0.60,
        beam_center_x_noise: float = 0.015,
        beam_center_y_noise: float = 0.015,
        beam_center_z_noise: float = 0.02,
        beam_yaw_noise: float = 0.0,
        hook_on_beam_y_range: float = 0.04,
        hook_on_beam_center_offset: float = 0.02,
        hook_rod_local_x: float = 0.0,
        hook_top_local_z: float | None = None,
        hook_rod_clearance: float = 0.002,
        gate_closed_qpos: float = 0.09,
        gate_open_qpos: float = -0.55,
        gate_spring_stiffness: float = 0.5,
        gate_spring_damping: float = 0.15,
        gate_force_limit: float = 0.8,
        gate_friction: float = 0.01,
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

        beam_length = abs(float(beam_length))
        self.beam_robot_side_length = abs(float(beam_robot_side_length))
        if beam_far_side_length is None:
            beam_far_side_length = max(
                beam_length - self.beam_robot_side_length, 0.04
            )
        self.beam_far_side_length = abs(float(beam_far_side_length))
        self.beam_length = self.beam_robot_side_length + self.beam_far_side_length
        self.beam_local_center_x = (
            self.beam_far_side_length - self.beam_robot_side_length
        ) / 2.0
        self.beam_end_margin = min(abs(float(beam_end_margin)), self.beam_length / 2.0)
        self.beam_radius = float(beam_radius)
        self.beam_center_x = float(beam_center_x)
        self.beam_center_y = float(beam_center_y)
        self.beam_center_z = float(beam_center_z)
        self.beam_center_x_noise = abs(float(beam_center_x_noise))
        self.beam_center_y_noise = abs(float(beam_center_y_noise))
        self.beam_center_z_noise = abs(float(beam_center_z_noise))
        self.beam_yaw_noise = abs(float(beam_yaw_noise))
        hook_min_offset = -self.beam_robot_side_length + self.beam_end_margin
        hook_max_offset = self.beam_far_side_length - self.beam_end_margin
        if hook_min_offset > hook_max_offset:
            hook_on_beam_center_offset = 0.0
            usable_half_length = 0.0
        else:
            hook_on_beam_center_offset = min(
                max(float(hook_on_beam_center_offset), hook_min_offset),
                hook_max_offset,
            )
            usable_half_length = max(
                min(
                    hook_on_beam_center_offset - hook_min_offset,
                    hook_max_offset - hook_on_beam_center_offset,
                ),
                0.0,
            )
        self.hook_on_beam_center_offset = float(hook_on_beam_center_offset)
        # Legacy name kept for compatibility; this is now measured along the beam axis.
        self.hook_on_beam_y_range = min(
            abs(float(hook_on_beam_y_range)), usable_half_length
        )
        self.hook_rod_local_x = float(hook_rod_local_x)
        self.hook_rod_clearance = float(hook_rod_clearance)
        if hook_top_local_z is None:
            hook_top_local_z = -(self.beam_radius + self.hook_rod_clearance)
        # Legacy name: this now means the rod centerline Z in the hook frame.
        self.hook_top_local_z = float(hook_top_local_z)
        self.hook_rod_local_z = self.hook_top_local_z
        self.gate_closed_qpos = float(gate_closed_qpos)
        self.gate_open_qpos = float(gate_open_qpos)
        self.gate_spring_stiffness = float(gate_spring_stiffness)
        self.gate_spring_damping = float(gate_spring_damping)
        self.gate_force_limit = float(gate_force_limit)
        self.gate_friction = float(gate_friction)

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
        pose = sapien_utils.look_at([-0.3, -0.2, self.beam_center_z+0.05], [0.0, 0.05, 0.2])
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
            urdf_path = assets_path / "hook2/urdf/hook2.urdf"
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
                p=[self._hook_origin_x, self._hook_origin_y, self._hook_origin_z],
                q=_Q_HOOK_NARROW_DOWN_ON_X_BEAM,
            )
            self.hook: Articulation = hook_builder.build()

        self.gate_joint = self.hook.active_joints_map["joint_bar"]
        self.gate_joint.set_limits(
            np.array(
                [
                    [
                        min(self.gate_open_qpos, self.gate_closed_qpos),
                        max(self.gate_open_qpos, self.gate_closed_qpos),
                    ]
                ],
                dtype=np.float32,
            )
        )
        self.gate_joint.set_drive_properties(
            self.gate_spring_stiffness,
            self.gate_spring_damping,
            self.gate_force_limit,
        )
        self.gate_joint.set_drive_target(self.gate_closed_qpos)
        self.gate_joint.set_drive_velocity_target(0.0)
        self.gate_joint.set_friction(self.gate_friction)

        # Beam origin is the manipulation reference point; the cylinder is shorter
        # on the robot side to keep it clear of the arm at reset.
        beam_builder = self.scene.create_actor_builder()
        beam_builder.initial_pose = sapien.Pose(
            p=[self.beam_center_x, self.beam_center_y, self.beam_center_z]
        )
        beam_builder.add_cylinder_collision(
            pose=sapien.Pose(p=[self.beam_local_center_x, 0, 0], q=_Q_BEAM_X),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
        )
        beam_builder.add_cylinder_visual(
            pose=sapien.Pose(p=[self.beam_local_center_x, 0, 0], q=_Q_BEAM_X),
            radius=self.beam_radius,
            half_length=self.beam_length / 2,
            material=sapien.render.RenderMaterial(base_color=[0.12, 0.36, 0.85, 1.0]),
        )
        self.beam = beam_builder.build_kinematic(name="beam")

    @property
    def _hook_origin_x(self) -> float:
        return self.beam_center_x

    @property
    def _hook_origin_y(self) -> float:
        return self.beam_center_y - self.hook_rod_local_x

    @property
    def _hook_origin_z(self) -> float:
        return self.beam_center_z - self.hook_rod_local_z

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            self._reset_robot_retracted_qpos(env_idx)
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
            rod_offset = self.hook_on_beam_center_offset + randomization.uniform(
                -self.hook_on_beam_y_range,
                self.hook_on_beam_y_range,
                size=(b,),
                device=self.device,
            )
            cos_yaw = torch.cos(beam_yaw)
            sin_yaw = torch.sin(beam_yaw)
            rod_axis_x = cos_yaw
            rod_axis_y = sin_yaw
            rod_target_x = beam_p[:, 0] + rod_axis_x * rod_offset
            rod_target_y = beam_p[:, 1] + rod_axis_y * rod_offset
            rod_target_z = beam_p[:, 2]

            hook_p = torch.zeros((b, 3), device=self.device)
            hook_local_x_world_x = -sin_yaw
            hook_local_x_world_y = cos_yaw
            hook_p[:, 0] = (
                rod_target_x - hook_local_x_world_x * self.hook_rod_local_x
            )
            hook_p[:, 1] = (
                rod_target_y - hook_local_x_world_y * self.hook_rod_local_x
            )
            hook_p[:, 2] = rod_target_z - self.hook_rod_local_z
            # Keep the hook on the beam with its narrow local -Z end pointing down.
            half_hook_yaw = (beam_yaw + math.pi / 2.0) * 0.5
            hook_q = torch.stack(
                [
                    torch.cos(half_hook_yaw),
                    torch.zeros_like(half_hook_yaw),
                    torch.zeros_like(half_hook_yaw),
                    torch.sin(half_hook_yaw),
                ],
                dim=-1,
            )
            self.hook.set_pose(Pose.create_from_pq(hook_p, hook_q))

            qpos0 = torch.full((b, 1), self.gate_closed_qpos, device=self.device)
            self.hook.set_qpos(qpos0)
            self.hook.set_qvel(torch.zeros((b, 1), device=self.device))
            self.gate_joint.set_drive_target(self.gate_closed_qpos)
            self.gate_joint.set_drive_velocity_target(0.0)

            self._hook_qpos_prev = qpos0[:, 0].clone()

    def evaluate(self):
        gate_qpos = self.gate_joint.qpos
        # We do a simple proxy for now; later change to beam-hang detection by keypoints.
        gate_angle = gate_qpos[:, 0] if gate_qpos.ndim == 2 else gate_qpos
        open_span = max(self.gate_closed_qpos - self.gate_open_qpos, 1e-6)
        progress = torch.clamp(
            (self.gate_closed_qpos - gate_angle) / open_span, 0.0, 1.0
        )
        success = progress >= torch.tensor(1.2, device=self.device)

        return {
            "success": success,
            "progress": progress,
            "hook_qpos": gate_angle,
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
