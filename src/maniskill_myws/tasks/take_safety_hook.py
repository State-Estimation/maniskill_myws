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
        beam_radius: float = 0.003,
        beam_center_x: float = 0.03,
        beam_center_y: float = 0.0,
        beam_center_z: float = 0.20,
        beam_center_x_noise: float = 0.015,
        beam_center_y_noise: float = 0.015,
        beam_center_z_noise: float = 0.02,
        beam_yaw_noise: float = 0.01,
        beam_pitch: float = math.pi / 2 ,
        hook_on_beam_y_range: float = 0.04,
        hook_on_beam_center_offset: float = 0.08,
        hook_rod_local_x: float = 0.0,
        hook_top_local_z: float | None = None,
        hook_rod_clearance: float = 0.002,
        # Cradle shelves: small plates bound to the rod that keep the hook seated at
        # reset. They form a U open toward the robot (left / right / back walls).
        hook_half_width_x: float = 0.028,
        hook_half_depth_y: float = 0.020,
        shelf_thickness: float = 0.004,
        shelf_height: float = 0.02,
        shelf_gap: float = 0.001,
        shelf_center_z: float = -0.0,
        upper_shelf_center_z: float = 0.013,
        gate_closed_qpos: float = 0.0,
        gate_open_qpos: float = -0.019,
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
        self.beam_pitch = float(beam_pitch)
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
        self.hook_half_width_x = abs(float(hook_half_width_x))
        self.hook_half_depth_y = abs(float(hook_half_depth_y))
        self.shelf_thickness = abs(float(shelf_thickness))
        self.shelf_height = abs(float(shelf_height))
        self.shelf_gap = float(shelf_gap)
        self.shelf_center_z = float(shelf_center_z)
        self.upper_shelf_center_z = float(upper_shelf_center_z)
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
        pose = sapien_utils.look_at([0.12, 0.08, self.beam_center_z+0.15], [0.0, 0.00, 0.0])
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

        # Cradle shelves: kinematic walls bound to the rod (built at the un-randomized
        # spawn pose; re-placed rigidly with the hook at every reset).
        self.shelves = []
        # At build time the assembly is at yaw 0, so local (lx,ly,lz) -> world
        # offset (-ly, lx, lz) relative to the hook origin.
        ox, oy, oz = self._hook_origin_x, self._hook_origin_y, self._hook_origin_z
        for name, (lx, ly, lz), half_extents in self._shelf_specs():
            shelf_builder = self.scene.create_actor_builder()
            shelf_builder.initial_pose = sapien.Pose(p=[ox - ly, oy + lx, oz + lz])
            shelf_builder.add_box_collision(half_size=half_extents)
            shelf_builder.add_box_visual(
                half_size=half_extents,
                material=sapien.render.RenderMaterial(base_color=[0.85, 0.62, 0.12, 1.0]),
            )
            setattr(self, name, shelf_builder.build_kinematic(name=name))
            self.shelves.append(getattr(self, name))

    @property
    def _hook_origin_x(self) -> float:
        return self.beam_center_x

    @property
    def _hook_origin_y(self) -> float:
        return self.beam_center_y - self.hook_rod_local_x

    @property
    def _hook_origin_z(self) -> float:
        return self.beam_center_z - self.hook_rod_local_z

    def _shelf_specs(self):
        """Cradle walls expressed in the hook's local frame.

        Each entry is (name, local_center, half_extents). Walls are bound to the
        rod: at reset they are placed at ``hook_pose`` offset by ``local_center``
        (rotated by the hook yaw), so the whole rod+shelves+hook assembly moves
        as one rigid body. The U is open toward the robot (local +Y), with walls
        on the left/right (local +/-X) and back (local -Y, the far side).
        """
        t = self.shelf_thickness
        h = self.shelf_height
        gap = self.shelf_gap
        hx = self.hook_half_width_x  # hook half-extent along local X (left/right)
        hy = self.hook_half_depth_y  # hook half-extent along local Y (along rod)
        zc = self.shelf_center_z
        uz = self.upper_shelf_center_z
        side_x = hx + gap + t / 2.0
        side_ly = hy + gap + t  # left/right walls span the hook depth + overlap
        back_y = -(hy + gap + t / 2.0)
        back_lx = hx + gap + t  # back wall spans the hook width + overlap
        return [
            # Side walls (U-shape open toward robot)
            ("shelf_left", (-side_x, 0.0, zc), (t / 2.0, side_ly, h / 2.0)),
            ("shelf_right", (side_x, 0.0, zc), (t / 2.0, side_ly, h / 2.0)),
            ("shelf_back", (0.0, back_y, zc), (back_lx, t / 2.0, h / 2.0)),
            # Top lid
            ("shelf_top", (0.0, 0.0, uz), (back_lx, side_ly, t / 2.0)),
        ]

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

            # The hook's position along the rod is fixed (its relative pose to the
            # rod is固化); only the whole rod+shelves+hook assembly is randomized
            # via beam_p / beam_yaw above, so the hook always sits flush in the cradle.
            rod_offset = self.hook_on_beam_center_offset
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

            # Place the cradle walls rigidly with the hook. local (lx,ly,lz) ->
            # world offset (-sin_yaw*lx - cos_yaw*ly, cos_yaw*lx - sin_yaw*ly, lz).
            for shelf, (_, (lx, ly, lz), _) in zip(self.shelves, self._shelf_specs()):
                wx = -sin_yaw * lx - cos_yaw * ly
                wy = cos_yaw * lx - sin_yaw * ly
                shelf_p = torch.stack(
                    [hook_p[:, 0] + wx, hook_p[:, 1] + wy, hook_p[:, 2] + lz],
                    dim=-1,
                )
                shelf.set_pose(Pose.create_from_pq(shelf_p, hook_q))

            # Rotate the entire rod+hook+shelves assembly around the world Y axis
            # by beam_pitch (default π/2). The beam center is the pivot point.
            pitch = self.beam_pitch
            if abs(pitch) > 1e-9:
                c = math.cos(pitch)
                s = math.sin(pitch)
                half = pitch * 0.5
                c_half = math.cos(half)
                s_half = math.sin(half)
                # q_pitch = (c_half, 0, s_half, 0)  in (w, x, y, z) order
                q_pitch_w = c_half
                q_pitch_y = s_half

                # --- Beam orientation ---
                # q_beam' = q_pitch * q_beam  (left-multiply: world-frame rotation)
                bw, bx, by, bz = beam_q[:, 0], beam_q[:, 1], beam_q[:, 2], beam_q[:, 3]
                beam_q_new = torch.stack(
                    [
                        q_pitch_w * bw - q_pitch_y * by,                # w
                        q_pitch_w * bx + q_pitch_y * bz,                # x
                        q_pitch_y * bw + q_pitch_w * by,                # y
                        q_pitch_w * bz - q_pitch_y * bx,                # z
                    ],
                    dim=-1,
                )
                self.beam.set_pose(Pose.create_from_pq(beam_p, beam_q_new))

                # --- Hook position (rotate around beam center) ---
                hook_rel = hook_p - beam_p  # (b, 3)
                hook_rel_rot = torch.stack(
                    [
                        c * hook_rel[:, 0] + s * hook_rel[:, 2],
                        hook_rel[:, 1],
                        -s * hook_rel[:, 0] + c * hook_rel[:, 2],
                    ],
                    dim=-1,
                )
                hook_p_rot = beam_p + hook_rel_rot
                # --- Hook orientation ---
                hw, hx, hy, hz = hook_q[:, 0], hook_q[:, 1], hook_q[:, 2], hook_q[:, 3]
                hook_q_rot = torch.stack(
                    [
                        q_pitch_w * hw - q_pitch_y * hy,                # w
                        q_pitch_w * hx + q_pitch_y * hz,                # x
                        q_pitch_y * hw + q_pitch_w * hy,                # y
                        q_pitch_w * hz - q_pitch_y * hx,                # z
                    ],
                    dim=-1,
                )
                self.hook.set_pose(Pose.create_from_pq(hook_p_rot, hook_q_rot))

                # --- Shelf poses (rotate around beam center) ---
                for shelf in self.shelves:
                    sp = shelf.pose.p  # current pos after flat-frame placement
                    sq = shelf.pose.q
                    shelf_rel = sp - beam_p
                    shelf_rel_rot = torch.stack(
                        [
                            c * shelf_rel[:, 0] + s * shelf_rel[:, 2],
                            shelf_rel[:, 1],
                            -s * shelf_rel[:, 0] + c * shelf_rel[:, 2],
                        ],
                        dim=-1,
                    )
                    shelf_p_rot = beam_p + shelf_rel_rot
                    sw, sx, sy, sz = sq[:, 0], sq[:, 1], sq[:, 2], sq[:, 3]
                    shelf_q_rot = torch.stack(
                        [
                            q_pitch_w * sw - q_pitch_y * sy,            # w
                            q_pitch_w * sx + q_pitch_y * sz,            # x
                            q_pitch_y * sw + q_pitch_w * sy,            # y
                            q_pitch_w * sz - q_pitch_y * sx,            # z
                        ],
                        dim=-1,
                    )
                    shelf.set_pose(Pose.create_from_pq(shelf_p_rot, shelf_q_rot))

            qpos0 = torch.full((b, 1), self.gate_closed_qpos, device=self.device)
            self.hook.set_qpos(qpos0)
            self.hook.set_qvel(torch.zeros((b, 1), device=self.device))
            self.gate_joint.set_drive_target(self.gate_closed_qpos)
            self.gate_joint.set_drive_velocity_target(0.0)

            self._hook_qpos_prev = qpos0[:, 0].clone()
            self._max_progress = torch.zeros((b,), device=self.device)
            self._min_hook_com_z = torch.full((b,), float("inf"), device=self.device)

    def evaluate(self):
        gate_qpos = self.gate_joint.qpos
        gate_angle = gate_qpos[:, 0] if gate_qpos.ndim == 2 else gate_qpos
        open_span = max(self.gate_closed_qpos - self.gate_open_qpos, 1e-6)
        cur_progress = torch.clamp(
            (self.gate_closed_qpos - gate_angle) / open_span, 0.0, 1.0
        )

        # Hook COM height (world Z).  frame_link has mass=100, so the root-link
        # world Z is a tight approximation of the full-articulation COM Z.
        cur_hook_com_z = self.hook.pose.p[:, 2]

        # Track historical best: the gate has a spring that pushes it closed,
        # and the hook can bounce back up, so success should latch on the
        # peak opening and the lowest COM Z reached so far this episode.
        self._max_progress = torch.maximum(self._max_progress, cur_progress)
        self._min_hook_com_z = torch.minimum(self._min_hook_com_z, cur_hook_com_z)

        gate_open_enough = self._max_progress >= 0.5
        hook_low_enough = self._min_hook_com_z < 0.03
        success = gate_open_enough & hook_low_enough

        return {
            "success": success,
            "progress": self._max_progress,
            "hook_qpos": gate_angle,
            "hook_com_z": self._min_hook_com_z,
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
