from __future__ import annotations

import numpy as np

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from maniskill_myws.task_prompts import TASK_PROMPTS
import sapien
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig, SceneConfig

@register_env("StackCube-v2", max_episode_steps=200)
class StackCubeV2Env(StackCubeEnv):
    """
    StackCube variant with standardized sensors for VLA data collection.

    Key differences vs StackCube-v1:
      - Standardized base + side camera for dataset collection.
      - Default robot is panda_wristcam (provides wrist camera).
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "panda_wristcam_custom_rot"]
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["StackCube-v2"]
    ROBOT_HOME_QPOS_PANDA = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )
    ROBOT_HOME_QPOS_PANDA_WRISTCAM = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )
    ROBOT_HOME_QPOS_PANDA_WRISTCAM_CUSTOM_ROT = np.array(
        [0.008, 0.105, 0.029, -2.747, 0.002, 2.772, 0.870, 0.04, 0.04],
        dtype=np.float32,
    )

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1])
        side_pose = sapien_utils.look_at(eye=[-0.35, 0.35, 0.35], target=[0.0, 0.0, 0.1])
        return [
            CameraConfig("base_camera", pose=base_pose, width=128, height=128, fov=np.pi / 2),
            CameraConfig("side_camera", pose=side_pose, width=128, height=128, fov=np.pi / 2),
        ]

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=20,scene_config=SceneConfig(gravity=[0, 0, -0.00098]))

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)
    
    def _load_scene(self, options):
        super()._load_scene(options)
        mark_builder = self.scene.create_actor_builder()
        mark_builder.add_box_visual(
            pose=sapien.Pose(),
            half_size=(0.05, 0.05, 0.05),
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1.0]),
        )
        self.mark = mark_builder.build_kinematic(name="mark")

    def _reset_robot_retracted_qpos(self, env_idx: torch.Tensor):
        b = len(env_idx)
        if self.robot_uids == "panda":
            base_qpos = self.ROBOT_HOME_QPOS_PANDA
        elif self.robot_uids == "panda_wristcam":
            base_qpos = self.ROBOT_HOME_QPOS_PANDA_WRISTCAM
        elif self.robot_uids == "panda_wristcam_custom_rot":
            base_qpos = self.ROBOT_HOME_QPOS_PANDA_WRISTCAM_CUSTOM_ROT
        else:
            raise ValueError(f"Unsupported robot_uids: {self.robot_uids}")
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
        super()._initialize_episode(env_idx, options)
        self._reset_robot_retracted_qpos(env_idx)
        with torch.device(self.device):
            b = len(env_idx)
            mark_p = torch.zeros((b, 3), device=self.device)
            mark_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)

            # 将 mark 放在桌面前方固定位置（可根据需求改为随机化）
            mark_p[:, 0] = 0.0
            mark_p[:, 1] = 0.25
            mark_p[:, 2] = 0.05

            self.mark.set_pose(Pose.create_from_pq(mark_p, mark_q))