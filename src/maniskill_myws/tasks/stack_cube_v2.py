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

@register_env("StackCube-v2", max_episode_steps=200)
class StackCubeV2Env(StackCubeEnv):
    """
    StackCube variant with standardized sensors for VLA data collection.

    Key differences vs StackCube-v1:
      - Standardized base + side camera for dataset collection.
      - Default robot is panda_wristcam (provides wrist camera).
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda"]
    DEFAULT_TASK_PROMPT = TASK_PROMPTS["StackCube-v2"]

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
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            mark_p = torch.zeros((b, 3), device=self.device)
            mark_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)

            # 将 mark 放在桌面前方固定位置（可根据需求改为随机化）
            mark_p[:, 0] = 0.0
            mark_p[:, 1] = 0.25
            mark_p[:, 2] = 0.05

            self.mark.set_pose(Pose.create_from_pq(mark_p, mark_q))