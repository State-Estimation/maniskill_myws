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

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        brush_xy_noise: float = 0.05,
        brush_y_noise: float = 0.05,
        brush_z: float = 0.055,
        brush_mass: float = 0.35,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.brush_xy_noise = brush_xy_noise
        self.brush_y_noise = brush_y_noise
        self.brush_z = brush_z
        self.brush_mass = brush_mass
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=20,scene_config=SceneConfig(gravity=[0, 0, -0.00098]))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.45, 0.0, 0.35], [0.0, 0.0, 0.10])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.6, 0.9], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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
            self.panel = panel_builder.build_static(name="solar_panel")
            # 设置位置（很重要！不然默认在原点）
            self.panel.set_pose(
                sapien.Pose(p=[0.5, 0.0, 0.0], q=[0.707, 0.707, 0, 0])  # 根据桌子高度调
            )

            # =========================
            #  可移动刷子（动态，URDF 的质量/惯量/材质信息会生效）
            # =========================
            self.brush = None
            brush_loader = self.scene.create_urdf_loader()
            brush_loader.fix_root_link = False
            brush_loader.name = "brush"

            base_dir = importlib_resources.files("maniskill_myws").joinpath("assets/brush/urdf")
            with importlib_resources.as_file(base_dir) as brush_dir:
                urdf_path = brush_dir / "brush.urdf"
                # ManiSkill's URDFLoader wraps parse() and load(), but its inherited
                # load_multiple() is incompatible with the wrapped parse() return type.
                parsed = brush_loader.parse(str(urdf_path), package_dir=str(brush_dir))
                articulations = parsed["articulation_builders"]
                actors = parsed["actor_builders"]
                scene_idxs = torch.arange(self.num_envs, dtype=torch.int32)
                if len(articulations) > 0:
                    builder = articulations[0]
                    builder.set_scene_idxs(scene_idxs)
                    builder.disable_self_collisions = brush_loader.disable_self_collisions
                    self.brush = builder.build()
                elif len(actors) > 0:
                    builder = actors[0]
                    builder.set_scene_idxs(scene_idxs)
                    self.brush = builder.build(builder.name)
                else:
                    raise RuntimeError("No object loaded from brush URDF")

                # 初始位置，后续每个 episode 会随机
                if hasattr(self.brush, "set_root_pose"):
                    self.brush.set_root_pose(
                        sapien.Pose(
                            p=[0.1, 0.0, self.brush_z],
                            q=[1, 0, 0, 0],
                        )
                    )
                else:
                    self.brush.set_pose(
                        sapien.Pose(
                            p=[0.1, 0.0, self.brush_z],
                            q=[1, 0, 0, 0],
                        )
                    )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 保留桌面与机器人初始化
        self.scene_builder.initialize(env_idx)

        # 随机放置刷子位置（每个艺术体单环境）
        # 如果 num_envs=1，这里也正常
        b = len(env_idx)
        p = torch.zeros((b, 3), device=self.device)
        p[:, 0] = -0.2 + randomization.uniform(
            -self.brush_xy_noise, self.brush_xy_noise, size=(b,), device=self.device
        )
        p[:, 1] = randomization.uniform(
            -self.brush_y_noise, self.brush_y_noise, size=(b,), device=self.device
        )
        p[:, 2] = self.brush_z

        q = randomization.random_quaternions(
            n=b,
            device=self.device,
            lock_x=True,
            lock_y=True,
            bounds=(0, 0),
        )

        self.brush.set_pose(Pose.create_from_pq(p, q))

    def evaluate(self):
        # 抓取判定：刷子被提升到一定高度
        p = self.brush.pose.p
        if isinstance(p, torch.Tensor):
            if p.ndim == 2:
                z = p[:, 2]
            else:
                z = p[2]
        else:
            # numpy or list
            p = np.asarray(p)
            if p.ndim == 2:
                z = p[:, 2]
            else:
                z = p[2]

        success = z > self.brush_z + 0.05
        return {"success": success, "brush_height": z}

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info.get("success", torch.tensor(False, device=self.device)).to(torch.float32)



            

    
