from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.registration import register_agent


@register_agent()
class PandaWristCamCustomRot(PandaWristCam):
    """Panda arm robot with wrist camera, with increased rotation range for pd_ee_delta_pose."""

    uid = "panda_wristcam_custom_rot"
    urdf_path = PandaWristCam.urdf_path

    @property
    def _sensor_configs(self):
        return super()._sensor_configs

    @property
    def _controller_configs(self):
        configs = super()._controller_configs

        configs["pd_ee_delta_pose"]["arm"].rot_lower = -0.5
        configs["pd_ee_delta_pose"]["arm"].rot_upper = 0.5

        configs["pd_ee_target_delta_pose"]["arm"].rot_lower = -0.5
        configs["pd_ee_target_delta_pose"]["arm"].rot_upper = 0.5

        return configs
