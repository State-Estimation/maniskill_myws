"""
VR Teleoperation for Panda (Relative / Delta Mode)
Mode: pd_ee_delta_pose
Logic: Calculate (Current_VR - Last_VR) * Transform_Matrix -> Action
"""

import os
import sys
import threading
import time
import numpy as np
import gymnasium as gym
import asyncio
import tyro
from dataclasses import dataclass
from typing import Annotated
import transforms3d.quaternions as tf_quat
from xlevr.inputs.vr_ws_server2 import ControlGoal
# =========================
# Path Setup
# =========================
XLEVR_PATH = "/home/firedust/mimic/maniskill_myws/third_party/XLeVR"
def setup_xlevr_environment():
    if XLEVR_PATH not in sys.path:
        sys.path.insert(0, XLEVR_PATH)
    os.chdir(XLEVR_PATH)
    os.environ["PYTHONPATH"] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"

setup_xlevr_environment()

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs import SimConfig
import maniskill_myws.tasks
from mani_skill.utils.structs.pose import Pose

# =========================
# Thread Safe Buffer & VR Thread
# =========================
class ThreadSafeLatestGoal:
    def __init__(self):
        self._lock = threading.Lock()
        self._goal = None

    def set(self, goal):
        with self._lock:
            self._goal = goal

    def get(self):
        with self._lock:
            return self._goal

class VRInputThread(threading.Thread):
    def __init__(self, latest_goal: ThreadSafeLatestGoal):
        super().__init__(daemon=True)
        self.latest_goal = latest_goal
        self.running = True

    def run(self):
        from vr_monitor import VRMonitor
        vr_monitor = VRMonitor()
        print("[VR Thread] Connecting...")

        async def loop_logic():
            monitor_task = asyncio.create_task(vr_monitor.start_monitoring())
            while self.running:
                goal = vr_monitor.get_right_goal_nowait()
                if goal is not None:
                    self.latest_goal.set(goal)
                await asyncio.sleep(0.002) 
            monitor_task.cancel()

        try:
            asyncio.run(loop_logic())
        except Exception as e:
            print(f"[VR Thread] Error: {e}")


def process_vr_quat(goal):
    """提取并统一 VR 四元数顺序为 [w, x, y, z]"""
    if not goal.metadata or "quaternion" not in goal.metadata:
        return np.array([1, 0, 0, 0])
        
    q_raw = goal.metadata["quaternion"]
    if isinstance(q_raw, dict):
        return np.array([q_raw['w'], q_raw['x'], q_raw['y'], q_raw['z']])
    else:
            # 假设列表是 [x, y, z, w]
        return np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])


# =========================
# Controller Class (Relative Logic)
# =========================
class VRPlannerController:
    def __init__(self, pos_scale=1.5, rot_scale=1.0):
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        
        # 上一帧的 VR Pose (Raw data)
        self.prev_vr_pos = None # np.array [x,y,z]
        self.prev_vr_rot = None # np.array [w,x,y,z]
        
        self.clutch_engaged = False # 记录上一帧是否按下了 squeeze

        # --- 坐标系转换矩阵 (VR Frame -> Robot Base Frame) ---
        # 假设：
        # VR: +X=右, +Y=上, -Z=前 (标准 OpenXR)
        # Robot: +X=前, +Y=左, +Z=上 (Panda Base)
        # 映射目标:
        # VR Right (+X) -> Robot Right (-Y)
        # VR Up    (+Y) -> Robot Up    (+Z)
        # VR Fwd   (-Z) -> Robot Fwd   (+X)
        self.coord_transform = np.array([
            [ 0,  0, -1],  # VR x -> Robot ? (Result X row) -> takes VR z (-1) -> +X
            [-1,  0,  0],  # VR y -> Robot ? (Result Y row) -> takes VR x (-1) -> -Y
            [ 0,  1,  0]   # VR z -> Robot ? (Result Z row) -> takes VR y (+1) -> +Z
        ])
        # 注意：上面的矩阵是将 VR 的 (x,y,z) 向量左乘，得到 Robot 的 (x,y,z)
        self.quat_transform_1 = np.array([0.707, -0.707, 0, 0])
        self.quat_transform_2 = np.array([0.707, 0, 0.707, 0])
        self.quat_transform_3 = np.array([0.707, 0, 0, -0.707])
        self.quat_transform_4 = np.array([0.707, 0, -0.707, 0])

    def step(self, latest_goal_obj:ControlGoal):
        """
        计算控制量
        Return: action (np.array 7 dim) or None
        """
        # 0. 初始化空动作 (保持不动)
        # pd_ee_delta_pose: 0 means hold pose
        action = np.zeros(7)
        # 默认 gripper open (1.0)
        action[6] = 1.0 

        if latest_goal_obj is None:
            return action

        # 1. 提取当前数据
        curr_pos = np.array(latest_goal_obj.target_position)
        curr_quat = process_vr_quat(latest_goal_obj)

        # 1.5 用 quat_transform_1 和 quat_transform_2 依次作用于 curr_quat，得到 bot_quat
        # bot_quat = tf_quat.qmult(self.quat_transform_2, tf_quat.qmult(self.quat_transform_1, curr_quat))
        # bot_quat = tf_quat.qmult(self.quat_transform_4, tf_quat.qmult(self.quat_transform_3, bot_quat))
        # bot_quat = bot_quat / np.linalg.norm(bot_quat)
        
        # 2. 检查离合器 (Squeeze)
        # 如果 squeeze 为 True，表示用户想控制；否则只更新 gripper
        if latest_goal_obj.metadata == None:
            is_squeezing = False
        else:
            is_squeezing = latest_goal_obj.metadata.get("squeeze", False) # 默认 False 安全

        # Gripper Logic (-1 closed, 1 open)
        if latest_goal_obj.gripper_closed:
            action[6] = -1.0
        
        # 3. 相对运动计算逻辑
        if is_squeezing:
            if not self.clutch_engaged:
                # 刚按下的一瞬间 (Rising Edge)
                # 重置上一帧数据为当前帧，防止跳变
                self.prev_vr_pos = curr_pos
                self.prev_vr_rot = curr_quat
                self.clutch_engaged = True
                # 本帧不移动
            else:
                # 持续按下中 (Holding) -> 计算 Delta
                
                # --- Position Delta ---
                # 1. 计算 VR 系下的位移
                d_pos_vr = curr_pos - self.prev_vr_pos
                
                # 2. 映射到 Robot Base 系
                d_pos_robot = self.coord_transform @ d_pos_vr
                
                # 3. 缩放
                action[:3] = d_pos_robot * self.pos_scale

                # --- Rotation Delta ---
                # 1. 计算四元数差分: Q_diff = Q_current_robot * Q_prev_robot^-1
                q_diff = tf_quat.qmult(curr_quat, tf_quat.qinverse(self.prev_vr_rot))
                
                # 2. 转为 Axis-Angle
                axis, angle = tf_quat.quat2axangle(q_diff)
                angle = -angle
                
                # 3. 映射旋转轴到 Robot Base 系
                # 旋转量是一个向量 (axis * angle)，同样需要经过坐标系变换
                axis_robot = self.coord_transform @ axis
                
                # 4. 组装并缩放
                # Normalize angle
                if angle > np.pi: angle -= 2*np.pi
                elif angle < -np.pi: angle += 2*np.pi
                
                action[3:6] = axis_robot * angle * self.rot_scale

                # --- Update History ---
                self.prev_vr_pos = curr_pos
                self.prev_vr_rot = curr_quat

        else:
            # 松开状态
            self.clutch_engaged = False
            self.prev_vr_pos = None
            self.prev_vr_rot = None
            # action[:6] 保持 0，即悬停

        return action

# =========================
# Main
# =========================
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "StackCube-v2"
    obs_mode: str = "rgb"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    record_dir: str = "hdf5-trajectory"
    viewer_shader: str = "rt-fast"
    # 灵敏度参数
    pos_sensitivity: float = 20.0 # Delta Pose 模式下，数值通常比较大，因为每一帧dt很小
    rot_sensitivity: float = 10.0

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Main] Creating environment: {args.env_id}")
    
    # Mode: pd_ee_delta_pose
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_delta_pose", 
        render_mode="rgb_array",
        reward_mode="none",
        robot_uids=args.robot_uid,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        source_type="teleoperation",
    )

    env.base_env.sim_config = SimConfig(sim_freq=200, control_freq=40)
    
    seed = 0
    env.reset(seed=seed)
    
    # 启动 VR
    latest_goal = ThreadSafeLatestGoal()
    vr_thread = VRInputThread(latest_goal)
    vr_thread.start()
    
    # 初始化控制器
    controller = VRPlannerController(
        pos_scale=args.pos_sensitivity, 
        rot_scale=args.rot_sensitivity
    )
    
    viewer = env.base_env.render_human()
    
    print("\n" + "="*50)
    print("XLeVR Relative Teleop Ready")
    print("Controls:")
    print("  [VR] Squeeze (Hold): Move Robot (Clutch)")
    print("  [VR] Trigger: Gripper")
    print("  [KB] S: Save | R: Retry | Q: Quit")
    print("="*50 + "\n")

    num_trajs = 0
    action_cmd = None
    last_mark_print = time.time()
    
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        while True:
            # 1. 获取 VR 数据
            goal = latest_goal.get()
            
            # 如果没有观测到 VR 目标，保持动作不变、跳过 mark 赋值
            if goal is None:
                action = np.zeros(7)
                action[6] = 1.0
            else:
                # 2. 控制器计算
                action = controller.step(goal)

                ################################################################
                # 调试：根据 coord_transform 映射 VR 位置、姿态到 mark
                coord_transform = np.array([
                    [ 0,  0, -1],  # VR x -> Robot ? (Result X row) -> takes VR z (-1) -> +X
                    [-1,  0,  0],  # VR y -> Robot ? (Result Y row) -> takes VR x (-1) -> -Y
                    [ 0,  1,  0]   # VR z -> Robot ? (Result Z row) -> takes VR y (+1) -> +Z
                ])
                curr_pos = np.array(goal.target_position)
                curr_quat = process_vr_quat(goal)

                # 根据映射关系 x-real->vr-x, z-real->vr-y, y-real-neg->vr-z，先构造 real->vr 旋转
                # 这等价于绕 x 轴 -90° 的旋转四元数
                real_to_vr_quat = np.array([0.70710678, -0.70710678, 0.0, 0.0])
                vr_quat = tf_quat.qmult(real_to_vr_quat, curr_quat)
                vr_quat = vr_quat / np.linalg.norm(vr_quat)

                env.base_env.mark.set_pose(Pose.create_from_pq(coord_transform @ curr_pos, curr_quat))
                #################################################################

            # 3. 环境步进
            env.step(action)
            
            env.base_env.render_human()

            # 每秒刷新 5 次 mark 四元数
            if time.time() - last_mark_print >= 1:
                try:
                    mstate = env.base_env.mark.get_state()
                    if hasattr(mstate, "ndim") and mstate.ndim >= 2:
                        mq = mstate[0, 3:7]
                    else:
                        mq = mstate[3:7]
                    if hasattr(mq, "cpu"):
                        mq = mq.cpu().numpy()
                    print(f"[mark quat] {mq}")
                except Exception as e:
                    print(f"[mark quat] read failed: {e}")
                last_mark_print = time.time()
            
            # 4. 键盘监听
            if viewer.window.key_press("q"):
                action_cmd = "quit"
                break
            elif viewer.window.key_press("s"):
                action_cmd = "save"
                break

        if action_cmd == "quit":
            num_trajs += 1
            break
        elif action_cmd == "save":
            num_trajs += 1
            seed += 1
            env.reset(seed=seed)             

    env.close()
    print(f"Saved {num_trajs} trajectories.")

if __name__ == "__main__":
    main(tyro.cli(Args))