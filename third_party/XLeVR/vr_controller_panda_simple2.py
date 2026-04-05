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
from dataclasses import dataclass
from typing import Annotated
import tyro
import gymnasium as gym
import numpy as np
import transforms3d.quaternions as tf_quat
from xlevr.inputs.vr_ws_server2 import ControlGoal
from pynput import keyboard
from mani_skill.utils.structs.types import SceneConfig
# =========================
# Path Setup
# =========================
XLEVR_PATH = "/home/firedust/mimic/maniskill_myws/third_party/XLeVR"
def setup_xlevr_environment():
    #if XLEVR_PATH not in sys.path:
    sys.path.insert(0, XLEVR_PATH)
    os.chdir(XLEVR_PATH)
    os.environ["PYTHONPATH"] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"

setup_xlevr_environment()

# from pathlib import Path

# XLEVR_PATH = Path(__file__).resolve().parent
# sys.path.insert(0, str(XLEVR_PATH))
# os.chdir(XLEVR_PATH)
# os.environ["PYTHONPATH"] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"

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
        #setup_xlevr_environment()
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


# =========================
# Keyboard Input Thread
# =========================
class KeyState:
    """线程安全的按键状态管理"""
    def __init__(self):
        self.lock = threading.Lock()
        self.save_pressed = False
        self.quit_pressed = False

    def set_save(self):
        with self.lock:
            self.save_pressed = True

    def set_quit(self):
        with self.lock:
            self.quit_pressed = True

    def consume_save(self):
        with self.lock:
            val = self.save_pressed
            self.save_pressed = False
            return val

    def consume_quit(self):
        with self.lock:
            val = self.quit_pressed
            self.quit_pressed = False
            return val


class KeyboardListener(threading.Thread):
    def __init__(self, key_state: KeyState):
        super().__init__(daemon=True)
        self.key_state = key_state

    def run(self):
        def on_press(key):
            try:
                if key.char == 's':
                    print("[Keyboard] Save pressed")
                    self.key_state.set_save()
                elif key.char == 'q':
                    print("[Keyboard] Quit pressed")
                    self.key_state.set_quit()
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


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
# Utility Functions
# =========================

def is_squeeze_pressed(goal):
    if goal is None:
        return False
    if goal.metadata is None:
        return False
    return bool(goal.metadata.get("squeeze", False))


def calculate_action(goal, prev_vr_pos, prev_vr_rot, clutch_engaged, coord_transform, pos_scale, rot_scale):
    """
    计算控制量
    Return: action, updated_prev_vr_pos, updated_prev_vr_rot, updated_clutch_engaged
    """
    action = np.zeros(7)
    action[6] = 1.0  # 默认 gripper open

    if goal is None:
        return action, None, None, False

    curr_pos = np.array(goal.target_position)
    curr_quat = process_vr_quat(goal)
    is_squeezing = is_squeeze_pressed(goal)

    if goal.gripper_closed:
        action[6] = -1.0

    if is_squeezing:
        if not clutch_engaged:
            prev_vr_pos = curr_pos
            prev_vr_rot = curr_quat
            clutch_engaged = True
        else:
            d_pos_vr = curr_pos - prev_vr_pos
            d_pos_robot = coord_transform @ d_pos_vr
            action[:3] = d_pos_robot * pos_scale

            q_diff = tf_quat.qmult(curr_quat, tf_quat.qinverse(prev_vr_rot))
            axis, angle = tf_quat.quat2axangle(q_diff)
            angle = -angle
            axis_robot = coord_transform @ axis

            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < -np.pi:
                angle += 2 * np.pi

            action[3:6] = axis_robot * angle * rot_scale
            prev_vr_pos = curr_pos
            prev_vr_rot = curr_quat
        
    else:
        clutch_engaged = False
        prev_vr_pos = None
        prev_vr_rot = None
        return action, prev_vr_pos, prev_vr_rot, clutch_engaged

    return action, prev_vr_pos, prev_vr_rot, clutch_engaged


# =========================
# Main
# =========================

list = ["OpenSafeDoor-v1", "OpenSafeDoor-v2", "StackCube-v2", "SolarPanelStatic-v1", "TakeSafetyHook-v1", "TurnGlobeValve-v2"]

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "OpenSafeDoor-v2"
    obs_mode: str = "rgb"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    record_dir: str = "demos2"
    viewer_shader: str = "rt-fast"
    number: int = 0
    # 灵敏度参数
    pos_sensitivity: float = 20.0 # Delta Pose 模式下，数值通常比较大，因为每一帧dt很小
    rot_sensitivity: float = 10.0


def create_environment(args: Args):
    import maniskill_myws.tasks  # noqa: F401

    output_dir = f"{args.record_dir}/{args.env_id}{args.number}/teleop/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Main] Creating environment: {args.env_id}")
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reward_mode="none",
        robot_uids=args.robot_uid,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        source_type="teleoperation",
    )

    return env


def start_vr_thread():
    latest_goal = ThreadSafeLatestGoal()
    vr_thread = VRInputThread(latest_goal)
    vr_thread.start()
    return latest_goal, vr_thread


def run_teleop_loop(env, latest_goal, key_state, pos_scale, rot_scale):
    coord_transform = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ])
    prev_vr_pos = None
    prev_vr_rot = None
    clutch_engaged = False
    prev_clutch_engaged = False

    env.base_env.render_human()

    print("\n" + "=" * 50)
    print("XLeVR Relative Teleop Ready")
    print("Controls:")
    print("  [VR] Squeeze : Enable Robot")
    print("  [VR] Trigger: Gripper")
    print("  [KB] S: Save | Q: Quit")
    print("=" * 50 + "\n")

    num_trajs = 0
    seed = 0
    action_cmd = None

    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        while True:
            #time0 = time.time()
            goal = latest_goal.get()

            #time1 = time.time()
            action, prev_vr_pos, prev_vr_rot, clutch_engaged = calculate_action(
                goal,
                prev_vr_pos,
                prev_vr_rot,
                clutch_engaged,
                coord_transform,
                pos_scale,
                rot_scale,
            )
            #time2 = time.time()

            is_squeezing = is_squeeze_pressed(goal)

            # if is_squeezing and not prev_clutch_engaged:
            #     print(f"Action calculation time: {(time2 - time1)*1000:.2f} ms")
            prev_clutch_engaged = clutch_engaged
            
            if is_squeezing:
                #time3 = time.time()
                env.step(action)
                #time4 = time.time()
                #print(f"Env step time: {(time4 - time3)*1000:.2f} ms")
                env.base_env.render_human()
                #time5 = time.time()
                #print(f"Render time: {(time5 - time0)*1000:.2f} ms")

            if key_state.consume_quit():
                action_cmd = "quit"
                break

            if key_state.consume_save():
                action_cmd = "save"
                break

        if action_cmd == "quit":
            num_trajs += 1
            break
        elif action_cmd == "save":
            num_trajs += 1
            seed += 1
            env.base_env.sim_config = SimConfig(sim_freq=200, control_freq=20,scene_config=SceneConfig(gravity=[0, 0, -0.00098]))
            env.reset(seed=seed,options={"reconfigure": True})
            env.base_env.render_human()
            
    env.close()
    print(f"Saved {num_trajs} trajectories.")


def main(args: Args):
    env = create_environment(args)
    env.base_env.sim_config = SimConfig(sim_freq=200, control_freq=20,scene_config=SceneConfig(gravity=[0, 0, -0.00098]))
    env.reset(seed=0,options={"reconfigure": True})
    
    latest_goal, vr_thread = start_vr_thread()

    key_state = KeyState()
    kb_thread = KeyboardListener(key_state)
    kb_thread.start()

    run_teleop_loop(env, latest_goal, key_state, args.pos_sensitivity, args.rot_sensitivity)


if __name__ == "__main__":
    main(tyro.cli(Args))