"""
VR Teleoperation for Panda (Absolute EE Pose Mode)
Mode: pd_ee_pose (use_delta=False)
Logic: VR absolute pose --[coord_transform]--> robot EE target (position + Euler XYZ)
"""

import os
import sys
import threading
import time
import numpy as np
import torch
import gymnasium as gym
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Annotated
import tyro
import transforms3d.quaternions as tf_quat
import transforms3d.euler as tf_euler

from xlevr.inputs.vr_ws_server2 import ControlGoal
from pynput import keyboard

# =========================
# Path Setup
# =========================
XLEVR_PATH = "/home/firedust/mimic/maniskill_myws/third_party/XLeVR"
def setup_xlevr_environment():
    sys.path.insert(0, XLEVR_PATH)
    os.chdir(XLEVR_PATH)
    os.environ["PYTHONPATH"] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"

setup_xlevr_environment()

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs import SimConfig
import maniskill_myws.tasks
from mani_skill.utils import common


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


# =========================
# Keyboard Input Thread
# =========================
class KeyState:
    """Thread-safe key state management."""
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


# =========================
# Utility Functions
# =========================

def process_vr_quat(goal):
    """Extract VR quaternion in [w, x, y, z] order."""
    if not goal.metadata or "quaternion" not in goal.metadata:
        return np.array([1, 0, 0, 0])

    q_raw = goal.metadata["quaternion"]
    if isinstance(q_raw, dict):
        return np.array([q_raw['w'], q_raw['x'], q_raw['y'], q_raw['z']])
    else:
        # Assume list is [x, y, z, w]
        return np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])


def is_squeeze_pressed(goal):
    if goal is None:
        return False
    if goal.metadata is None:
        return False
    return bool(goal.metadata.get("squeeze", False))


def compute_target_ee_pose(curr_vr_pos, curr_vr_quat, vr_ref_pos, vr_ref_quat,
                           sim_target_pos, sim_target_quat, coord_transform,
                           pos_scale, rot_scale):
    """Compute absolute target robot EE pose from VR displacement.

    1. Compute VR displacement (pos + orientation delta) from the VR reference
    2. Rotate pos delta and rotation axis via *coord_transform*
    3. Apply to the persistent sim-space reference (*sim_target_pos*, *sim_target_quat*)

    Returns ``(target_pos, target_quat)`` in the same frame as the sim reference.
    """
    # --- position ---
    d_pos_vr = curr_vr_pos - vr_ref_pos
    d_pos_sim = coord_transform @ d_pos_vr * pos_scale
    target_pos = sim_target_pos + d_pos_sim

    # --- orientation ---
    q_diff = tf_quat.qmult(curr_vr_quat, tf_quat.qinverse(vr_ref_quat))
    axis, angle = tf_quat.quat2axangle(q_diff)
    #angle = -angle
    axis_sim = coord_transform @ axis

    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi

    angle_scaled = angle * rot_scale

    if np.abs(angle_scaled) > 1e-12:
        q_delta_sim = tf_quat.axangle2quat(axis_sim, angle_scaled)
    else:
        q_delta_sim = np.array([1.0, 0.0, 0.0, 0.0])

    target_quat = tf_quat.qmult(q_delta_sim, sim_target_quat)

    return target_pos, target_quat


# =========================
# State Machine
# =========================

class State(Enum):
    PAUSED_RELEASE = "paused_release"  # squeeze released, no target update
    TRACKING = "tracking"              # squeeze pressed, stepping with absolute EE pose


# =========================
# Main
# =========================

task_list = ["OpenSafeDoor-v1", "OpenSafeDoor-v2", "StackCube-v2",
             "SolarPanelStatic-v2", "TakeSafetyHook-v1", "TurnGlobeValve-v1"]

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "SolarPanelStatic-v2"
    obs_mode: str = "rgb"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    record_dir: str = "demos2"
    viewer_shader: str = "rt-fast"
    number: int = 0
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    auto_end_success_frames: int = 10  # consecutive success frames to auto-end trajectory (0 = disable)


def create_environment(args: Args):
    import maniskill_myws.tasks  # noqa: F401

    output_dir = f"{args.record_dir}/{args.env_id}{args.number}/teleop/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Main] Creating environment: {args.env_id}")
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_pose",
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


def run_teleop_loop(env, latest_goal, key_state, pos_scale, rot_scale,
                    auto_end_success_frames, base_seed):
    # ---- coordinate transform (VR → robot) ----
    coord_transform = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ])

    # ---- robot root offset (world → base frame) ----
    # The pd_ee_pose controller expects EE pose in robot base frame.
    # tcp_pose is in world frame, so we compute the offset once.
    robot = env.unwrapped.agent.robot
    root_pose_world = robot.get_root_pose()
    root_offset_pos = common.to_numpy(root_pose_world.p).reshape(-1)
    root_q = common.to_numpy(root_pose_world.q).reshape(-1)
    root_offset_quat = np.array([root_q[0], root_q[1], root_q[2], root_q[3]])  # [w, x, y, z]

    def world_to_base(world_pos, world_quat):
        """Convert EE pose from world frame to robot base frame."""
        base_pos = world_pos - root_offset_pos
        base_quat = tf_quat.qmult(tf_quat.qinverse(root_offset_quat), world_quat)
        return base_pos, base_quat

    def base_to_world(base_pos, base_quat):
        """Convert EE pose from robot base frame to world frame (for viz)."""
        world_pos = base_pos + root_offset_pos
        world_quat = tf_quat.qmult(root_offset_quat, base_quat)
        return world_pos, world_quat

    # ---- state machine ----
    state = State.PAUSED_RELEASE

    # Persistent sim-space reference (absolute EE pose in robot BASE frame)
    sim_target_pos = None
    sim_target_quat = None

    # VR reference pose (set on PAUSED_RELEASE→TRACKING)
    vr_ref_pos = None
    vr_ref_quat = None

    # Last achieved EE in BASE frame (updated every frame in TRACKING)
    last_valid_pos = None
    last_valid_quat = None

    # ---- gripper targets (normalized: action space is [-1, 1] for mimic joint) ----
    gripper_open = 1.0
    gripper_close = -1.0

    env.base_env.render_human()

    # ---- target-pose visualization (coordinate axes in viewer) ----
    def _create_viewer_frames(_env):
        """Create (or recreate after reset) coordinate-axis nodes in the viewer."""
        v = _env.unwrapped.viewer
        if v is None:
            return None, None
        import sapien
        tf = v.add_coordinate_frame(
            sapien.Pose([0, 0, 0], [1, 0, 0, 0]), length=0.08,
        )
        ef = v.add_coordinate_frame(
            sapien.Pose([0, 0, 0], [1, 0, 0, 0]), length=0.06,
        )
        return tf, ef

    target_frame_node, ee_frame_node = _create_viewer_frames(env)

    print("\n" + "=" * 50)
    print("XLeVR Teleop Ready  (pd_ee_pose, absolute mode)")

    print("Controls:")
    print("  [VR] Squeeze : Enable Tracking")
    print("  [VR] Trigger : Gripper")
    print("  [KB] S: Save | Q: Quit")
    if auto_end_success_frames > 0:
        print(f"  [Auto] End trajectory after {auto_end_success_frames} consecutive success frames")
    print("=" * 50 + "\n")

    num_trajs = 0
    seed = base_seed
    action_cmd = None
    success_counter = 0
    last_idle_render = 0.0
    last_target_pos = None  # for viz
    last_target_quat = None  # [w, x, y, z], for viz

    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")

        success_counter = 0
        action_dim = env.unwrapped.single_action_space.shape[0]

        while True:
            goal = latest_goal.get()
            is_squeezing = is_squeeze_pressed(goal)

            if goal is not None:
                curr_vr_pos = np.array(goal.target_position)
                curr_vr_quat = process_vr_quat(goal)

                # ---- default action: absolute EE pose = current EE + gripper ----
                action = np.zeros(action_dim)
                action[-1] = gripper_close if goal.gripper_closed else gripper_open

                # ====================================================
                #  State machine
                # ====================================================

                if state == State.PAUSED_RELEASE:
                    if is_squeezing:
                        # ---- RELEASE → TRACKING ----
                        if sim_target_pos is None:
                            ee_pose = env.unwrapped.agent.tcp_pose
                            ee_p_world = common.to_numpy(ee_pose.p).reshape(-1)
                            ee_q_world = common.to_numpy(ee_pose.q).reshape(-1)
                            sim_target_pos, sim_target_quat = world_to_base(ee_p_world, ee_q_world)
                        vr_ref_pos = curr_vr_pos.copy()
                        vr_ref_quat = curr_vr_quat.copy()
                        state = State.TRACKING
                        # fall through to TRACKING logic below

                if state == State.TRACKING:
                    if not is_squeezing:
                        # ---- TRACKING → PAUSED_RELEASE ----
                        sim_target_pos = last_valid_pos
                        sim_target_quat = last_valid_quat
                        state = State.PAUSED_RELEASE
                    else:
                        # Compute absolute target EE pose from VR displacement
                        last_target_pos, last_target_quat = compute_target_ee_pose(
                            curr_vr_pos, curr_vr_quat,
                            vr_ref_pos, vr_ref_quat,
                            sim_target_pos, sim_target_quat,
                            coord_transform, pos_scale, rot_scale,
                        )

                        # Convert target quaternion to Euler XYZ for the action
                        target_euler = tf_euler.quat2euler(last_target_quat, axes='rxyz')
                        action[:3] = last_target_pos
                        action[3:6] = target_euler

                        env.step(action)
                        env.base_env.render_human()

                        # Update last achieved EE (in base frame)
                        ee_pose = env.unwrapped.agent.tcp_pose
                        ee_p_world = common.to_numpy(ee_pose.p).reshape(-1)
                        ee_q_world = common.to_numpy(ee_pose.q).reshape(-1)
                        last_valid_pos, last_valid_quat = world_to_base(ee_p_world, ee_q_world)

            # ---- update target-pose coordinate frame in viewer ----
            if target_frame_node is not None and last_target_pos is not None:
                viz_pos, viz_quat = base_to_world(last_target_pos, last_target_quat)
                target_frame_node.set_position(viz_pos)
                target_frame_node.set_rotation(viz_quat)

            # ---- update actual-EE coordinate frame (direct from sim) ----
            if ee_frame_node is not None:
                ee_pose = env.unwrapped.agent.tcp_pose
                ee_p = common.to_numpy(ee_pose.p).reshape(-1)
                ee_q = common.to_numpy(ee_pose.q).reshape(-1)
                ee_frame_node.set_position(ee_p)
                ee_frame_node.set_rotation(ee_q)  # [w, x, y, z]

            # ---- idle render: keep GUI alive in paused states ----
            if state != State.TRACKING:
                now = time.time()
                if now - last_idle_render > 0.033:  # ~30 FPS
                    env.base_env.render_human()
                    last_idle_render = now

            # ---- auto-end on consecutive success frames ----
            if auto_end_success_frames > 0:
                eval_result = env.base_env.evaluate()
                if eval_result.get("success", False):
                    if isinstance(eval_result["success"], torch.Tensor):
                        is_success = bool(eval_result["success"].item())
                    else:
                        is_success = bool(eval_result["success"])
                    if is_success:
                        success_counter += 1
                    else:
                        success_counter = 0
                else:
                    success_counter = 0
                if success_counter >= auto_end_success_frames:
                    print(f"[Auto-End] {auto_end_success_frames} consecutive success frames, saving trajectory")
                    action_cmd = "save"
                    break

            # ---- keyboard commands ----
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
            env.reset(seed=seed, options={"reconfigure": True})
            env.base_env.render_human()
            # Viewer frames are destroyed on reset; re-create them
            target_frame_node, ee_frame_node = _create_viewer_frames(env)
            # Reset persistent state so sim_target is re-initialized from new EE pose
            sim_target_pos = None
            sim_target_quat = None
            last_valid_pos = None
            last_valid_quat = None
            last_target_pos = None
            last_target_quat = None
            state = State.PAUSED_RELEASE

    env.close()
    print(f"Saved {num_trajs} trajectories.")


def main(args: Args):
    env = create_environment(args)
    base_seed = args.number * 100
    env.reset(seed=base_seed, options={"reconfigure": True})

    latest_goal, vr_thread = start_vr_thread()

    key_state = KeyState()
    kb_thread = KeyboardListener(key_state)
    kb_thread.start()

    run_teleop_loop(env, latest_goal, key_state,
                    args.pos_scale, args.rot_scale,
                    args.auto_end_success_frames, base_seed)


if __name__ == "__main__":
    main(tyro.cli(Args))
