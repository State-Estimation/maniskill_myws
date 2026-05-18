"""
VR Teleoperation for Panda (Absolute Joint Position Mode)
Mode: pd_joint_pos
Logic: VR absolute pose --[coord_transform]--> robot EE target --> Pink IK --> joint targets
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
import pinocchio as pin
import qpsolvers
import transforms3d.quaternions as tf_quat

import pink
from pink import Configuration
from pink.tasks import FrameTask

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
# Pink Planner (adapted for Panda)
# =========================

class PinkPlanner:
    """IK solver powered by pink for Panda arm.

    Uses pink's QP-based differential IK with ``FrameTask`` for end-effector
    pose tracking.  The QP optimizes joint displacement ``Δq`` directly;
    velocity ``v = Δq / dt`` is only post-processing.

    Parameters:
        model: pinocchio Model.
        data: pinocchio Data.
        frame_name: Name of the end-effector frame (e.g. ``"panda_hand_tcp"``).
        fixed_joint_indices: Joint indices to keep fixed at their initial
            values throughout IK (e.g. gripper fingers).
        dt: Integration timestep for each IK iteration.
        damping: Tikhonov regularization added to the QP Hessian.
        stop_thres: Convergence threshold on task error norm.
        max_steps: Maximum IK iterations per solve.
        pos_weight: Position error weight per axis ``[x, y, z]``.
        ori_weight: Orientation error weight per axis ``[rx, ry, rz]``.
        q_change_thres: Early-stop threshold on max joint change.
        q_change_patience: Consecutive small-change steps before early stop.
    """

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        frame_name: str,
        fixed_joint_indices: list[int] | None = None,
        dt: float = 1e-2,
        damping: float = 1e-8,
        stop_thres: float = 1e-6,
        max_steps: int = 50,
        pos_weight: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
        ori_weight: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
        q_change_thres: float = 1e-6,
        q_change_patience: int = 3,
    ):
        self.model = model
        self.data = data
        self.frame_name = frame_name
        self.fixed_joint_indices = fixed_joint_indices or []
        self.dt = dt
        self.damping = damping
        self.stop_thres = stop_thres
        self.max_steps = max_steps
        self.pos_weight = list(pos_weight)
        self.ori_weight = list(ori_weight)
        self.q_change_thres = q_change_thres
        self.q_change_patience = q_change_patience

        self.low = self.model.lowerPositionLimit
        self.high = self.model.upperPositionLimit
        self._fixed_q = None

        self._solver = (
            "daqp"
            if "daqp" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )

    def _make_frame_task(self, target_pose: pin.SE3) -> FrameTask:
        task = FrameTask(self.frame_name, self.pos_weight, self.ori_weight)
        task.set_target(target_pose)
        return task

    def _fix_joints(self, q: np.ndarray) -> np.ndarray:
        if self._fixed_q is not None:
            for idx in self.fixed_joint_indices:
                q[idx] = self._fixed_q[idx]
        return q

    def _ik_step(self, configuration: Configuration, task: FrameTask):
        """Single pink IK step: solve QP → integrate → clip → fix joints."""
        dv = pink.solve_ik(
            configuration,
            tasks=[task],
            dt=self.dt,
            damping=self.damping,
            solver=self._solver,
        )
        q = pin.integrate(self.model, configuration.q, dv * self.dt)
        q = np.clip(q, self.low, self.high)
        self._fix_joints(q)
        cfg = Configuration(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return cfg

    def solve_ik(self, q_init: np.ndarray, target_pose: pin.SE3):
        """Run IK to convergence, returning the final joint configuration.

        Returns ``(q_solution, n_steps, final_error)``.
        """
        q = q_init.copy()
        self._fixed_q = q_init.copy()
        self._fix_joints(q)
        task = self._make_frame_task(target_pose)
        cfg = Configuration(self.model, self.data, q)

        n_steps = 0
        small_change_count = 0
        error_norm = np.linalg.norm(task.compute_error(cfg))
        while error_norm > self.stop_thres and n_steps < self.max_steps:
            q_prev = cfg.q.copy()
            cfg = self._ik_step(cfg, task)
            error_norm = np.linalg.norm(task.compute_error(cfg))
            n_steps += 1
            if np.max(np.abs(cfg.q - q_prev)) < self.q_change_thres:
                small_change_count += 1
                if small_change_count >= self.q_change_patience:
                    break
            else:
                small_change_count = 0

        return cfg.q.copy(), n_steps, error_norm


def create_panda_pinocchio_model():
    """Load the Panda (panda_v3) URDF and return a pinocchio Model + Data pair."""
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam

    urdf_path = PandaWristCam.urdf_path
    with open(urdf_path, "r") as fh:
        urdf_str = fh.read()
    model = pin.buildModelFromXML(urdf_str)
    data = model.createData()
    return model, data


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
    """Compute target robot EE pose from VR displacement.

    1. Compute VR displacement (pos + orientation delta) from the VR reference
    2. Rotate pos delta and rotation axis via *coord_transform*
    3. Apply to the persistent sim-space reference (*sim_target_pos*, *sim_target_quat*)

    Returns ``(pin.SE3, target_pos, target_quat)``.
    """
    # --- position ---
    d_pos_vr = curr_vr_pos - vr_ref_pos
    d_pos_sim = coord_transform @ d_pos_vr * pos_scale
    target_pos = sim_target_pos + d_pos_sim

    # --- orientation ---
    q_diff = tf_quat.qmult(curr_vr_quat, tf_quat.qinverse(vr_ref_quat))
    axis, angle = tf_quat.quat2axangle(q_diff)
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

    # Build target SE3
    rot_target = pin.Quaternion(
        float(target_quat[0]), float(target_quat[1]),
        float(target_quat[2]), float(target_quat[3]),
    ).toRotationMatrix()

    return pin.SE3(rot_target, target_pos), target_pos, target_quat


# =========================
# State Machine
# =========================

class State(Enum):
    PAUSED_RELEASE = "paused_release"  # squeeze released, no target update
    TRACKING = "tracking"              # squeeze pressed, IK OK, stepping
    PAUSED_IK = "paused_ik"            # squeeze pressed, IK failed, target updates but no step


# =========================
# Main
# =========================

task_list = ["OpenSafeDoor-v1", "OpenSafeDoor-v2", "StackCube-v2",
             "SolarPanelStatic-v2", "TakeSafetyHook-v1", "TurnGlobeValve-v1"]

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "TurnGlobeValve-v1"
    obs_mode: str = "rgb"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam_custom_rot"
    record_dir: str = "demos2"
    viewer_shader: str = "rt-fast"
    number: int = 0
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    ik_error_threshold: float = 0.01  # IK error norm above which we pause
    auto_end_success_frames: int = 10  # consecutive success frames to auto-end trajectory (0 = disable)


def create_environment(args: Args):
    import maniskill_myws.tasks  # noqa: F401

    output_dir = f"{args.record_dir}/{args.env_id}{args.number}/teleop/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Main] Creating environment: {args.env_id}")
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
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
                    ik_error_threshold, auto_end_success_frames, base_seed):
    # ---- coordinate transform (VR → robot) ----
    coord_transform = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])

    # ---- pinocchio model & Pink planner ----
    pin_model, pin_data = create_panda_pinocchio_model()
    ee_frame_id = pin_model.getFrameId("panda_hand_tcp")
    q_low = pin_model.lowerPositionLimit
    q_high = pin_model.upperPositionLimit
    planner = PinkPlanner(
        pin_model, pin_data,
        frame_name="panda_hand_tcp",
        fixed_joint_indices=[7, 8],
        max_steps=50,
    )
    n_arm = 7

    # ---- robot root offset (pinocchio vs SAPIEN coordinate frames) ----
    # pinocchio FK places the robot at origin; SAPIEN may offset the root.
    # We compute the offset once and use it to transform targets for IK.
    root_offset_pos = np.zeros(3)

    # ---- state machine ----
    state = State.PAUSED_RELEASE

    # Persistent sim-space reference (updated on TRACKING→PAUSED_* and PAUSED_IK→PAUSED_RELEASE)
    sim_target_pos = None
    sim_target_quat = None

    # VR reference pose (set on PAUSED_RELEASE→TRACKING and TRACKING→PAUSED_IK)
    vr_ref_pos = None
    vr_ref_quat = None

    # Last valid achieved EE (updated every frame in TRACKING where IK succeeded)
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
    print("XLeVR Pink Teleop Ready  (pd_joint_pos + Pink IK)")
    print(f"  IK error threshold: {ik_error_threshold}")

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
    last_target_ee = None  # pin.SE3, for updating the viz frame

    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")

        success_counter = 0
        action_dim = env.unwrapped.single_action_space.shape[0]

        while True:
            goal = latest_goal.get()
            is_squeezing = is_squeeze_pressed(goal)

            # ---- get current robot state ----
            qpos_np = common.to_numpy(env.unwrapped.agent.robot.get_qpos())
            if qpos_np.ndim == 2:
                qpos_np = qpos_np[0]
            # Clip to pinocchio joint limits (SAPIEN may have fp rounding above limits)
            qpos_np = np.clip(qpos_np, q_low, q_high)

            # ---- default action: hold current arm pos + open gripper ----
            action = np.zeros(action_dim)       # (8,) = 7 arm + 1 gripper (mimic)
            action[:n_arm] = qpos_np[:n_arm]    # arm joints from current sim state

            if goal is not None:
                curr_vr_pos = np.array(goal.target_position)
                curr_vr_quat = process_vr_quat(goal)

                # --- gripper (always active when goal available) ---
                action[n_arm] = gripper_close if goal.gripper_closed else gripper_open

                # ====================================================
                #  State machine
                # ====================================================

                if state == State.PAUSED_RELEASE:
                    # Squeeze not pressed, sim_target frozen.
                    if is_squeezing:
                        # ---- RELEASE → TRACKING ----
                        if sim_target_pos is None:
                            # Use pinocchio FK (not SAPIEN) so IK target is in the same
                            # coordinate frame the planner works in (robot root at origin).
                            pin.forwardKinematics(pin_model, pin_data, qpos_np)
                            pin.updateFramePlacements(pin_model, pin_data)
                            ee_pose = pin_data.oMf[ee_frame_id]
                            sim_target_pos = ee_pose.translation.copy()
                            r = ee_pose.rotation
                            q = pin.Quaternion(r)
                            sim_target_quat = np.array([q.w, q.x, q.y, q.z])
                            # Compute SAPIEN→pinocchio root offset for viz
                            sapien_tcp = common.to_numpy(
                                env.unwrapped.agent.tcp_pose.p
                            ).reshape(-1)
                            root_offset_pos = sapien_tcp - sim_target_pos
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
                        # Compute target & solve IK
                        last_target_ee, _, _ = compute_target_ee_pose(
                            curr_vr_pos, curr_vr_quat,
                            vr_ref_pos, vr_ref_quat,
                            sim_target_pos, sim_target_quat,
                            coord_transform, pos_scale, rot_scale,
                        )
                        q_sol, _n_steps, ik_err = planner.solve_ik(qpos_np, last_target_ee)

                        if ik_err > ik_error_threshold:
                            # ---- TRACKING → PAUSED_IK ----
                            if last_valid_pos is not None:
                                sim_target_pos = last_valid_pos
                                sim_target_quat = last_valid_quat
                            vr_ref_pos = curr_vr_pos.copy()
                            vr_ref_quat = curr_vr_quat.copy()
                            state = State.PAUSED_IK
                        else:
                            # IK OK — step the environment
                            action[:n_arm] = q_sol[:n_arm]
                            env.step(action)
                            env.base_env.render_human()

                            # Update last valid achieved EE
                            pin.forwardKinematics(pin_model, pin_data, q_sol)
                            pin.updateFramePlacements(pin_model, pin_data)
                            achieved_ee = pin_data.oMf[ee_frame_id]
                            last_valid_pos = achieved_ee.translation.copy()
                            r = achieved_ee.rotation
                            q = pin.Quaternion(r)
                            last_valid_quat = np.array([q.w, q.x, q.y, q.z])

                elif state == State.PAUSED_IK:
                    if not is_squeezing:
                        # ---- PAUSED_IK → PAUSED_RELEASE ----
                        sim_target_pos = last_valid_pos
                        sim_target_quat = last_valid_quat
                        vr_ref_pos = None
                        vr_ref_quat = None
                        state = State.PAUSED_RELEASE
                    else:
                        # Keep updating target_ee (VR is still moving)
                        # but do NOT step until IK recovers
                        last_target_ee, _, _ = compute_target_ee_pose(
                            curr_vr_pos, curr_vr_quat,
                            vr_ref_pos, vr_ref_quat,
                            sim_target_pos, sim_target_quat,
                            coord_transform, pos_scale, rot_scale,
                        )
                        q_sol, _n_steps, ik_err = planner.solve_ik(qpos_np, last_target_ee)

                        if ik_err <= ik_error_threshold:
                            # ---- PAUSED_IK → TRACKING (auto-recover) ----
                            action[:n_arm] = q_sol[:n_arm]
                            env.step(action)
                            env.base_env.render_human()

                            pin.forwardKinematics(pin_model, pin_data, q_sol)
                            pin.updateFramePlacements(pin_model, pin_data)
                            achieved_ee = pin_data.oMf[ee_frame_id]
                            last_valid_pos = achieved_ee.translation.copy()
                            r = achieved_ee.rotation
                            q = pin.Quaternion(r)
                            last_valid_quat = np.array([q.w, q.x, q.y, q.z])
                            state = State.TRACKING
                        else:
                            # Still failing — update last_valid from FK
                            # (the IK returned the nearest feasible pose)
                            pin.forwardKinematics(pin_model, pin_data, q_sol)
                            pin.updateFramePlacements(pin_model, pin_data)
                            achieved_ee = pin_data.oMf[ee_frame_id]
                            last_valid_pos = achieved_ee.translation.copy()
                            r = achieved_ee.rotation
                            q = pin.Quaternion(r)
                            last_valid_quat = np.array([q.w, q.x, q.y, q.z])

            # ---- update target-pose coordinate frame in viewer ----
            if target_frame_node is not None and last_target_ee is not None:
                # target_ee is in pinocchio frame; add root offset for SAPIEN viewer
                t = last_target_ee.translation + root_offset_pos
                r = last_target_ee.rotation
                q = pin.Quaternion(r)
                target_frame_node.set_position(t)
                target_frame_node.set_rotation([q.w, q.x, q.y, q.z])

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
            last_target_ee = None
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
                    args.ik_error_threshold, args.auto_end_success_frames, base_seed)


if __name__ == "__main__":
    main(tyro.cli(Args))