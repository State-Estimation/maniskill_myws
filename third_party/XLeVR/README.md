# Pink-based VR Teleoperation for Panda

## Architecture Overview

```
VR Controller (absolute pose)
    │
    ├─ coord_transform (3×3 rotation): VR axes → robot axes
    │       [ 0  0 -1]    VR x → robot -z
    │       [-1  0  0]    VR y → robot -x
    │       [ 0  1  0]    VR z → robot +y
    │
    ├─ Per-frame: delta = coord @ (vr_current - vr_ref)
    │
    ├─ target_ee = sim_target + delta    (in pinocchio frame)
    │
    ├─ PinkPlanner.solve_ik(qpos, target_ee) → joint targets
    │
    └─ env.step(action)   [pd_joint_pos, 8-dim: 7 arm + 1 gripper]
```

### Coordinate Frames

Two coordinate systems coexist:

| Frame | Origin | Used for |
|---|---|---|
| **pinocchio** | robot root at `[0,0,0]` | All IK computation, `sim_target`, `last_valid` |
| **SAPIEN (viewer)** | scene origin (robot may be offset) | `tcp_pose` reading, coordinate-axis visualization |

A constant `root_offset_pos = sapien_tcp - pin_fk` is computed once at init. Target/EE frames are displayed in SAPIEN frame by adding this offset to pinocchio-frame positions.

---

## PinkPlanner (QP-based Differential IK)

### How it works

Each IK call runs an iterative loop:

1. **QP step**: `pink.solve_ik(config, tasks, dt, damping)` solves for joint velocity `dv`
2. **Integrate**: `q_new = integrate(q_prev, dv * dt)`
3. **Clip**: `q_new = clamp(q_new, joint_limits)`
4. **Fix joints**: restore gripper joint values (`fixed_joint_indices=[7, 8]`)
5. Repeat until `error < stop_thres` or `n_steps >= max_steps`

Velocity limits are enforced as QP inequality constraints: `|Δq| ≤ dt * v_max`.

### Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `dt` | 1e-2 | Integration timestep per iteration; also bounds max displacement per step |
| `damping` | 1e-8 | Tikhonov regularization on QP Hessian |
| `stop_thres` | 1e-6 | Convergence threshold on FrameTask error norm |
| `max_steps` | 100 (for VR) | Max iterations; lower = faster per-frame but rougher convergence |

### Why iterative IK with `pd_joint_pos` works for real-time VR

Each frame starts IK from the **current** joint state. With VR running at ~100Hz, consecutive targets are close together, so `n_steps` is typically < 20. The PD controller's stiffness/damping provides natural low-pass smoothing, and the QP velocity constraints prevent physically impossible jumps.

---

## State Machine

Three states, designed to decouple IK failure handling from user intent:

![State Machine](assets/State%20Machine.png)



### State Behaviors

| State | target_ee updated? | env.step()? | sim_target |
|---|---|---|---|
| **PAUSED_RELEASE** | No | No | Frozen (saved on release) |
| **TRACKING** | Yes | Yes | Fixed during squeeze session |
| **PAUSED_IK** | Yes (VR keeps moving) | No | Frozen (saved on IK fail) |

### Why two different PAUSE states?

- **PAUSED_RELEASE** (squeeze released): The user intentionally stopped. Target is frozen.
- **PAUSED_IK** (IK failure): The user is still moving VR, but the target left the workspace. Target **continues updating** so the user can move VR back and auto-recover without re-clutching.

### Transitions in detail

**PAUSED_RELEASE → TRACKING** (squeeze pressed):
1. If first-ever press: init `sim_target` from pinocchio FK + `bias_pos`
2. Record `vr_ref` = current VR absolute pose (fixed for this session)
3. Fall through to TRACKING logic

**TRACKING → PAUSED_IK** (IK error > threshold):
1. Save `sim_target = last_valid` (the FK result of the last successful IK)
2. Record `vr_ref` = current VR pose (the pose that caused the failure)
3. Target continues updating from this new reference

**PAUSED_IK → TRACKING** (IK error ≤ threshold, auto-recover):
- No state variable changes needed; just resume stepping. The target formula `sim_target + coord @ (vr - vr_ref)` remains valid.

**TRACKING/PAUSED_IK → PAUSED_RELEASE** (squeeze released):
1. Save `sim_target = last_valid` (preserve progress in SAPIEN frame)
2. Next press continues from this position

---

## Workspace Boundary Handling

When the user pushes VR past the robot's reachable workspace:

1. `target_ee` exceeds joint limits → IK returns the **nearest feasible** joint config (at the boundary)
2. FK of the IK result → `last_valid` = boundary pose (always reachable)
3. On release, `sim_target = last_valid` (boundary, not the unreachable target)
4. Next press starts from the boundary — no jump, no stuck target

Additionally, if IK error exceeds `ik_error_threshold` (default 0.05), the state machine pauses stepping (→ PAUSED_IK) to prevent the robot from driving into joint limits at high speed. The user can move VR back toward the workspace to auto-resume.

---

## Action Space (`pd_joint_pos`)

Shape `(8,)` = 7 arm joints + 1 gripper (mimic).

| Index | Content | Space | Range |
|---|---|---|---|
| 0-6 | Arm joint targets | Absolute radians | Joint limits from URDF (e.g. ±2.8973) |
| 7 | Gripper target | **Normalized** [-1, 1] | -1 → -0.01m, +1 → 0.04m |

### Gripper normalization

The gripper uses `PDJointPosMimicController` with `normalize_action=True`:

```
actual (meters) = lower + (action + 1) / 2 * (upper - lower)
                = -0.01 + (action + 1) / 2 * 0.05
```

| Action value | Actual position | Meaning |
|---|---|---|
| `1.0` | 0.04 m | Fully open |
| `-0.6` | 0.00 m | Closed |
| `-1.0` | -0.01 m | Slightly negative (squeeze thin objects) |

The mimic joint (`panda_finger_joint2`) automatically follows `panda_finger_joint1`, so only one action value is needed.

---

## Persistence Across Squeeze Sessions

The key insight: `sim_target` persists across squeeze-release cycles, allowing the user to:

1. Press squeeze → move robot
2. Release squeeze → robot freezes, `sim_target` saved at last achieved pose
3. Reposition VR controller comfortably
4. Press squeeze again → `vr_ref` reset, tracking continues from `sim_target` (no jump)

The formula `target = sim_target + coord @ (vr - vr_ref)` ensures:
- `sim_target` is **fixed** during a squeeze session
- `vr_ref` is **fixed** at press time (or at IK-failure time)
- Frame-to-frame movement is purely incremental: `Δtarget = coord @ (vr_t - vr_{t-1})`

---

## Visualization

Two coordinate-axis frames in the SAPIEN viewer (RGB = XYZ):

| Frame | Length | Meaning |
|---|---|---|
| Long axes (8 cm) | VR-mapped **target** pose | Where the user wants to go |
| Short axes (6 cm) | Robot **actual** EE pose (via `agent.tcp_pose`) | Where the robot actually is |

During normal tracking, the two frames converge. Divergence indicates either tracking lag or workspace boundary constraint.

---

## Key Files

| File | Purpose |
|---|---|
| `vr_controller_panda_pink.py` | Main teleop script: PinkPlanner, state machine, VR loop |
| `vr_controller_panda_simple2.py` | Legacy delta-pose version (`pd_ee_delta_pose`) |
| `tools_in_maniskill/joint_state_access.md` | Reference: accessing robot state in ManiSkill |
| `tools_in_maniskill/step_action_analysis.md` | Reference: action dispatch logic in ManiSkill |

## Usage

```bash
python vr_controller_panda_pink.py \
    -e TurnGlobeValve-v1 \
    -r panda_wristcam \
    --pos-scale 1.0 \
    --rot-scale 1.0 \
    --ik-error-threshold 0.05 \
    --bias-pos 0.0 0.0 0.0
```

### Key CLI Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--pos-scale` | 1.0 | Multiplier on VR position displacement |
| `--rot-scale` | 1.0 | Multiplier on VR rotation displacement |
| `--ik-error-threshold` | 0.05 | IK error above which tracking pauses |
| `--bias-pos` | 0 0 0 | `[x, y, z]` offset added to initial sim_target (calibration) |

### Controls

| Input | Action |
|---|---|
| VR Squeeze | Enable tracking (hold to move robot) |
| VR Trigger | Close gripper |
| Keyboard `S` | Save trajectory + reset env |
| Keyboard `Q` | Quit |
