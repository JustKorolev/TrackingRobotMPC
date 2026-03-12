# Self-Collision Avoidance -- UR10e

## Overview

The system prevents the UR10e from colliding with itself (or the ground) by
checking every candidate joint configuration **before** it reaches the robot.
Two independent layers run the same `collision_check()` function at two
different pipeline stages, providing defence in depth.

## Layers

### Layer 1 -- Joint position limits

Each of the 6 joint angles is compared against a per-joint absolute limit
(default +/- 350 deg, configurable via `JOINT_POS_LIMITS` in
`src/combined_main.py`).  Any joint exceeding its limit triggers a hard stop.

### Layer 2 -- FK link-to-link distance

Forward kinematics computes the 3D origin of every link frame (7 origins:
base + joints 1-6).  Each link is modelled as a line segment between
consecutive origins.  The minimum distance between all non-adjacent segment
pairs is computed.  If any pair is closer than `MIN_LINK_DISTANCE` (default
5 cm), a hard stop fires.

Non-adjacent pairs checked (10 total):

```
(0,2)  (0,3)  (0,4)  (0,5)
(1,3)  (1,4)  (1,5)
(2,4)  (2,5)
(3,5)
```

A ground-plane check (link origin z < 0) is also included.

## Check Points

### Check Point 1 -- `append_joint_target()` (trajectory queue entry)

Every joint-angle target produced by the IMU/IK pipeline is checked before
it enters `trajectory_queue`.  This includes every interpolated waypoint
inserted by `interpolate_joint_segment()`.

Location: `SharedTrajectoryState.append_joint_target()` in
`src/combined_main.py`.

### Check Point 2 -- `send_command()` (robot command gate)

Before sending `speedj` to the real robot, the predicted next joint state
(`current_joints + velocity * dt`) is checked.  This is the final safety
gate and catches anything the MPC solver might produce.

Location: `URXControlThread.send_command()` in `src/urx_control_thread.py`.

## Hard Stop Behaviour

When either check point detects a collision:

1. All joint velocities are zeroed (`u_curr = 0`)
2. `following_trajectory` is set to `False`
3. `robot_enabled` is set to `False`
4. `collision_detected` flag is set with a reason string
5. `[COLLISION] HARD STOP: <reason>` is printed to console
6. The GUI shows a red collision banner with the reason

The hard stop is **irreversible** until the user clicks "Clear Collision"
in the GUI.  After clearing, the user should re-home the robot before
resuming trajectory following.

## Configuration

All constants live in `src/combined_main.py`:

| Constant | Default | Description |
|---|---|---|
| `JOINT_POS_LIMITS` | 6.1087 rad (350 deg) per joint | Absolute joint angle limits |
| `MIN_LINK_DISTANCE` | 0.05 m (5 cm) | Minimum distance between non-adjacent link segments |
| `JOINT_VEL_LIMITS` | 1.0 rad/s per joint | Velocity limits (used by interpolation, not collision check) |

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `collision_check()` | `src/utils.py` | Core check: joint limits + FK origins + ground + segment distances |
| `segment_segment_distance()` | `src/utils.py` | Minimum distance between two 3D line segments |
| `hard_stop()` | `src/combined_main.py` | Sets collision flag, disables robot |
| `clear_collision()` | `src/combined_main.py` | Resets collision flag |

## Performance

The check runs ~7 DH transforms + 10 segment-distance calculations per
call.  This adds negligible latency at the 50-100 Hz control rates used
by the IMU and URX threads.
