"""
Combined script that runs both IMU GUI and MPC simulation simultaneously.
Run this instead of running main.py and imu.py separately.
"""

import threading
import sys
import time
import tkinter as tk
import numpy as np
from collections import deque

from src.imu import IMUGUI
from src.trajectory_tracking import MPCSimulationThread

try:
    from src.urx_control_thread import URXControlThread
    HAS_URX = True
except ImportError:
    HAS_URX = False

SAMPLING_RATE = 100 # Hz
MPC_HORIZON = SAMPLING_RATE // 20 # sec = horizon_samples / sampling_rate

# ── UR10e joint limits (CHANGE THESE for your actual robot) ──────────────
# Hardware max from datasheet:
#   Joints 0-1 (base, shoulder): 2.094 rad/s  (120 deg/s)
#   Joints 2-5 (elbow, wrists):  3.142 rad/s  (180 deg/s)
# Working limits (conservative for safety):
VJ = 0.5   # rad/s  -- uniform working velocity limit for MPC
AJ = 1.0   # rad/s² -- uniform working acceleration limit for MPC

# Per-joint working velocity limits (rad/s).
# UPDATE THESE when you know exact per-joint limits for your setup.
JOINT_VEL_LIMITS = np.array([
    VJ,   # joint 0 - base (shoulder pan)
    VJ,   # joint 1 - shoulder (lift)
    VJ,   # joint 2 - elbow
    VJ,   # joint 3 - wrist 1
    VJ,   # joint 4 - wrist 2
    VJ,   # joint 5 - wrist 3
])

# Cartesian velocity limits for pre-IK interpolation [x,y,z,roll,pitch,yaw].
# Position channels (m/s) match robot Cartesian speed.
# Orientation channels (rad/s) are generous -- real enforcement happens
# in joint space after IK using JOINT_VEL_LIMITS above.
CART_VEL_LIMITS = np.array([
    0.5,   # x  (m/s)
    0.5,   # y  (m/s)
    0.5,   # z  (m/s)
    5.0,   # roll  (rad/s)
    5.0,   # pitch (rad/s)
    5.0,   # yaw   (rad/s)
])


def interpolate_joint_segment(theta_prev, theta_next, dt, v_max):
    """Subdivide a joint-space segment so every joint respects its velocity limit.

    Parameters
    ----------
    theta_prev : (6,) current joint angles
    theta_next : (6,) target joint angles
    dt         : float, timestep between the two points
    v_max      : (6,) per-joint velocity limits (rad/s)

    Returns
    -------
    list of (6,) arrays -- intermediate waypoints (excluding theta_prev,
    including theta_next).  If the segment is already feasible the list
    contains only theta_next.
    """
    delta = theta_next - theta_prev
    if dt <= 0:
        return [theta_next]

    v_required = np.abs(delta) / dt
    ratios = v_required / v_max
    r = np.max(ratios)

    N = max(1, int(np.ceil(r)))

    if N == 1:
        return [theta_next]

    points = []
    for i in range(1, N + 1):
        points.append(theta_prev + (i / N) * delta)
    return points


class SharedTrajectoryState:
    """Thread-safe shared state for IMU and MPC communication."""

    def __init__(self):
        self.lock = threading.Lock()

        self.following_trajectory = False
        self.trajectory_window = deque(maxlen=MPC_HORIZON+1)

        self.u_curr = np.zeros((6,1))
        self.robot_enabled = False
        self.shutdown = False
        self.joint_pos = None

        self._last_joint_target = None

    def append_joint_target(self, theta_next, dt):
        """Append a joint-space target with automatic interpolation.

        If the step from the previous target to theta_next would violate
        any joint velocity limit, intermediate waypoints are inserted so
        the robot follows the same path at a feasible speed.
        """
        with self.lock:
            if self._last_joint_target is None:
                self._last_joint_target = theta_next.copy()
                self.trajectory_window.append(theta_next)
                return

            points = interpolate_joint_segment(
                self._last_joint_target, theta_next, dt, JOINT_VEL_LIMITS)

            if len(points) > 1:
                delta = theta_next - self._last_joint_target
                v_req = np.abs(delta) / dt
                worst = np.argmax(v_req / JOINT_VEL_LIMITS)
                print(f"[INTERP] Joint {worst} needs "
                      f"{v_req[worst]:.2f} rad/s (limit {JOINT_VEL_LIMITS[worst]:.2f}), "
                      f"N={len(points)}")

            for pt in points:
                self.trajectory_window.append(pt)

            self._last_joint_target = theta_next.copy()

    def start_following(self):
        with self.lock:
            self.trajectory_window = deque(maxlen=MPC_HORIZON+1)
            self.following_trajectory = True
            self.robot_enabled = True

    def stop_following(self):
        with self.lock:
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6,1))
            self._last_joint_target = None


def run_imu_gui(shared_state):
    """Run the IMU GUI in the main thread (required for Tkinter)."""
    root = tk.Tk()
    app = IMUGUI(root, shared_state, SAMPLING_RATE, MPC_HORIZON)
    root.mainloop()


def run_mpc_background(shared_state, mpc_horizon, status_callback=None):
    """
    Run MPC simulation in a background thread, toggling between active/idle based on
    shared_state.following_trajectory.
    """
    sim_thread = None
    last_status = None

    try:
        while True:
            with shared_state.lock:
                following = shared_state.following_trajectory
                traj_len = len(shared_state.trajectory_window)

            if following:
                if sim_thread is None:
                    if traj_len == mpc_horizon+1:
                        print("[MPC Thread] Trajectory following activated with a valid trajectory!")
                        msg = "MPC: Trajectory following activated, starting simulation..."
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg

                        sim_thread = MPCSimulationThread(
                            shared_state=shared_state,
                            mpc_horizon=mpc_horizon,
                            dt= 1 / SAMPLING_RATE
                        )
                        sim_thread.start()
                    else:
                        msg = "MPC: Waiting for full trajectory window..."
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        time.sleep(0.01)

                elif sim_thread.is_alive():
                    msg = f"MPC Status: {sim_thread.status}"
                    if status_callback and msg != last_status:
                        status_callback(msg)
                        last_status = msg
                    time.sleep(0.05)

                else:
                    if sim_thread.status == "completed":
                        msg = "MPC simulation completed!"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        print("[MPC Thread] Simulation completed!")
                    else:
                        msg = f"MPC Error: {sim_thread.status} - {sim_thread.error_msg}"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        print(f"[MPC Thread] Simulation crashed with status: {sim_thread.status}")

                        shared_state.stop_following()
                        print("[MPC Thread] Automatically disabled trajectory following due to crash")

                    sim_thread = None
                    time.sleep(0.01)

            else:
                if sim_thread is not None:
                    if sim_thread.is_alive():
                        print("[MPC Thread] Trajectory following disabled, stopping simulation...")
                        msg = "MPC: Paused (waiting for trajectory following)"
                        if status_callback and msg != last_status:
                            status_callback(msg)
                            last_status = msg
                        sim_thread.join(timeout=2)

                    sim_thread = None

                msg = "MPC: Idle (waiting for 'Start Following')"
                if status_callback and msg != last_status:
                    status_callback(msg)
                    last_status = msg

                time.sleep(0.01)

    except Exception as e:
        msg = f"MPC Error: {str(e)}"
        if status_callback and msg != last_status:
            status_callback(msg)
        print(f"[MPC Thread] Error: {e}")


def main():
    """Launch both IMU GUI and MPC simulation."""
    # Create shared state
    shared_state = SharedTrajectoryState()

    # Start MPC simulation in a daemon thread
    # It will wait until user clicks "Start Following" button
    mpc_thread = threading.Thread(target=run_mpc_background, args=(shared_state, MPC_HORIZON), daemon=True)
    mpc_thread.start()

    urx_thread = None
    if HAS_URX:
        urx_thread = URXControlThread(
            shared_state=shared_state,
            robot_ip="192.168.1.10",
            hz=100,
            vj=VJ,
            aj=AJ
        )
        urx_thread.start()
    else:
        print("[WARN] urx not installed -- running without robot arm")

    print("Starting combined IMU + MPC application...")
    print("MPC simulation is waiting for 'Start Following' button to be pressed in the GUI.")
    run_imu_gui(shared_state)

    with shared_state.lock:
        shared_state.shutdown = True

    if urx_thread is not None:
        urx_thread.stop()
        urx_thread.join(timeout=2)


if __name__ == "__main__":
    main()