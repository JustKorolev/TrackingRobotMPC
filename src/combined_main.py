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
from src.urx_control_thread import URXControlThread

SAMPLING_RATE = 100 # Hz
MPC_HORIZON = SAMPLING_RATE // 20 # sec = horizon_samples / sampling_rate

class SharedTrajectoryState:
    """Thread-safe shared state for IMU and MPC communication."""

    def __init__(self):
        self.lock = threading.Lock()

        # Trajectory data
        self.following_trajectory = False
        self.trajectory_window = deque(maxlen=MPC_HORIZON+1)
        self.trajectory_window = np.ones((MPC_HORIZON+1, 6)) # TODO: REMOVE!!

        # Robot data
        self.u_curr = np.zeros((6,1))
        self.robot_enabled = False
        self.shutdown = False
        self.joint_pos = None

    def start_following(self):
        """Start trajectory following (called by MPC)."""
        with self.lock:
            self.following_trajectory = True
            self.robot_enabled = True

    def stop_following(self):
        """Stop trajectory following (called by MPC)."""
        with self.lock:
            self.following_trajectory = False
            self.robot_enabled = False
            self.u_curr = np.zeros((6,1))
            # self.trajectory_window = deque(maxlen=MPC_HORIZON*SAMPLING_RATE) #TODO:: UNCOMMENT THIS


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
                            dt=1 / SAMPLING_RATE
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

    urx_thread = URXControlThread(
        shared_state=shared_state,
        robot_ip="192.168.1.10",
        hz=100
    )
    urx_thread.start()

    # Run IMU GUI in main thread (this blocks until GUI closes)
    print("Starting combined IMU + MPC application...")
    print("MPC simulation is waiting for 'Start Following' button to be pressed in the GUI.")
    run_imu_gui(shared_state)

    # After GUI is killed
    with shared_state.lock:
        shared_state.shutdown = True

    urx_thread.stop()
    urx_thread.join(timeout=2)


if __name__ == "__main__":
    main()