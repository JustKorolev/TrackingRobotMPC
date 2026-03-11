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

SAMPLING_RATE = 150 # Hz
MPC_HORIZON = 1 # sec

class SharedTrajectoryState:
    """Thread-safe shared state for IMU and MPC communication."""

    def __init__(self):
        self.lock = threading.Lock()

        # Trajectory data
        self.following_trajectory = False
        self.trajectory_window = deque(maxlen=MPC_HORIZON*SAMPLING_RATE)

    def start_following(self):
        """Start trajectory following (called by MPC)."""
        with self.lock:
            self.following_trajectory = True

    def stop_following(self):
        """Stop trajectory following (called by MPC)."""
        with self.lock:
            self.following_trajectory = False


def run_imu_gui(shared_state):
    """Run the IMU GUI in the main thread (required for Tkinter)."""
    root = tk.Tk()
    app = IMUGUI(root, shared_state, SAMPLING_RATE, MPC_HORIZON)
    root.mainloop()


def run_mpc_background(shared_state, mpc_horizon, status_callback=None):
    """
    Run MPC simulation in a background thread only when trajectory following is active.
    Waits for shared_state.following_trajectory to be True before starting.
    """
    try:
        print("[MPC Thread] Waiting for trajectory following to be activated...")

        # Wait until trajectory following is enabled
        while not shared_state.following_trajectory:
            time.sleep(0.5)

        print("[MPC Thread] Trajectory following activated! Starting simulation...")
        if status_callback:
            status_callback("MPC: Trajectory following activated, starting simulation...")

        # Create and start MPC simulation thread
        sim_thread = MPCSimulationThread(shared_state=shared_state, mpc_horizon=mpc_horizon, dt=1/SAMPLING_RATE)
        sim_thread.start()

        # Monitor the thread
        while sim_thread.is_alive():
            if status_callback:
                status_callback(f"MPC Status: {sim_thread.status}")

            # If trajectory following stops, gracefully stop MPC
            if not shared_state.following_trajectory:
                print("[MPC Thread] Trajectory following disabled, stopping simulation...")
                break

            sim_thread.join(timeout=1)

        if sim_thread.status == "completed":
            if status_callback:
                status_callback("MPC simulation completed!")
            print("[MPC Thread] Simulation completed!")

            # Optionally visualize results
            try:
                env = sim_thread.results['env']
                env.visualize()
                env.visualize_error()
            except Exception as e:
                print(f"[MPC Thread] Could not visualize results: {e}")
        else:
            if status_callback:
                status_callback(f"MPC Status: {sim_thread.status} - {sim_thread.error_msg}")
            print(f"[MPC Thread] Simulation status: {sim_thread.status}")

    except Exception as e:
        if status_callback:
            status_callback(f"MPC Error: {str(e)}")
        print(f"[MPC Thread] Error: {e}")


def main():
    """Launch both IMU GUI and MPC simulation."""
    # Create shared state
    shared_state = SharedTrajectoryState()

    # Start MPC simulation in a daemon thread
    # It will wait until user clicks "Start Following" button
    mpc_thread = threading.Thread(target=run_mpc_background, args=(shared_state, MPC_HORIZON), daemon=True)
    mpc_thread.start()

    # Run IMU GUI in main thread (this blocks until GUI closes)
    print("Starting combined IMU + MPC application...")
    print("MPC simulation is waiting for 'Start Following' button to be pressed in the GUI.")
    run_imu_gui(shared_state)


if __name__ == "__main__":
    main()