import threading
import time
import os
import glob
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class IMUGUI:
    def __init__(self, root, shared_state=None):
        self.root = root
        self.root.title("IMU Trajectory Recorder")
        self.shared_state = shared_state

        self.G = 9.80665
        self.ALPHA = 0.98
        self.ACCEL_DEADBAND = 0.2
        self.VEL_DEADBAND = 0.01
        self.GYRO_STATIONARY_THRESH = 0.05
        self.ACCEL_STATIONARY_THRESH = 0.4
        self.VELOCITY_DECAY = 0.998
        self.ZUPT_VELOCITY_DECAY = 0.8
        self.MAX_VELOCITY = 2.0
        self.MAX_RAW_SAMPLES = 500

        self.MAX_LINEAR_VELOCITY = 0.5
        self.MAX_LINEAR_ACCELERATION = 2.0
        self.MAX_ANGULAR_VELOCITY = 1.0
        self.MAX_ANGULAR_ACCELERATION = 5.0
        self.TRAJECTORY_SMOOTHING = True

        self.ser = None
        self.header_seen = False
        self.running = True

        self.state_lock = threading.Lock()

        self.mode = "idle"
        self.status_text = "Disconnected"

        self.request_calibrate = False
        self.request_record = False
        self.request_replay = False
        self.request_stop = False

        self.trajectory_ready = False
        self.replay_enabled = False
        self.replay_start_wall = 0.0

        self.calibration_time = 7.0
        self.record_time = 10.0

        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.gravity_direction = np.array([0.0, 0.0, self.G])

        self.recorded_positions = np.zeros((1, 3))
        self.recorded_times = np.zeros(1)
        self.recorded_orientations = np.zeros((1, 3))

        self.trajectories_dir = r"..\trajectories"
        self.current_trajectory_name = ""

        self.raw_t = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_ax = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_ay = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_az = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_gx = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_gy = deque(maxlen=self.MAX_RAW_SAMPLES)
        self.raw_gz = deque(maxlen=self.MAX_RAW_SAMPLES)

        self.build_gui()
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.ani = FuncAnimation(self.fig, self.update_plots, interval=30, cache_frame_data=False)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_gui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="COM Port").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value=self.auto_detect_port())
        self.port_entry = ttk.Entry(top, textvariable=self.port_var, width=12)
        self.port_entry.grid(row=0, column=1, padx=5)

        self.refresh_btn = ttk.Button(top, text="Refresh Ports", command=self.refresh_ports)
        self.refresh_btn.grid(row=0, column=2, padx=5)

        self.connect_btn = ttk.Button(top, text="Connect", command=self.connect_serial)
        self.connect_btn.grid(row=0, column=3, padx=5)

        self.test_btn = ttk.Button(top, text="Test Connection", command=self.test_serial_connection)
        self.test_btn.grid(row=0, column=4, padx=5)

        ttk.Label(top, text="Calibration Time (s)").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.calib_var = tk.StringVar(value="7.0")
        self.calib_entry = ttk.Entry(top, textvariable=self.calib_var, width=12)
        self.calib_entry.grid(row=1, column=1, padx=5, pady=(10, 0))

        self.calib_btn = ttk.Button(top, text="Calibrate", command=self.start_calibration)
        self.calib_btn.grid(row=1, column=2, padx=5, pady=(10, 0))

        ttk.Label(top, text="Record Time (s)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.record_var = tk.StringVar(value="10.0")
        self.record_entry = ttk.Entry(top, textvariable=self.record_var, width=12)
        self.record_entry.grid(row=2, column=1, padx=5, pady=(10, 0))

        ttk.Label(top, text="Trajectory Name").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.traj_name_var = tk.StringVar(value=self.get_next_trajectory_name())
        self.traj_name_entry = ttk.Entry(top, textvariable=self.traj_name_var, width=12)
        self.traj_name_entry.grid(row=3, column=1, padx=5, pady=(10, 0))

        self.record_btn = ttk.Button(top, text="Record", command=self.start_recording)
        self.record_btn.grid(row=2, column=2, padx=5, pady=(10, 0))

        ttk.Label(top, text="Load Trajectory").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.traj_list_var = tk.StringVar()
        self.traj_combo = ttk.Combobox(top, textvariable=self.traj_list_var, width=12, state="readonly")
        self.traj_combo.grid(row=4, column=1, padx=5, pady=(10, 0))

        self.load_btn = ttk.Button(top, text="Load", command=self.load_trajectory)
        self.load_btn.grid(row=4, column=2, padx=5, pady=(10, 0))

        self.replay_btn = ttk.Button(top, text="Replay", command=self.start_replay)
        self.replay_btn.grid(row=5, column=0, padx=5, pady=(10, 0), sticky="w")

        self.stop_btn = ttk.Button(top, text="Stop Replay", command=self.stop_replay)
        self.stop_btn.grid(row=5, column=1, padx=5, pady=(10, 0), sticky="w")

        self.follow_btn = ttk.Button(top, text="Start Following", command=self.shared_state.start_following)
        self.follow_btn.grid(row=5, column=2, padx=5, pady=(10, 0), sticky="w")

        self.stop_follow_btn = ttk.Button(top, text="Stop Following", command=self.shared_state.stop_following)
        self.stop_follow_btn.grid(row=5, column=3, padx=5, pady=(10, 0), sticky="w")


        ttk.Label(top, text="Plot Trajectory").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.plot_traj_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(top, textvariable=self.plot_traj_var, width=12, state="readonly")
        self.plot_combo.grid(row=6, column=1, padx=5, pady=(10, 0))

        self.plot_save_btn = ttk.Button(top, text="Plot and Save", command=self.plot_and_save_trajectory)
        self.plot_save_btn.grid(row=6, column=2, padx=5, pady=(10, 0))

        self.debug_btn = ttk.Button(top, text="Debug Raw IMU", command=self.debug_raw_imu)
        self.debug_btn.grid(row=6, column=3, padx=5, pady=(10, 0))

        ttk.Label(top, text="Max Linear Vel (m/s)").grid(row=7, column=0, sticky="w", pady=(10, 0))
        self.max_lin_vel_var = tk.StringVar(value="0.5")
        self.max_lin_vel_entry = ttk.Entry(top, textvariable=self.max_lin_vel_var, width=8)
        self.max_lin_vel_entry.grid(row=7, column=1, padx=5, pady=(10, 0), sticky="w")

        ttk.Label(top, text="Max Angular Vel (rad/s)").grid(row=7, column=2, sticky="w", pady=(10, 0))
        self.max_ang_vel_var = tk.StringVar(value="1.0")
        self.max_ang_vel_entry = ttk.Entry(top, textvariable=self.max_ang_vel_var, width=8)
        self.max_ang_vel_entry.grid(row=7, column=3, padx=5, pady=(10, 0), sticky="w")

        self.smoothing_var = tk.BooleanVar(value=False)
        self.smoothing_check = ttk.Checkbutton(top, text="Enable Trajectory Smoothing", variable=self.smoothing_var)
        self.smoothing_check.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))


        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.status_label = ttk.Label(top, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=9, column=0, columnspan=4, sticky="w", pady=(12, 0))

        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        self.ax_raw = self.fig.add_subplot(2, 1, 1)
        self.ax_traj = self.fig.add_subplot(2, 1, 2, projection="3d")

        self.line_ax, = self.ax_raw.plot([], [], label="ax")
        self.line_ay, = self.ax_raw.plot([], [], label="ay")
        self.line_az, = self.ax_raw.plot([], [], label="az")
        self.line_gx, = self.ax_raw.plot([], [], label="gx")
        self.line_gy, = self.ax_raw.plot([], [], label="gy")
        self.line_gz, = self.ax_raw.plot([], [], label="gz")

        self.ax_raw.set_title("Live Raw IMU Data")
        self.ax_raw.set_xlabel("Time (s)")
        self.ax_raw.set_ylabel("Value")
        self.ax_raw.grid(True)
        self.ax_raw.legend(loc="upper right")

        self.traj_line, = self.ax_traj.plot([], [], [], linewidth=2)
        self.traj_point, = self.ax_traj.plot([], [], [], marker="o", markersize=7)

        self.ax_traj.set_title("IMU Frame Trajectory", pad=5)
        self.ax_traj.set_xlabel("X IMU (m)")
        self.ax_traj.set_ylabel("Y IMU (m)")
        self.ax_traj.set_zlabel("Z IMU (m)")

        canvas_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.refresh_trajectory_list()

    def auto_detect_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino" in p.description or "CH340" in p.description or "USB Serial" in p.description:
                return p.device
        return "COM3"

    def refresh_ports(self):
        ports = list(serial.tools.list_ports.comports())
        if ports:
            self.port_var.set(ports[0].device)
        else:
            messagebox.showinfo("Ports", "No serial ports found.")

    def get_next_trajectory_name(self):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        pattern = os.path.join(self.trajectories_dir, "traj_*.txt")
        existing_files = glob.glob(pattern)

        if not existing_files:
            return "traj_1"

        numbers = []
        for file in existing_files:
            basename = os.path.basename(file)
            if basename.startswith("traj_") and basename.endswith(".txt"):
                try:
                    num_str = basename[5:-4]
                    numbers.append(int(num_str))
                except ValueError:
                    continue

        if numbers:
            return f"traj_{max(numbers) + 1}"
        else:
            return "traj_1"

    def refresh_trajectory_list(self):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        pattern = os.path.join(self.trajectories_dir, "*.txt")
        files = glob.glob(pattern)
        trajectory_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        trajectory_names.sort()

        self.traj_combo['values'] = trajectory_names
        self.plot_combo['values'] = trajectory_names
        if trajectory_names:
            self.traj_combo.set(trajectory_names[0])
            self.plot_combo.set(trajectory_names[0])

    def save_trajectory(self, times, positions, orientations, name):
        if not os.path.exists(self.trajectories_dir):
            os.makedirs(self.trajectories_dir)

        filepath = os.path.join(self.trajectories_dir, f"{name}.txt")
        stats = self.analyze_trajectory_stats(times, positions, orientations)

        with open(filepath, 'w') as f:
            f.write("# IMU Trajectory Data\n")
            f.write("# Format: time(s) x(m) y(m) z(m) roll(rad) pitch(rad) yaw(rad)\n")
            f.write(f"# Duration: {stats.get('duration', 0):.3f}s, Samples: {stats.get('samples', 0)}\n")
            f.write(f"# Max Linear Vel: {stats.get('max_linear_vel', 0):.3f} m/s, Max Angular Vel: {stats.get('max_angular_vel', 0):.3f} rad/s\n")
            f.write(f"# Max Linear Accel: {stats.get('max_linear_accel', 0):.3f} m/s², Max Angular Accel: {stats.get('max_angular_accel', 0):.3f} rad/s²\n")
            f.write(f"# Total Distance: {stats.get('total_distance', 0):.3f}m, Total Rotation: {stats.get('total_rotation', 0):.3f}rad\n")
            f.write(f"# Frame: IMU (X=right, Y=front, Z=up)\n")

            for i in range(len(times)):
                f.write(f"{times[i]:.6f} {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} "
                       f"{orientations[i,0]:.6f} {orientations[i,1]:.6f} {orientations[i,2]:.6f}\n")

    def load_trajectory(self):
        selected = self.traj_list_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a trajectory to load.")
            return

        filepath = os.path.join(self.trajectories_dir, f"{selected}.txt")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Trajectory file {selected}.txt not found.")
            return

        try:
            times = []
            positions = []
            orientations = []

            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 7:
                        times.append(float(parts[0]))
                        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        orientations.append([float(parts[4]), float(parts[5]), float(parts[6])])

            if len(times) < 2:
                raise ValueError("Not enough data points in trajectory file.")

            with self.state_lock:
                self.recorded_times = np.array(times)
                self.recorded_positions = np.array(positions)
                self.recorded_orientations = np.array(orientations)
                self.trajectory_ready = True
                self.replay_enabled = False
                self.current_trajectory_name = selected

            self.set_status(f"Loaded trajectory: {selected}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load trajectory: {str(e)}")

    def load_trajectory_data(self, filepath):
        times = []
        positions = []
        orientations = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue

                parts = line.split()
                if len(parts) >= 7:
                    times.append(float(parts[0]))
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    orientations.append([float(parts[4]), float(parts[5]), float(parts[6])])

        return np.array(times), np.array(positions), np.array(orientations)

    def plot_and_save_trajectory(self):
        selected = self.plot_traj_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a trajectory to plot.")
            return

        filepath = os.path.join(self.trajectories_dir, f"{selected}.txt")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Trajectory file {selected}.txt not found.")
            return

        try:
            times, positions, orientations = self.load_trajectory_data(filepath)

            if len(times) < 2:
                raise ValueError("Not enough data points in trajectory file.")

            smoothed_filepath = os.path.join(self.trajectories_dir, f"{selected}_smoothed.txt")
            has_smoothed = os.path.exists(smoothed_filepath)

            if has_smoothed:
                times_smooth, positions_smooth, orientations_smooth = self.load_trajectory_data(smoothed_filepath)

            fig1 = plt.figure(figsize=(15, 10))

            ax1 = fig1.add_subplot(2, 3, 1, projection='3d')
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Original', alpha=0.7)
            if has_smoothed:
                ax1.plot(positions_smooth[:, 0], positions_smooth[:, 1], positions_smooth[:, 2], 'r-', linewidth=2, label='Smoothed')
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
            ax1.set_xlabel('X IMU (m)')
            ax1.set_ylabel('Y IMU (m)')
            ax1.set_zlabel('Z IMU (m)')
            ax1.set_title('3D Trajectory (IMU Frame)')
            ax1.legend()
            ax1.grid(True)

            ax2 = fig1.add_subplot(2, 3, 2)
            ax2.plot(times, positions[:, 0], 'r-', label='X Original', linewidth=2, alpha=0.7)
            ax2.plot(times, positions[:, 1], 'g-', label='Y Original', linewidth=2, alpha=0.7)
            ax2.plot(times, positions[:, 2], 'b-', label='Z Original', linewidth=2, alpha=0.7)
            if has_smoothed:
                ax2.plot(times_smooth, positions_smooth[:, 0], 'r--', label='X Smoothed', linewidth=2)
                ax2.plot(times_smooth, positions_smooth[:, 1], 'g--', label='Y Smoothed', linewidth=2)
                ax2.plot(times_smooth, positions_smooth[:, 2], 'b--', label='Z Smoothed', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position IMU (m)')
            ax2.set_title('XYZ Position vs Time (IMU Frame)')
            ax2.legend()
            ax2.grid(True)

            ax3 = fig1.add_subplot(2, 3, 3)
            ax3.plot(times, np.degrees(orientations[:, 0]), 'r-', label='Roll Original', linewidth=2, alpha=0.7)
            ax3.plot(times, np.degrees(orientations[:, 1]), 'g-', label='Pitch Original', linewidth=2, alpha=0.7)
            ax3.plot(times, np.degrees(orientations[:, 2]), 'b-', label='Yaw Original', linewidth=2, alpha=0.7)
            if has_smoothed:
                ax3.plot(times_smooth, np.degrees(orientations_smooth[:, 0]), 'r--', label='Roll Smoothed', linewidth=2)
                ax3.plot(times_smooth, np.degrees(orientations_smooth[:, 1]), 'g--', label='Pitch Smoothed', linewidth=2)
                ax3.plot(times_smooth, np.degrees(orientations_smooth[:, 2]), 'b--', label='Yaw Smoothed', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Angular Position (degrees)')
            ax3.set_title('IMU Rotation vs Time')
            ax3.legend()
            ax3.grid(True)

            ax4 = fig1.add_subplot(2, 3, 4)
            ax4.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Original XY', alpha=0.7)
            if has_smoothed:
                ax4.plot(positions_smooth[:, 0], positions_smooth[:, 1], 'r-', linewidth=2, label='Smoothed XY')
            ax4.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
            ax4.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
            ax4.set_xlabel('X IMU (m)')
            ax4.set_ylabel('Y IMU (m)')
            ax4.set_title('XY Trajectory (IMU Frame Top View)')
            ax4.legend()
            ax4.grid(True)
            ax4.axis('equal')

            dt_orig = np.diff(times)
            pos_diff_orig = np.diff(positions, axis=0)
            vel_orig = np.linalg.norm(pos_diff_orig, axis=1) / dt_orig

            ax5 = fig1.add_subplot(2, 3, 5)
            ax5.plot(times[:-1], vel_orig, 'b-', linewidth=2, label='Original Velocity', alpha=0.7)
            if has_smoothed:
                dt_smooth = np.diff(times_smooth)
                pos_diff_smooth = np.diff(positions_smooth, axis=0)
                vel_smooth = np.linalg.norm(pos_diff_smooth, axis=1) / dt_smooth
                ax5.plot(times_smooth[:-1], vel_smooth, 'r-', linewidth=2, label='Smoothed Velocity')
                ax5.axhline(y=self.MAX_LINEAR_VELOCITY, color='k', linestyle='--', label='Velocity Limit')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Linear Velocity (m/s)')
            ax5.set_title('Velocity Profile Comparison')
            ax5.legend()
            ax5.grid(True)

            ax6 = fig1.add_subplot(2, 3, 6)
            stats_orig = self.analyze_trajectory_stats(times, positions, orientations)
            stats_text = f"Original Trajectory:\n"
            stats_text += f"Duration: {stats_orig.get('duration', 0):.2f}s\n"
            stats_text += f"Samples: {stats_orig.get('samples', 0)}\n"
            stats_text += f"Max Vel: {stats_orig.get('max_linear_vel', 0):.3f} m/s\n"
            stats_text += f"Avg Vel: {stats_orig.get('avg_linear_vel', 0):.3f} m/s\n"
            stats_text += f"Distance: {stats_orig.get('total_distance', 0):.3f}m\n"

            if has_smoothed:
                stats_smooth = self.analyze_trajectory_stats(times_smooth, positions_smooth, orientations_smooth)
                stats_text += f"\nSmoothed Trajectory:\n"
                stats_text += f"Duration: {stats_smooth.get('duration', 0):.2f}s\n"
                stats_text += f"Samples: {stats_smooth.get('samples', 0)}\n"
                stats_text += f"Max Vel: {stats_smooth.get('max_linear_vel', 0):.3f} m/s\n"
                stats_text += f"Avg Vel: {stats_smooth.get('avg_linear_vel', 0):.3f} m/s\n"
                stats_text += f"Distance: {stats_smooth.get('total_distance', 0):.3f}m\n"

            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis('off')
            ax6.set_title('Trajectory Statistics')

            plt.tight_layout()

            plot_filename = os.path.join(self.trajectories_dir, f"{selected}_comparison_plot.png")
            fig1.savefig(plot_filename, dpi=300, bbox_inches='tight')

            plt.show()

            status_msg = f"Comparison plot saved as {selected}_comparison_plot.png"
            if not has_smoothed:
                status_msg += " (No smoothed version found)"
            self.set_status(status_msg)

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot trajectory: {str(e)}")

    def smooth_trajectory(self, times, positions, orientations):
        if not self.TRAJECTORY_SMOOTHING or len(times) < 3:
            return times, positions, orientations

        if HAS_SCIPY:
            sigma = 2.0

            smoothed_positions = np.zeros_like(positions)
            smoothed_orientations = np.zeros_like(orientations)

            for i in range(3):
                smoothed_positions[:, i] = gaussian_filter1d(positions[:, i], sigma=sigma)
                smoothed_orientations[:, i] = gaussian_filter1d(orientations[:, i], sigma=sigma)

            return times.copy(), smoothed_positions, smoothed_orientations
        else:
            smoothed_positions = []
            smoothed_orientations = []
            smoothed_times = []

            window_size = 5

            for i in range(len(times)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(times), i + window_size // 2 + 1)

                pos_avg = np.mean(positions[start_idx:end_idx], axis=0)
                ori_avg = np.mean(orientations[start_idx:end_idx], axis=0)

                smoothed_times.append(times[i])
                smoothed_positions.append(pos_avg)
                smoothed_orientations.append(ori_avg)

            return np.array(smoothed_times), np.array(smoothed_positions), np.array(smoothed_orientations)

    def apply_acceleration_limits(self, times, positions, orientations):
        if len(times) < 3:
            return times, positions, orientations

        dt_avg = np.mean(np.diff(times))

        limited_positions = positions.copy()
        limited_orientations = orientations.copy()

        for iteration in range(3):
            pos_vel = np.gradient(limited_positions, dt_avg, axis=0)
            ori_vel = np.gradient(limited_orientations, dt_avg, axis=0)

            pos_vel_mag = np.linalg.norm(pos_vel, axis=1)
            ori_vel_mag = np.linalg.norm(ori_vel, axis=1)

            pos_scale = np.ones(len(pos_vel_mag))
            ori_scale = np.ones(len(ori_vel_mag))

            pos_over = pos_vel_mag > self.MAX_LINEAR_VELOCITY
            ori_over = ori_vel_mag > self.MAX_ANGULAR_VELOCITY

            pos_scale[pos_over] = self.MAX_LINEAR_VELOCITY / pos_vel_mag[pos_over]
            ori_scale[ori_over] = self.MAX_ANGULAR_VELOCITY / ori_vel_mag[ori_over]

            for i in range(3):
                pos_vel[:, i] *= pos_scale
                ori_vel[:, i] *= ori_scale

            limited_positions = np.cumsum(pos_vel * dt_avg, axis=0)
            limited_orientations = np.cumsum(ori_vel * dt_avg, axis=0)

            limited_positions += positions[0] - limited_positions[0]
            limited_orientations += orientations[0] - limited_orientations[0]

        return times, limited_positions, limited_orientations

    def analyze_trajectory_stats(self, times, positions, orientations):
        if len(times) < 2:
            return {}

        dt = np.diff(times)
        dt = dt[dt > 0]

        pos_diff = np.diff(positions, axis=0)
        ori_diff = np.diff(orientations, axis=0)

        linear_velocities = np.linalg.norm(pos_diff, axis=1) / dt
        angular_velocities = np.linalg.norm(ori_diff, axis=1) / dt

        linear_accelerations = np.diff(linear_velocities) / dt[:-1]
        angular_accelerations = np.diff(angular_velocities) / dt[:-1]

        stats = {
            'duration': times[-1] - times[0],
            'samples': len(times),
            'avg_sample_rate': len(times) / (times[-1] - times[0]) if times[-1] > times[0] else 0,
            'max_linear_vel': np.max(linear_velocities) if len(linear_velocities) > 0 else 0,
            'avg_linear_vel': np.mean(linear_velocities) if len(linear_velocities) > 0 else 0,
            'max_angular_vel': np.max(angular_velocities) if len(angular_velocities) > 0 else 0,
            'avg_angular_vel': np.mean(angular_velocities) if len(angular_velocities) > 0 else 0,
            'max_linear_accel': np.max(np.abs(linear_accelerations)) if len(linear_accelerations) > 0 else 0,
            'max_angular_accel': np.max(np.abs(angular_accelerations)) if len(angular_accelerations) > 0 else 0,
            'total_distance': np.sum(np.linalg.norm(pos_diff, axis=1)) if len(pos_diff) > 0 else 0,
            'total_rotation': np.sum(np.linalg.norm(ori_diff, axis=1)) if len(ori_diff) > 0 else 0
        }

        return stats

    def set_status(self, text):
        with self.state_lock:
            self.status_text = text
        self.root.after(0, lambda: self.status_var.set(f"Status: {text}"))

    def connect_serial(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Error", "Enter a COM port.")
            return

        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()

            self.ser = serial.Serial(port, 115200, timeout=1)
            self.header_seen = False

            time.sleep(0.1)
            self.ser.flushInput()

            self.ser.write(b'\n')
            self.ser.flush()
            time.sleep(0.1)

            self.set_status(f"Connected to {port} - Testing connection...")
            self.root.after(100, self.auto_test_connection)
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.set_status("Disconnected")

    def test_serial_connection(self):
        if not self.check_serial():
            messagebox.showwarning("No Connection", "Please connect to IMU first.")
            return

        self.set_status("Testing connection...")

        self.ser.flushInput()
        time.sleep(0.1)

        test_samples = []
        t0 = time.time()
        test_duration = 3.0
        header_detected = False

        while time.time() - t0 < test_duration:
            sample = self.read_one_sample()
            if sample is not None:
                test_samples.append(sample)
                if not header_detected:
                    header_detected = True
                    self.set_status("Data detected, collecting samples...")
            time.sleep(0.01)

        elapsed = time.time() - t0

        if len(test_samples) == 0:
            messagebox.showerror("Test Failed",
                "No valid IMU data received. Check:\n" +
                "• Arduino is connected and powered\n" +
                "• Correct COM port selected\n" +
                "• Arduino code is running\n" +
                "• Baud rate is 115200\n" +
                "• USB cable is working")
            self.set_status("Test failed: No data")
            return

        rate = len(test_samples) / elapsed

        # Analyze the data quality
        times = [s[0] for s in test_samples]
        accels = [(s[1], s[2], s[3]) for s in test_samples]
        gyros = [(s[4], s[5], s[6]) for s in test_samples]

        # Check if data looks reasonable
        avg_accel_mag = np.mean([np.sqrt(a[0]**2 + a[1]**2 + a[2]**2) for a in accels])

        if rate < 50:
            messagebox.showwarning("Low Data Rate",
                f"Data rate: {rate:.1f} Hz (Expected: ~100 Hz)\n" +
                f"Samples collected: {len(test_samples)}\n" +
                f"Average acceleration magnitude: {avg_accel_mag:.2f} m/s²\n\n" +
                "This may work but calibration will take longer.")
            self.set_status(f"Test warning: {rate:.1f} Hz")
        else:
            messagebox.showinfo("Test Successful",
                f"✓ Connection working perfectly!\n\n" +
                f"Data rate: {rate:.1f} Hz\n" +
                f"Samples in {elapsed:.1f}s: {len(test_samples)}\n" +
                f"Average acceleration: {avg_accel_mag:.2f} m/s²\n\n" +
                "Ready for calibration and recording!")
            self.set_status(f"Test passed: {rate:.1f} Hz")

    def auto_test_connection(self):
        if not self.check_serial():
            return

        self.ser.flushInput()
        time.sleep(0.1)

        test_samples = []
        t0 = time.time()
        test_duration = 2.0

        while time.time() - t0 < test_duration:
            sample = self.read_one_sample()
            if sample is not None:
                test_samples.append(sample)
            time.sleep(0.01)

        elapsed = time.time() - t0

        if len(test_samples) == 0:
            self.set_status("Connection failed - No IMU data received")
            return

        rate = len(test_samples) / elapsed

        if rate < 50:
            self.set_status(f"Connected - Low data rate: {rate:.1f} Hz")
        else:
            self.set_status(f"Connected - Ready! Data rate: {rate:.1f} Hz")

    def debug_raw_imu(self):
        if not self.check_serial():
            messagebox.showwarning("No Connection", "Please connect to IMU first.")
            return

        self.set_status("Debug mode: Collecting 5 seconds of raw IMU data...")

        raw_samples = []
        t0 = time.time()
        debug_duration = 5.0

        while time.time() - t0 < debug_duration:
            sample = self.read_one_sample()
            if sample is not None:
                t_us, ax, ay, az, gx, gy, gz = sample
                raw_samples.append([time.time() - t0, ax, ay, az, gx, gy, gz])
            time.sleep(0.01)

        if len(raw_samples) < 10:
            messagebox.showerror("Debug Failed", "Not enough raw data collected.")
            return

        raw_data = np.array(raw_samples)
        times = raw_data[:, 0]
        accels = raw_data[:, 1:4]
        gyros = raw_data[:, 4:7]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(times, accels[:, 0], 'r-', label='X accel', linewidth=2)
        ax1.plot(times, accels[:, 1], 'g-', label='Y accel', linewidth=2)
        ax1.plot(times, accels[:, 2], 'b-', label='Z accel', linewidth=2)
        ax1.set_title('Raw Accelerometer Data')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(times, gyros[:, 0], 'r-', label='X gyro', linewidth=2)
        ax2.plot(times, gyros[:, 1], 'g-', label='Y gyro', linewidth=2)
        ax2.plot(times, gyros[:, 2], 'b-', label='Z gyro', linewidth=2)
        ax2.set_title('Raw Gyroscope Data')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.legend()
        ax2.grid(True)

        accels_cal = accels - self.accel_bias
        gyros_cal = gyros - self.gyro_bias

        ax3.plot(times, accels_cal[:, 0], 'r-', label='X accel (cal)', linewidth=2)
        ax3.plot(times, accels_cal[:, 1], 'g-', label='Y accel (cal)', linewidth=2)
        ax3.plot(times, accels_cal[:, 2], 'b-', label='Z accel (cal)', linewidth=2)
        ax3.set_title('Calibrated Accelerometer Data')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(times, gyros_cal[:, 0], 'r-', label='X gyro (cal)', linewidth=2)
        ax4.plot(times, gyros_cal[:, 1], 'g-', label='Y gyro (cal)', linewidth=2)
        ax4.plot(times, gyros_cal[:, 2], 'b-', label='Z gyro (cal)', linewidth=2)
        ax4.set_title('Calibrated Gyroscope Data')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

        accel_means = np.mean(np.abs(accels_cal), axis=0)
        gyro_means = np.mean(np.abs(gyros_cal), axis=0)

        messagebox.showinfo("Debug Results",
            f"Raw IMU Data Analysis:\n\n" +
            f"Average |Acceleration| (m/s²):\n" +
            f"  X: {accel_means[0]:.3f}\n" +
            f"  Y: {accel_means[1]:.3f}\n" +
            f"  Z: {accel_means[2]:.3f}\n\n" +
            f"Average |Angular Velocity| (rad/s):\n" +
            f"  X: {gyro_means[0]:.3f}\n" +
            f"  Y: {gyro_means[1]:.3f}\n" +
            f"  Z: {gyro_means[2]:.3f}\n\n" +
            f"Move IMU in different directions and\n" +
            f"note which axis shows the most change!")

        self.set_status("Debug complete - Check plots and results")

    def start_calibration(self):
        if not self.check_serial():
            return
        try:
            self.calibration_time = float(self.calib_var.get())
            if self.calibration_time <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Calibration time must be a positive number.")
            return

        with self.state_lock:
            self.request_calibrate = True

    def start_recording(self):
        if not self.check_serial():
            return
        try:
            self.record_time = float(self.record_var.get())
            if self.record_time <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Record time must be a positive number.")
            return

        with self.state_lock:
            self.request_record = True

    def start_replay(self):
        with self.state_lock:
            if not self.trajectory_ready:
                messagebox.showwarning("No trajectory", "Record a trajectory first.")
                return
            self.request_replay = True

    def stop_replay(self):
        with self.state_lock:
            self.request_stop = True

    def check_serial(self):
        return self.ser is not None and self.ser.is_open

    def read_one_sample(self):
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                if not self.header_seen:
                    if line == "t,ax,ay,az,gx,gy,gz" or line.startswith("t,ax,ay,az,gx,gy,gz"):
                        self.header_seen = True
                        continue

                    parts = line.split(",")
                    if len(parts) == 7:
                        try:
                            t_us, ax, ay, az, gx, gy, gz = map(float, parts)
                            if 1000000 < t_us < 999999999999 and -50 < ax < 50 and -50 < ay < 50 and -50 < az < 50:
                                self.header_seen = True
                                return t_us, ax, ay, az, gx, gy, gz
                        except ValueError:
                            pass
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    continue

                try:
                    t_us, ax, ay, az, gx, gy, gz = map(float, parts)
                    return t_us, ax, ay, az, gx, gy, gz
                except ValueError:
                    continue
        except Exception as e:
            pass

        return None

    def rotation_matrix(self, roll, pitch, yaw):
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr]
        ])

        ry = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp]
        ])

        rz = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0]
        ])

        return rz @ ry @ rx

    def set_axes_equal_3d(self, ax, x, y, z):
        if len(x) < 2:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            return

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        xmid = 0.5 * (x.min() + x.max())
        ymid = 0.5 * (y.min() + y.max())
        zmid = 0.5 * (z.min() + z.max())

        xr = max(x.max() - x.min(), 0.1)
        yr = max(y.max() - y.min(), 0.1)
        zr = max(z.max() - z.min(), 0.1)

        r = 0.5 * max(xr, yr, zr)

        ax.set_xlim(xmid - r, xmid + r)
        ax.set_ylim(ymid - r, ymid + r)
        ax.set_zlim(zmid - r, zmid + r)

    def calibrate_imu(self):
        self.set_status(f"Calibrating for {self.calibration_time:.2f} s. Keep IMU still.")

        samples = []
        t0 = time.time()
        last_status_time = t0
        header_attempts = 0
        no_data_count = 0

        while time.time() - t0 < self.calibration_time and self.running:
            sample = self.read_one_sample()
            if sample is None:
                no_data_count += 1
                time.sleep(0.005)

                current_time = time.time()
                if current_time - last_status_time > 1.0:
                    elapsed = current_time - t0
                    remaining = self.calibration_time - elapsed
                    if not self.header_seen:
                        header_attempts += 1
                        self.set_status(f"Waiting for IMU header... ({remaining:.1f}s remaining, attempt {header_attempts})")
                    else:
                        self.set_status(f"Collecting samples: {len(samples)} collected ({remaining:.1f}s remaining)")
                    last_status_time = current_time
                continue

            samples.append(sample)

            if len(samples) % 50 == 0:
                elapsed = time.time() - t0
                remaining = self.calibration_time - elapsed
                rate = len(samples) / elapsed if elapsed > 0 else 0
                self.set_status(f"Calibrating: {len(samples)} samples at {rate:.1f} Hz ({remaining:.1f}s remaining)")

        elapsed_total = time.time() - t0

        if not self.header_seen and len(samples) == 0:
            raise RuntimeError(f"No IMU data detected after {elapsed_total:.1f}s. Check:\n• Arduino connection and power\n• Correct COM port\n• Arduino code running\n• USB cable working")

        if len(samples) < 20:
            rate = len(samples) / elapsed_total if elapsed_total > 0 else 0
            raise RuntimeError(f"Not enough samples during calibration: got {len(samples)}/20 minimum in {elapsed_total:.1f}s (rate: {rate:.1f} Hz). Check IMU connection and data rate.")

        self.set_status(f"Processing {len(samples)} calibration samples...")

        acc = np.array([[s[1], s[2], s[3]] for s in samples], dtype=float)
        gyr = np.array([[s[4], s[5], s[6]] for s in samples], dtype=float)

        gyro_bias = gyr.mean(axis=0)

        acc_mean = acc.mean(axis=0)
        acc_norm = np.linalg.norm(acc_mean)
        if acc_norm < 1e-6:
            raise RuntimeError("Bad accelerometer calibration: near-zero acceleration detected.")

        gravity_body = self.G * acc_mean / acc_norm
        accel_bias = acc_mean - gravity_body
        self.gravity_direction = gravity_body.copy()

        final_rate = len(samples) / elapsed_total
        self.set_status(f"Calibration complete: {len(samples)} samples at {final_rate:.1f} Hz")

        return gyro_bias, accel_bias

    def record_trajectory(self):
        self.set_status(f"Recording for {self.record_time:.2f} s. Move the IMU now.")

        raw_samples = []
        first_sample_time = None

        with self.state_lock:
            self.raw_t.clear()
            self.raw_ax.clear()
            self.raw_ay.clear()
            self.raw_az.clear()
            self.raw_gx.clear()
            self.raw_gy.clear()
            self.raw_gz.clear()

        while self.running:
            sample = self.read_one_sample()
            if sample is None:
                time.sleep(0.001)
                continue

            t_us, ax, ay, az, gx, gy, gz = sample
            t = t_us / 1e6

            with self.state_lock:
                self.raw_t.append(t)
                self.raw_ax.append(ax)
                self.raw_ay.append(ay)
                self.raw_az.append(az)
                self.raw_gx.append(gx)
                self.raw_gy.append(gy)
                self.raw_gz.append(gz)

            if first_sample_time is None:
                first_sample_time = t

            elapsed = t - first_sample_time
            if elapsed > self.record_time:
                break

            raw_samples.append([elapsed, ax, ay, az, gx, gy, gz])

        if len(raw_samples) < 10:
            raise RuntimeError("Not enough samples recorded.")

        return self.process_trajectory_data(raw_samples)

    def process_trajectory_data(self, raw_samples):
        raw_data = np.array(raw_samples)
        times = raw_data[:, 0]
        accels = raw_data[:, 1:4] - self.accel_bias
        gyros = raw_data[:, 4:7] - self.gyro_bias

        n = len(times)
        positions = np.zeros((n, 3))
        orientations = np.zeros((n, 3))
        velocities = np.zeros((n, 3))

        a0 = accels[0]
        a0_norm = np.linalg.norm(a0)
        if a0_norm > 1e-6:
            roll = np.arctan2(a0[1], a0[2])
            pitch = np.arctan2(-a0[0], np.sqrt(a0[1]**2 + a0[2]**2))
        else:
            roll, pitch = 0.0, 0.0
        yaw = 0.0
        orientations[0] = [roll, pitch, yaw]

        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt <= 0 or dt > 0.5:
                positions[i] = positions[i - 1]
                orientations[i] = orientations[i - 1]
                velocities[i] = velocities[i - 1]
                continue

            ax, ay, az = accels[i]
            gx, gy, gz = gyros[i]
            prev_roll, prev_pitch, prev_yaw = orientations[i - 1]

            # Gyro-predicted orientation
            roll_gyro = prev_roll + gx * dt
            pitch_gyro = prev_pitch + gy * dt
            yaw_gyro = prev_yaw + gz * dt

            # Accel-based roll/pitch (only reliable when near 1g, i.e. not accelerating)
            accel_norm = np.sqrt(ax**2 + ay**2 + az**2)
            if 0.5 * self.G < accel_norm < 1.5 * self.G:
                roll_acc = np.arctan2(ay, az)
                pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
                roll = self.ALPHA * roll_gyro + (1 - self.ALPHA) * roll_acc
                pitch = self.ALPHA * pitch_gyro + (1 - self.ALPHA) * pitch_acc
            else:
                roll = roll_gyro
                pitch = pitch_gyro
            yaw = yaw_gyro
            orientations[i] = [roll, pitch, yaw]

            # Rotate gravity into sensor frame and subtract to get linear acceleration
            R = self.rotation_matrix(roll, pitch, yaw)
            gravity_in_sensor = R.T @ np.array([0.0, 0.0, self.G])
            linear_accel_sensor = accels[i] - gravity_in_sensor

            # Transform linear acceleration into world frame for position integration
            linear_accel_world = R @ linear_accel_sensor

            # Zero-velocity update: detect stationary periods
            gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            accel_dev = abs(accel_norm - self.G)
            is_stationary = (gyro_mag < self.GYRO_STATIONARY_THRESH and
                             accel_dev < self.ACCEL_STATIONARY_THRESH)

            if is_stationary:
                velocities[i] = velocities[i - 1] * self.ZUPT_VELOCITY_DECAY
                if np.linalg.norm(velocities[i]) < self.VEL_DEADBAND:
                    velocities[i] = np.zeros(3)
            else:
                accel_filtered = linear_accel_world.copy()
                accel_filtered[np.abs(accel_filtered) < self.ACCEL_DEADBAND] = 0.0

                velocities[i] = velocities[i - 1] + accel_filtered * dt
                velocities[i] *= self.VELOCITY_DECAY

                vel_mag = np.linalg.norm(velocities[i])
                if vel_mag > self.MAX_VELOCITY:
                    velocities[i] *= self.MAX_VELOCITY / vel_mag

            positions[i] = positions[i - 1] + 0.5 * (velocities[i] + velocities[i - 1]) * dt

        return times, positions, orientations

    def apply_axis_mapping(self, data):
        # IMU frame: X=right(+)/left(-), Y=front(+)/back(-), Z=up(+)/down(-)
        # No remapping needed when sensor axes match desired frame.
        # If your sensor axes differ, adjust signs/order here, but keep
        # the mapping a proper rotation (determinant +1) or the orientation
        # tracking math will break.
        return data.copy()

    def worker_loop(self):
        while self.running:
            do_calib = False
            do_record = False
            do_replay = False
            do_stop = False

            with self.state_lock:
                if self.request_calibrate:
                    self.request_calibrate = False
                    do_calib = True
                if self.request_record:
                    self.request_record = False
                    do_record = True
                if self.request_replay:
                    self.request_replay = False
                    do_replay = True
                if self.request_stop:
                    self.request_stop = False
                    do_stop = True

            try:
                if do_calib:
                    if not self.check_serial():
                        self.set_status("Disconnected")
                        continue
                    self.gyro_bias, self.accel_bias = self.calibrate_imu()
                    self.set_status("Calibration complete.")

                elif do_record:
                    if not self.check_serial():
                        self.set_status("Disconnected")
                        continue

                    trajectory_name = self.traj_name_var.get().strip()
                    if not trajectory_name:
                        trajectory_name = self.get_next_trajectory_name()

                    self.gyro_bias, self.accel_bias = self.calibrate_imu()
                    times, positions, orientations = self.record_trajectory()

                    self.save_trajectory(times, positions, orientations, trajectory_name)

                    try:
                        self.MAX_LINEAR_VELOCITY = float(self.max_lin_vel_var.get())
                        self.MAX_ANGULAR_VELOCITY = float(self.max_ang_vel_var.get())
                        self.TRAJECTORY_SMOOTHING = self.smoothing_var.get()
                    except ValueError:
                        pass

                    if self.TRAJECTORY_SMOOTHING:
                        s_times, s_pos, s_ori = self.smooth_trajectory(
                            times, positions, orientations)
                        s_times, s_pos, s_ori = self.apply_acceleration_limits(
                            s_times, s_pos, s_ori)
                        self.save_trajectory(s_times, s_pos, s_ori,
                                             f"{trajectory_name}_smoothed")
                        times, positions, orientations = s_times, s_pos, s_ori

                    with self.state_lock:
                        self.recorded_times = times
                        self.recorded_positions = positions
                        self.recorded_orientations = orientations
                        self.trajectory_ready = True
                        self.replay_enabled = False
                        self.current_trajectory_name = trajectory_name

                    self.root.after(0, self.refresh_trajectory_list)
                    self.root.after(0, lambda: self.traj_name_var.set(
                        self.get_next_trajectory_name()))

                    status_msg = f"Recording complete. Saved as {trajectory_name}."
                    if self.TRAJECTORY_SMOOTHING:
                        status_msg += (f" Smoothed version saved as"
                                       f" {trajectory_name}_smoothed.")
                    self.set_status(status_msg)

                elif do_replay:
                    with self.state_lock:
                        if self.trajectory_ready:
                            self.replay_enabled = True
                            self.mode = "replaying"
                    self.replay_start_wall = time.time()
                    self.set_status("Replay enabled.")

                elif do_stop:
                    with self.state_lock:
                        self.replay_enabled = False
                        self.mode = "idle"
                    self.set_status("Replay stopped.")

            except Exception as e:
                self.set_status(f"Error: {e}")

            time.sleep(0.02)

    def update_plots(self, frame):
        with self.state_lock:
            rt = list(self.raw_t)
            rax = list(self.raw_ax)
            ray = list(self.raw_ay)
            raz = list(self.raw_az)
            rgx = list(self.raw_gx)
            rgy = list(self.raw_gy)
            rgz = list(self.raw_gz)

            traj_ready = self.trajectory_ready
            replay_enabled = self.replay_enabled
            traj_t = self.recorded_times.copy()
            traj_p = self.recorded_positions.copy()
            traj_o = self.recorded_orientations.copy()
            replay_start = self.replay_start_wall

        if len(rt) > 0:
            t0 = rt[0]
            tp = [x - t0 for x in rt]

            self.line_ax.set_data(tp, rax)
            self.line_ay.set_data(tp, ray)
            self.line_az.set_data(tp, raz)
            self.line_gx.set_data(tp, rgx)
            self.line_gy.set_data(tp, rgy)
            self.line_gz.set_data(tp, rgz)

            xmin = tp[0]
            xmax = tp[-1] if tp[-1] > 1.0 else 1.0
            self.ax_raw.set_xlim(xmin, xmax)

            vals = rax + ray + raz + rgx + rgy + rgz
            ymin = min(vals)
            ymax = max(vals)
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            self.ax_raw.set_ylim(ymin - 0.5, ymax + 0.5)

        if traj_ready and len(traj_t) >= 2 and len(traj_p) >= 2:
            if replay_enabled:
                duration = traj_t[-1]
                if duration <= 0:
                    duration = 1.0
                tplay = (time.time() - replay_start) % duration
                idx = np.searchsorted(traj_t, tplay, side="right")
                idx = max(1, min(idx, len(traj_t)))
                seg = traj_p[:idx]
            else:
                seg = traj_p

            if len(seg) > 0:
                self.traj_line.set_data(seg[:, 0], seg[:, 1])
                self.traj_line.set_3d_properties(seg[:, 2])

                if len(seg) >= 1:
                    self.traj_point.set_data([seg[-1, 0]], [seg[-1, 1]])
                    self.traj_point.set_3d_properties([seg[-1, 2]])

                self.set_axes_equal_3d(self.ax_traj, traj_p[:, 0], traj_p[:, 1], traj_p[:, 2])
        else:
            self.traj_line.set_data([], [])
            self.traj_line.set_3d_properties([])
            self.traj_point.set_data([], [])
            self.traj_point.set_3d_properties([])

        self.canvas.draw_idle()

        return (
            self.line_ax, self.line_ay, self.line_az,
            self.line_gx, self.line_gy, self.line_gz,
            self.traj_line, self.traj_point
        )

    def on_close(self):
        self.running = False
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = IMUGUI(root)
    root.mainloop()