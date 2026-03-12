import threading
import time
import numpy as np
import urx

class URXControlThread(threading.Thread):
    def __init__(self, shared_state, robot_ip, hz=100, vj=0.5, aj=0.1):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.robot_ip = robot_ip
        self.dt = 1.0 / hz
        self.robot = None
        self.running = True
        self.vj = vj
        self.aj = aj

    def run(self):
        try:
            print(f"[URX] Connecting to robot at {self.robot_ip}...")
            self.robot = urx.Robot(self.robot_ip)

            self.shared_state.joint_pos = self.robot.getj()

            with self.shared_state.lock:
                self.shared_state.robot_connected = True

            print("[URX] Connected.")

            # TODO: FIRST MAKE SURE THAT WE ARE AT THE CONSTANT STARTING POINT

            while self.running:
                with self.shared_state.lock:
                    shutdown = self.shared_state.shutdown
                    enabled = self.shared_state.robot_enabled
                    u_curr = self.shared_state.u_curr.copy()

                if shutdown:
                    break

                try:
                    if enabled:
                        self.send_command(u_curr)
                    else:
                        self.send_zero()
                except Exception as e:
                    print(f"[URX] Command error: {e}")
                    self.send_zero()
                    time.sleep(0.1)

                time.sleep(self.dt)

        except Exception as e:
            print(f"[URX] Connection error: {e}")

        finally:
            try:
                if self.robot is not None:
                    self.send_zero()
                    self.robot.close()
            except Exception:
                pass

            with self.shared_state.lock:
                self.shared_state.robot_connected = False

            print("[URX] Thread exited.")

    def send_command(self, u):
        cmd = np.array(u).reshape(-1)

        joint_vels = cmd.tolist()
        joint_vels = np.clip(joint_vels, -self.vj, self.vj).tolist()
        print(joint_vels)

        self.robot.speedj(
            joint_vels,
            acc=self.aj,
            min_time=self.dt
        )

    def send_zero(self):
        if self.robot is not None:
            self.robot.speedj([0, 0, 0, 0, 0, 0], acc=self.aj, min_time=self.dt)

    def stop(self):
        self.running = False