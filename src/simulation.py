import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((6, 0))

        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u, error = self.controller(x, i * self.dt)
            x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(6, 1), axis=1)

        _, error = self.controller(x_next, i * self.dt)
        e_vec = np.append(e_vec, error.reshape((6, 1)), axis=1)

        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data for UR10e joint-space model.
        State: q in R^6
        Input: qdot in R^6
        """
        variables = [self.t, self.x_vec, self.u_vec, self.sim_loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")
            return

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Joint positions
        ax1.clear()
        ax1.set_title("UR10e Joint States")
        for i in range(6):
            ax1.plot(t, x_vec[i, :], '--', label=f"q{i+1}")
        ax1.set_ylabel("Joint Angle [rad]")
        ax1.grid()
        ax1.legend()

        # Joint velocities
        ax2.clear()
        ax2.set_title("UR10e Joint Velocity Commands")
        for i in range(6):
            ax2.plot(t[:-1], u_vec[i, :], '--', label=f"qdot{i+1}")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Joint Velocity [rad/s]")
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def visualize_error(self):
        """
        Offline plotting of tracking error for UR10e joint-space model.
        Error: q - q_ref in R^6
        Input: qdot in R^6
        """
        variables = [self.t, self.e_vec, self.u_vec, self.sim_loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")
            return

        t = self.t
        e_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Joint position tracking error
        ax1.clear()
        ax1.set_title("UR10e Joint Tracking Error")
        for i in range(6):
            ax1.plot(t, e_vec[i, :], '--', label=f"e_q{i+1}")
        ax1.set_ylabel("Joint Error [rad]")
        ax1.grid()
        ax1.legend()

        # Joint velocity commands
        ax2.clear()
        ax2.set_title("UR10e Joint Velocity Commands")
        for i in range(6):
            ax2.plot(t[:-1], u_vec[i, :], '--', label=f"qdot{i+1}")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Joint Velocity [rad/s]")
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.show()