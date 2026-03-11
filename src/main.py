import numpy as np

from src.ur10e import UR10e
from src.mpc import MPC
from src.simulation import EmbeddedSimEnvironment

# Q1
ur10e = UR10e()

# Instantiate controller
x_lim, u_lim, delta_u_lim = ur10e.get_limits()

# Create MPC Solver
MPC_HORIZON = 10
tracking_ctl = MPC(model=ur10e,
                   dynamics=ur10e.model,
                   param='P1',
                   N=MPC_HORIZON,
                   xlb=-x_lim, xub=x_lim,
                   ulb=-u_lim, uub=u_lim,
                   delta_ulb=-delta_u_lim, delta_uub=delta_u_lim)
sim_env_tracking = EmbeddedSimEnvironment(model=ur10e,
                                          dynamics=ur10e.model,
                                          controller=tracking_ctl.mpc_controller,
                                          time=13)
x0 = ur10e.get_initial_pose()
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()