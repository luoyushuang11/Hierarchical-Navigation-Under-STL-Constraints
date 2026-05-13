# config.py
import numpy as np

# General parameters
dt = 0.02  # Time step

# Parameters for MPPI
# N = 200  # horizon for maze
N = 200
n_samples = 500  # Number of samples for MPPI

# Q = np.diag([3.0, 3.0, 0.0, 10.0])  # Weight for state
# QT = Q.copy() * N / 5  # Weight for terminal state
# q_ref = np.array([-2.0, -2.0, 0.0, 0.0])  # Reference state
R = np.diag([20.0, 20.0])  # Weight for control
MAX_OMEGA = 1.0   # Maximum angular velocity (rad/s)
MAX_ACCEL = 4.0   # Maximum acceleration (m/s^2)
V_LIMIT = 2.0     # Maximum speed (m/s)



NOISE_SIGMA = np.array([0.3, 0.3])
Temperature = 1.0

# Simulation parameters
T = 60  # Total time for simulation 



ratio_sim_mppi = 100  # how fast simulator run faster than mppi controller





