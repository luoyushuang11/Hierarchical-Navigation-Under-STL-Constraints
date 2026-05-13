# config.py
import numpy as np

# General parameters
dt = 0.02  # Time step

# Parameters for MPPI
# N = 200  # horizon for maze
N = 300
n_samples = 500  # Number of samples for MPPI

# Q = np.diag([3.0, 3.0, 0.0, 10.0])  # Weight for state
# QT = Q.copy() * N / 5  # Weight for terminal state
# q_ref = np.array([-2.0, -2.0, 0.0, 0.0])  # Reference state
R = np.diag([20.0, 20.0])  # Weight for control
MAX_OMEGA = 1.0   # (rad/s)
MAX_ACCEL = 4.0   # (m/s^2)
V_LIMIT = 3.0     # for overtaking


NOISE_SIGMA = np.array([0.3, 0.3])
Temperature = 1.0


T = 35  
#T = 25  

ratio_sim_mppi = 100  # how fast simulator run faster than mppi controller





