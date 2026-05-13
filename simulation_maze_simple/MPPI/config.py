# config.py
import numpy as np

# General parameters
dt = 0.02  # Time step

# Parameters for MPPI
# N = 200  # horizon for maze
N = 150
n_samples = 500  # Number of samples for MPPI

# Q = np.diag([3.0, 3.0, 0.0, 10.0])  # Weight for state
# QT = Q.copy() * N / 5  # Weight for terminal state
# q_ref = np.array([-2.0, -2.0, 0.0, 0.0])  # Reference state
R = np.diag([2.0, 2.0])  # Weight for control
MAX_OMEGA = 1.0   
MAX_ACCEL = 4.0  
V_LIMIT = 2.0     



NOISE_SIGMA = np.array([0.5, 0.5])
Temperature = 1.0

# Simulation parameters

T = 40  # for office 
ratio_sim_mppi = 100  # how fast simulator run faster than mppi controller





