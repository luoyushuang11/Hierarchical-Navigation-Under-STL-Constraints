# config.py
import numpy as np

# General parameters
dt = 0.02  # Time step

# Parameters for MPPI
# N = 200  # horizon for maze
N = 200
n_samples = 500 

R = np.diag([20.0, 20.0])
MAX_OMEGA = 1.0   
MAX_ACCEL = 4.0   
V_LIMIT = 2.0     



NOISE_SIGMA = np.array([0.3, 0.3])
Temperature = 1.0

# Simulation parameters
T = 60  



ratio_sim_mppi = 100  # how fast simulator run faster than mppi controller





