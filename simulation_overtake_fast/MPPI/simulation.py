# simulation.py
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax
import time  


import config
from model import jax_dynamics
from cbf_functions import h_x_numpy
from controller import jax_mppi


@jax.jit
def _simulation_step(sim_state, u_mppi_cbf, num_steps, dt_per_step):

    def loop_body(i, state):
        next_state = jax_dynamics(state, u_mppi_cbf, dt_per_step)
        return next_state
    final_state = lax.fori_loop(0, num_steps, loop_body, sim_state)
    return final_state

def run_simulation(initial_state, initial_U, rng_key):
    """
    Run a complete closed-loop simulation.
    """

    
    # Initialize
    sim_state = initial_state
    global_U = initial_U
    
    # Logging list
    states = [np.array(sim_state)]
    h_values = [h_x_numpy(sim_state, 0.0)]
    executed_controls = []
    mppi_sampled_us = [np.zeros((config.n_samples, config.N, 2))]
    global_Us = [np.array(global_U)]
    
    computation_times = [] # <--- 2. Add a list for timing

    print(f"Start the simulation... (Total duration: {config.T}s, Step length: {config.dt}s)")
    
    # Optimized main loop
    for t in np.arange(0, config.T, config.dt):

        # --- 3. Start Timing ---
        start_time = time.perf_counter()
        
        rng_key, subkey = random.split(rng_key)
        
        # 1. JAX running on GPU (queued)
        u_mppi_cbf, global_U, sampled_us = jax_mppi(sim_state, global_U, subkey, t, cbf_cost_weight=1e4)
        
        # 2. JAX continues to run on GPU (queued)
        sim_state = _simulation_step(sim_state, u_mppi_cbf, config.ratio_sim_mppi, config.dt / config.ratio_sim_mppi)

        # 3. Forcing Python to wait for the GPU to do all the work
        sim_state.block_until_ready()
        
        # --- 4. End timer and record ---
        end_time = time.perf_counter()
        computation_times.append(end_time - start_time)

        # 4. Log logs (converted to Numpy)
        current_state_np = np.array(sim_state)
        states.append(current_state_np)
        # ... (No change in the rest of the log records) ...
        mppi_sampled_us.append(np.array(sampled_us))
        global_Us.append(np.array(global_U))
        h_values.append(h_x_numpy(current_state_np, t + config.dt)) 
        executed_controls.append(np.array(u_mppi_cbf))

        if (t * 10) % (config.T) == 0:
             print(f"  Simulation progress: {t:.1f}s / {config.T}s")

    print("Simulation completed.")
    
    # --- 5. Returns the timing results ---
    return (
        np.array(states), h_values, executed_controls, 
        mppi_sampled_us, global_Us, computation_times
    )