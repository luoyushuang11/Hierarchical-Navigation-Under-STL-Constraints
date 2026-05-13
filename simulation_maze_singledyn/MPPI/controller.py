# MPPI/controller.py
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

import config
from model import jax_dynamics
from cbf_functions import h_x_jax

@jax.jit
def jax_mppi(state, U, rng_key, t_start,
             n_samples=config.n_samples,
             N=config.N,
             R=config.R,
             dt=config.dt,
             cbf_cost_weight=1e4):

    # --- Cost Calculation Function ---
    def scan_fn(carry, u): 
        cost, sim_state, t_k = carry
        
        # 1. Cost of control: u is a physical quantity, and R is adjusted for a physical magnitude
        new_cost = cost + jnp.dot(u, jnp.dot(R, u))
        
        # 2. Kinetic deduction
        new_state = jax_dynamics(sim_state, u, dt=dt)

        t_next = t_k + dt

        # 3. CBF Barrier Cost
        barrier_value = h_x_jax(new_state, t_next)
        cbf_cost = cbf_cost_weight * jnp.clip(-barrier_value, 0.0, np.inf)

        new_cost += cbf_cost
        return (new_cost, new_state, t_next), None

    # --- Sampling and Evaluation ---
    def single_sample_cost(rng_subkey):
        # [Modified] Noise Generation (Physical Space)
        noise = random.normal(rng_subkey, (N, 2))
        # noise_scaled: [N, 2] * [sigma_omega, sigma_accel]
        noise_scaled = noise * jnp.array(config.NOISE_SIGMA)
        
        u_seq = U + noise_scaled
        
        # [Core Modification] Physical Clipping
        # Targeting Omega truncation
        u_seq = u_seq.at[:, 0].set(jnp.clip(u_seq[:, 0], -config.MAX_OMEGA, config.MAX_OMEGA))
        # Truncated against Accel
        u_seq = u_seq.at[:, 1].set(jnp.clip(u_seq[:, 1], -config.MAX_ACCEL, config.MAX_ACCEL))
        
        # Scan the track
        cost_and_state, _ = lax.scan(scan_fn, (0.0, state, t_start), u_seq)
        
        return (cost_and_state[0], cost_and_state[1]), u_seq

    # --- Parallel Computing ---
    rng_keys = random.split(rng_key, n_samples)
    all_costs_and_state, all_seqs = jax.vmap(single_sample_cost)(rng_keys)

    # --- Path Integral Weighting ---
    final_costs = all_costs_and_state[0]
    temperature = config.Temperature
    
    min_cost = jnp.min(final_costs)
    exp_cost = jnp.exp(-temperature * (final_costs - min_cost))
    denom = jnp.sum(exp_cost) + 1e-10
    
    best_U = jnp.sum(exp_cost[..., None, None] * all_seqs, axis=0) / denom

    # --- Warm Start ---
    new_U = jnp.roll(best_U, shift=-1, axis=0)
    new_U = new_U.at[-1].set(new_U[-2])
    
    return best_U[0], new_U, all_seqs