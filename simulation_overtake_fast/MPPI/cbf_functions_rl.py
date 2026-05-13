# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t, time_vars, reach_flags):
    
    
   
    


    
    
    
    
    
    
    
    
    


    
    # (JAX JIT version) Vectorized Efficient Computation Edition CBF

    # Use expand_dims to add a dimension at the end (..., 1)
    # So that when operating with an array of obstacles with shape (4,), JAX will automatically broadcast as (..., 4)
    x1 = jnp.expand_dims(x[..., 0], axis=-1)
    x2 = jnp.expand_dims(x[..., 1], axis=-1)

    # --- 1. Dynamic Obstacle Prediction (All Combined to Array Parallel Calculations) ---
    obs_x_init = jnp.array([4.0, 8.0, 12.0, 16.0])
    obs_y      = jnp.array([1.0, 3.0, 1.0, 3.0])
    target_y   = jnp.array([3.0, 1.0, 3.0, 1.0])
    obs_vx     = jnp.array([0.4, 0.6, 0.8, 1.0])
    
    # x_obs_t shape: (4,)
    x_obs_t = obs_x_init + obs_vx * t

    # --- 2. Dynamic Safety Obstacle Avoidance Restraints (b_safe) ---
    # Using the broadcast mechanism, the safety margin from the trolley to 4 obstacles is calculated at one time
    # b_safe shape: (..., 4)
    b_safe = -1.22 + jnp.square(x1 - x_obs_t) + jnp.square(x2 - obs_y)

    # --- 3. Dynamic Overtake Target Funnel (Dual Insurance Latch Mechanism Vectorized) ---
    # Parse time variables: t_rl (Predicted Soft Deadline) and t_hard (Absolute Hard Deadline)
    t_rl = jnp.round(time_vars[:4]) * 0.2
    t_hard = jnp.array([5.0, 10.0, 20.0, 30.0])

    # dist shape： （...， 4）
    dist = jnp.sqrt(jnp.square(x1 - (x_obs_t + 2.0)) + jnp.square(x2 - target_y))

    # Calculate the Hard and RL funnel values for 4 goals at once
    b_hard_val = -1.5 * t + t_hard * 1.5 + 0.44 - dist
    b_rl_val   = -2.0 * t + t_rl * 2.0 + 0.44 - dist
    
    # The latch is updated in parallel
    is_t_match = jnp.abs(t - t_rl) < 1e-4
    # flag update: If the history is already True, or if the frame meets the match criteria
    new_reach_flags = reach_flags | (is_t_match & (b_rl_val >= 0))
    
    # Discard the constraint that timed out (assigned to positive infinity)
    b_rl   = jnp.where(t > t_rl, jnp.inf, b_rl_val)
    b_hard = jnp.where(t > t_hard, jnp.inf, jnp.where(new_reach_flags, jnp.inf, b_hard_val))

    # --- 4. Road boundary constraints ---
    # b_lane shape: (..., )
    b_lane = jnp.where(t >= 30.0, jnp.inf, 1.95 - jnp.square(x[..., 1] - 2.0))
    b_lane_final = jnp.where(t < 30.0, jnp.inf, 0.15 - jnp.square(x[..., 1] - 1.0))

    # In order to splice with the previous (..., 4), the boundary is also expanded by a dimension -> (..., 1)
    b_lane = jnp.expand_dims(b_lane, axis=-1)
    b_lane_final = jnp.expand_dims(b_lane_final, axis=-1)

    # --- 5. Merge to take the smallest ---
    # 4 (safe) + 4 (rl) + 4 (hard) + 1 (lane) + 1 (lane_final) = 14
    all_b = jnp.concatenate([b_safe, b_rl, b_hard, b_lane, b_lane_final], axis=-1)

    # 取全局的最小值
    h = jnp.min(all_b, axis=-1)
    


   
    


    



    return h, new_reach_flags
