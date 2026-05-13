# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t, time_vars, reach_flags):
    
    
   

  



    x1 = jnp.expand_dims(x[..., 0], axis=-1)
    x2 = jnp.expand_dims(x[..., 1], axis=-1)

    
    obs_x_init = jnp.array([4.0, 8.0, 12.0, 16.0])
    obs_y      = jnp.array([1.0, 3.0, 1.0, 3.0])
    target_y   = jnp.array([3.0, 1.0, 3.0, 1.0])
    obs_vx     = jnp.array([0.5, 0.5, 0.5, 0.5])
    
    # x_obs_t shape: (4,)
    x_obs_t = obs_x_init + obs_vx * t

    
    # b_safe shape: (..., 4)
    b_safe = -1.22 + jnp.square(x1 - x_obs_t) + jnp.square(x2 - obs_y)

 
    t_rl = jnp.round(time_vars[:4]) * 0.2
    t_hard = jnp.array([5.0, 10.0, 15.0, 20.0])

   
    dist = jnp.sqrt(jnp.square(x1 - (x_obs_t + 2.0)) + jnp.square(x2 - target_y))

   
    b_hard_val = -1.6 * t + t_hard * 1.6 + 0.44 - dist
    b_rl_val   = -2.0 * t + t_rl * 2.0 + 0.44 - dist
    
   
    is_t_match = jnp.abs(t - t_rl) < 1e-4
  
    new_reach_flags = reach_flags | (is_t_match & (b_rl_val >= 0))
    

    b_rl   = jnp.where(t > t_rl, jnp.inf, b_rl_val)
    b_hard = jnp.where(t > t_hard, jnp.inf, jnp.where(new_reach_flags, jnp.inf, b_hard_val))

   
    b_lane = jnp.where(t >= 20.0, jnp.inf, 1.95 - jnp.square(x[..., 1] - 2.0))
    b_lane_final = jnp.where(t < 20.0, jnp.inf, 0.15 - jnp.square(x[..., 1] - 1.0))

  
    b_lane = jnp.expand_dims(b_lane, axis=-1)
    b_lane_final = jnp.expand_dims(b_lane_final, axis=-1)

 
    all_b = jnp.concatenate([b_safe, b_rl, b_hard, b_lane, b_lane_final], axis=-1)


    h = jnp.min(all_b, axis=-1)
    


    



    return h, new_reach_flags
