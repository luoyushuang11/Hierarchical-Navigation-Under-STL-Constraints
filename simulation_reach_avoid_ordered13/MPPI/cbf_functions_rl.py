# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t, time_vars, reach_flags):
    
    
    
  
    flag1, flag2, flag3 = reach_flags[0], reach_flags[1], reach_flags[2]
    x1 = x[..., 0]
    x2 = x[..., 1]
    pos = x[..., :2] 
    safe_dist_sq = 1.0 # 0.75 +0.2 +0.01

   
    t_rl_1 = jnp.round(time_vars[0]) * 0.2
    b1_hard_val = -2.0 * t + 55.99 + 1.0 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 + 2.5, 2))
    b1_rl_val = -2.0 * t + t_rl_1 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 + 2.5, 2))
    

    is_t1_match = jnp.abs(t - t_rl_1) < 1e-4
    flag1_new = flag1 | (is_t1_match & (b1_rl_val >= 0))

    b1_rl = jnp.where(t > t_rl_1, jnp.inf, b1_rl_val)
    b1_hard = jnp.where(t > 28.00, jnp.inf,
                jnp.where(flag1_new, jnp.inf, b1_hard_val))




    t_rl_2 = jnp.round(time_vars[1]) * 0.2
    b12_hard_val = -2.0 * t + 87.99 + 1.0 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 2.5, 2))
    b12_rl_val = -2.0 * t + t_rl_2 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 2.5, 2))
    
    is_t2_match = jnp.abs(t - t_rl_2) < 1e-4
    flag2_new = flag2 | (is_t2_match & (b12_rl_val >= 0))

    b12_rl = jnp.where(t > t_rl_2, jnp.inf, b12_rl_val)
    b12_hard = jnp.where(t > 44.00, jnp.inf,
                jnp.where(flag2_new, jnp.inf, b12_hard_val))
    
    
    
  
    t_rl_3 = jnp.round(time_vars[2]) * 0.2
    b13_hard_val = -2.0 * t + 119.99 + 1.0 - jnp.sqrt(jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 5.0, 2))
    b13_rl_val = -2.0 * t + t_rl_3 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 5.0, 2))
    
    is_t3_match = jnp.abs(t - t_rl_3) < 1e-4
    flag3_new = flag3 | (is_t3_match & (b13_rl_val >= 0))

    b13_rl = jnp.where(t > t_rl_3, jnp.inf, b13_rl_val)
  
    b13_hard = jnp.where(t > 60.00, jnp.inf,
                jnp.where(flag3_new, jnp.inf, b13_hard_val))

    obs_centers = jnp.array([
        [-2.5,  0.0], # b2
        [ 6.0, -2.5], # b3
        [ 2.5,  0.0], # b4
        [ 2.5,  5.0], # b5
        [-2.5, -2.5], # b6
        [ 6.0,  2.5], # b7
        [ 6.0,  5.0], # b8
        [ 6.0,  0.0], # b9
        [-2.5,  2.5], # b10
        [ 0.0, -2.5], # b14
        [ 0.0,  0.0], # b15
        [ 0.0,  2.5], # b16
        [ 0.0,  5.0], # b17
    ])
    dists_sq = jnp.sum(jnp.power(pos - obs_centers, 2), axis=-1)
    b_static_obs = dists_sq - safe_dist_sq

    b11 = 99.10 - (jnp.power(x1, 2) + jnp.power(x2, 2))

    all_b = jnp.concatenate([
        jnp.atleast_1d(b1_rl), jnp.atleast_1d(b1_hard),      
        b_static_obs[..., 0:9], jnp.atleast_1d(b11),          
        jnp.atleast_1d(b12_rl), jnp.atleast_1d(b12_hard),     
        jnp.atleast_1d(b13_rl), jnp.atleast_1d(b13_hard),     
        b_static_obs[..., 9:]         
    ], axis=-1)

    

    h = jnp.min(all_b, axis=-1)
    

    new_reach_flags = jnp.array([flag1_new, flag2_new, flag3_new])
    return h, new_reach_flags
    
    


    
    
    
