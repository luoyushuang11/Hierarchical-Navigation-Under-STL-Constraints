# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t):
    
    
   
    


    
    
    
    
    
    
  


    
   
    x1 = x[0]
    x2 = x[1]

    
    obs_x_init = jnp.array([4.0, 8.0, 12.0, 16.0])

    obs_y      = jnp.array([1.0, 3.0, 1.0, 3.0])
    target_y      = jnp.array([3.0, 1.0, 3.0, 1.0])

    
    obs_vx     = jnp.array([0.5, 0.5, 0.5, 0.5])
    x_obs_t = obs_x_init + obs_vx * t

 
    b_safe = -1.22 + jnp.square(x1 - x_obs_t) + jnp.square(x2 - obs_y)
    
  
    overtake_offsets = jnp.array([9.99, 19.99, 29.99, 39.99]) # 对应 b6, b7, b8, b9 的常数项
    time_thresholds  = jnp.array([5.0, 10.0, 15.0, 20.0])    # 对应 5s, 10s...
  
    b_overtake_raw = overtake_offsets + 0.45 - 2.0 * t - jnp.sqrt(jnp.square(x1 - (x_obs_t + 2.0)) + jnp.square(x2 - target_y))

    b_overtake = jnp.where(t > time_thresholds, jnp.inf, b_overtake_raw)


    b_lane_raw = 1.95 - jnp.square(x2 - 2.0) # b5
    b_lane = jnp.where(t >= 20.0, jnp.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - jnp.square(x2 - 1.0)

    b_lane_final = jnp.where(t < 20.0, jnp.inf, b_lane_final_raw) # b10

    
    all_neg_b = jnp.concatenate([
        -b_safe, 
        jnp.array([-b_lane]), 
        -b_overtake, 
        jnp.array([-b_lane_final])
    ])
   
    alpha = 500.0 
    

   
    h = -jax_nn.logsumexp(all_neg_b * alpha) / alpha
    


    



    return h

def h_x_numpy(x, t):
    
    
    
    
    
   
    
    
    x1 = x[0]
    x2 = x[1]

    
    obs_x_init = np.array([4.0, 8.0, 12.0, 16.0])
  
    obs_y      = np.array([1.0, 3.0, 1.0, 3.0])
    target_y      = np.array([3.0, 1.0, 3.0, 1.0])

    
   
    obs_vx     = np.array([0.5, 0.5, 0.5, 0.5])
    


    x_obs_t = obs_x_init + obs_vx * t

 
    b_safe = -1.22 + np.square(x1 - x_obs_t) + np.square(x2 - obs_y)
    
   
    overtake_offsets = np.array([9.99, 19.99, 29.99, 39.99])
    time_thresholds  = np.array([5.0, 10.0, 15.0, 20.0])
    
    
    b_overtake_raw = overtake_offsets + 0.45 - 2.0 * t - np.sqrt(np.square(x1 - (x_obs_t + 2.0)) + np.square(x2 - target_y))

 
    b_overtake = np.where(t > time_thresholds, np.inf, b_overtake_raw)


    b_lane_raw = 1.95 - np.square(x2 - 2.0) # b5
    b_lane = np.where(t >= 20.0, np.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - np.square(x2 - 1.0)

   
    b_lane_final = np.where(t < 20.0, np.inf, b_lane_final_raw) # b10

  
    all_neg_b = np.concatenate([
        -b_safe, 
        np.array([-b_lane]), 
        -b_overtake, 
        np.array([-b_lane_final])
    ])

   
    alpha = 500.0 
    

  
    scaled_neg_b = all_neg_b * alpha
    
    m = np.max(scaled_neg_b)

    h = -(np.log(np.sum(np.exp(scaled_neg_b - m))) + m) / alpha
    
    

    
    



    


   
    

    
    return h
    