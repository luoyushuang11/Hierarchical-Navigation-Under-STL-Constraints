# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t):
    
    
   

    
    
   
    x1 = x[..., 0]
    x2 = x[..., 1]
    pos = x[..., :2] 
    safe_dist_sq = 0.91 # 0.75 +0.2 +0.01
    b1_val = -4.0 * t + 111.7 + 0.95 - (jnp.power(x1 - 2.5, 2) + jnp.power(x2 + 2.5, 2))
    b1 = jnp.where(t > 28.00, jnp.inf, b1_val)
    b12_val = -4.0 * t + 175.7 + 0.95 - (jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 2.5, 2))
    b12 = jnp.where(t > 44.00, jnp.inf, b12_val)
    b13 = -4.0 * t + 239.7 + 0.95 - (jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 5.0, 2))

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
        jnp.atleast_1d(b1),           # b1
        b_static_obs[..., 0:9],       # b2 - b10
        jnp.atleast_1d(b11),          # b11
        jnp.atleast_1d(b12),          # b12
        jnp.atleast_1d(b13),          # b13
        b_static_obs[..., 9:]         # b14 - b17
    ], axis=-1)

    neg_b = -all_b
    alpha = 100.0
    h = -jax_nn.logsumexp(neg_b * alpha) / alpha
    
    


    
    
    
    
    
    
    

    return h

def h_x_numpy(x, t):
    
    
    
   
    x1 = x[..., 0]
    x2 = x[..., 1]
    
    
    pos = x[..., :2] 

  
        
    safe_dist_sq = 0.91

    
    b1_val = -4.0 * t + 111.7 + 0.95 - (np.power(x1 - 2.5, 2) + np.power(x2 + 2.5, 2))
    b1 = np.where(t > 28.00, np.inf, b1_val)
    
    b12_val = -4.0 * t + 175.7 + 0.95 - (np.power(x1 - 2.5, 2) + np.power(x2 - 2.5, 2))
    b12 = np.where(t > 44.00, np.inf, b12_val)
    b13 = -4.0 * t + 239.7 + 0.95 - (np.power(x1 + 2.5, 2) + np.power(x2 - 5.0, 2))

    
    obs_centers = np.array([
        [-2.5,  0.0], [ 6.0, -2.5], [ 2.5,  0.0], [ 2.5,  5.0],
        [-2.5, -2.5], [ 6.0,  2.5], [ 6.0,  5.0], [ 6.0,  0.0],
        [-2.5,  2.5], [ 0.0, -2.5], [ 0.0,  0.0], [ 0.0,  2.5],
        [ 0.0,  5.0],
    ])
    
  
    # (..., 2) - (13, 2) -> (..., 13, 2)
    dists_sq = np.sum(np.power(pos - obs_centers, 2), axis=-1)
    b_static_obs = dists_sq - safe_dist_sq

   
    b11 = 99.10 - (np.power(x1, 2) + np.power(x2, 2))

    all_b = np.concatenate([
        np.array([b1]),
        b_static_obs[0:9],
        np.array([b11]),
        np.array([b12]),
        np.array([b13]),
        b_static_obs[9:]
    ])

    neg_b = -all_b
    alpha = 100.0
    scaled_neg_b = neg_b * alpha
    m = np.max(scaled_neg_b)
    h = -(np.log(np.sum(np.exp(scaled_neg_b - m))) + m) / alpha
    
    
    


    
    return h
    