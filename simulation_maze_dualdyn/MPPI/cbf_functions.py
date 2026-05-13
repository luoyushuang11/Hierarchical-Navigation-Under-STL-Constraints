# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t):
    




    
    ROBOT_RADIUS = 0.2
    
   
    # Center [cx, cy]
    obs_static_centers = jnp.array([
        [2.0, 1.0], [1.5, 4.25], [2.0, 7.0], [4.75, 6.0],
        [8.25, 4.0], [11.25, 3.5], [9.25, 5.5], [11.5, 9.0]
    ])
    # Half-Extents [w/2, h/2][0.75, 0.5], 
    obs_static_extents = jnp.array([
        [1.0, 1.0], [1.5, 0.25], [1.0, 1.0], [0.25, 4.0],
        [0.25, 4.0], [0.75, 0.5],[0.75, 0.5], [0.5, 1.0]
    ])

    """
    dyn_x_fixed = jnp.array([5.8, 7.3])
    # Span = 9.0, Cycle = 18.0
    dyn_phases = jnp.array([0.0, 9.0]) # 0.0=从底向上, 9.0=从顶向下
    DYN_RADIUS = 0.5
    """
    #1balls
    dyn_x_fixed = jnp.array([600.0])
    dyn_phases = jnp.array([0.0])
    DYN_RADIUS = 0.5
    
    
    

  
    

    
    # ---  [0, 12] x [0, 10] ---
    ws_center  = jnp.array([6.0, 5.0])
    ws_extents = jnp.array([6.0, 5.0])

   
    pos = x[:2] 

    # ==========================================
    # Part 1: 8 (Standard SDF)
    # ==========================================
    #  SDF: d = |p - c| - e
    p_rel_static = jnp.abs(pos - obs_static_centers)
    d_static = p_rel_static - obs_static_extents
    
    # 
    dist_out = jnp.linalg.norm(jnp.maximum(d_static, 0.0), axis=1)
    dist_in  = jnp.minimum(jnp.max(d_static, axis=1), 0.0)
    
    b_static = (dist_out + dist_in) - ROBOT_RADIUS-0.01

    
    dist_travel = 0.4 * t + dyn_phases
    #dist_travel = 0.8 * t + dyn_phases
    
   
    y_offsets = 9.0 - jnp.abs((dist_travel % 18.0) - 9.0)
    obs_dyn_y = 0.5 + y_offsets
 
    obs_dyn_pos = jnp.stack([dyn_x_fixed, obs_dyn_y], axis=1)
    dist_dyn = jnp.linalg.norm(pos - obs_dyn_pos, axis=1)
    

    b_dynamic = dist_dyn - (ROBOT_RADIUS + DYN_RADIUS)-0.01

   
    p_rel_ws = jnp.abs(pos - ws_center)
    d_ws = p_rel_ws - ws_extents
    sdf_ws = jnp.linalg.norm(jnp.maximum(d_ws, 0.0)) + jnp.minimum(jnp.max(d_ws), 0.0)
    
  
    b_boundary = jnp.array([-sdf_ws - ROBOT_RADIUS-0.01])



    dist_goal1 = jnp.linalg.norm(pos - jnp.array([1.0, 3.0]))
    b_goal1_raw = -2.0 * t + 59.99 + 0.6 - dist_goal1
   
    b_goal1 = jnp.where(t > 30.00, jnp.inf, b_goal1_raw)
    
   
    dist_goal2 = jnp.linalg.norm(pos - jnp.array([2.25, 9.0]))
   
    b_goal2 = -2.0 * t + 79.99 + 0.6 - dist_goal2

    
    b_goals = jnp.array([b_goal1, b_goal2])

    
    #all_b = jnp.concatenate([b_static, b_dynamic, b_boundary, b_goals])
    all_b = jnp.concatenate([b_static, b_boundary, b_goals])
    
    
    # min(x) ≈ -log(sum(exp(-alpha * x))) / alpha
    alpha = 500.0
    neg_b = -all_b
    h = -jax_nn.logsumexp(neg_b * alpha) / alpha
    return h

def h_x_numpy(x, t):
    
    
    


    

    pos = x[:2] # shape (2,)

   
    ROBOT_RADIUS = 0.2
    SAFETY_MARGIN = 0.01 

    # Center [cx, cy] [9.25, 5.5],
    obs_static_centers = np.array([
        [2.0, 1.0], [1.5, 4.25], [2.0, 7.0], [4.75, 6.0],
        [8.25, 4.0], [11.25, 3.5], [9.25, 5.5], [11.5, 9.0]
    ])
    # Half-Extents [w/2, h/2]
    obs_static_extents = np.array([
        [1.0, 1.0], [1.5, 0.25], [1.0, 1.0], [0.25, 4.0],
        [0.25, 4.0], [0.75, 0.5],[0.75, 0.5], [0.5, 1.0]
    ])
    

    """
    dyn_x_fixed = np.array([5.8, 7.3])
    dyn_phases  = np.array([0.0, 9.0])
    DYN_RADIUS  = 0.5
    
    """
    
    dyn_x_fixed = np.array([600.0])
    dyn_phases  = np.array([0.0])
    DYN_RADIUS  = 0.5
    
  
    ws_center  = np.array([6.0, 5.0])
    ws_extents = np.array([6.0, 5.0])


  
    p_rel_static = np.abs(pos - obs_static_centers)
    d_static = p_rel_static - obs_static_extents
    
    
    dist_out = np.linalg.norm(np.maximum(d_static, 0.0), axis=1)
    

    dist_in  = np.minimum(np.max(d_static, axis=1), 0.0)
    
   
    b_static = (dist_out + dist_in) - ROBOT_RADIUS - 0.01


  
    dist_travel = 0.4 * t + dyn_phases
    #dist_travel = 0.8 * t + dyn_phases
    
    
    y_offsets = 9.0 - np.abs((dist_travel % 18.0) - 9.0)
    obs_dyn_y = 0.5 + y_offsets
    
  
    obs_dyn_pos = np.stack([dyn_x_fixed, obs_dyn_y], axis=1)
    
   
    dist_dyn = np.linalg.norm(pos - obs_dyn_pos, axis=1)
    
  
    b_dynamic = dist_dyn - (ROBOT_RADIUS + DYN_RADIUS) - 0.01


    
    p_rel_ws = np.abs(pos - ws_center)
    d_ws = p_rel_ws - ws_extents
    
    sdf_ws = np.linalg.norm(np.maximum(d_ws, 0.0)) + np.minimum(np.max(d_ws), 0.0)
    
    
    b_boundary = np.array([-sdf_ws - ROBOT_RADIUS - SAFETY_MARGIN])


 
    dist_goal1 = np.linalg.norm(pos - np.array([1.0, 3.0]))
    b_goal1_raw = -2.0 * t + 59.99 + 0.6 - dist_goal1

  
    b_goal1 = np.where(t > 30.00, np.inf, b_goal1_raw)

  
    dist_goal2 = np.linalg.norm(pos - np.array([2.25, 9.0]))
    b_goal2 = -2.0 * t + 79.99 + 0.6 - dist_goal2

    b_goals = np.array([b_goal1, b_goal2])


    #all_b = np.concatenate([b_static, b_dynamic, b_boundary, b_goals])
    all_b = np.concatenate([b_static, b_boundary, b_goals])
   
    neg_b = -all_b

    alpha = 500.0

   
    scaled_neg_b = neg_b * alpha
    
   
    m = np.max(scaled_neg_b)
    
    
    lse_value = m + np.log(np.sum(np.exp(scaled_neg_b - m)))
    
   
    h = -lse_value / alpha
    

    
    return h
    