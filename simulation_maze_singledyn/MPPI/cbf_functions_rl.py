# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t, time_vars, reach_flags):
    


    flag1, flag2 = reach_flags[0], reach_flags[1]

    
    # --- Robot parameters ---
    ROBOT_RADIUS = 0.2
    
    # --- 8 static rectangular obstacle parameters ---
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

    
   
    #Subscene 1balls
    dyn_x_fixed = jnp.array([6.0])
    dyn_phases = jnp.array([0.0])
    DYN_RADIUS = 0.5
    
    

    
    # --- Workspace boundary parameter [0, 12] x [0, 10] ---
    ws_center  = jnp.array([6.0, 5.0])
    ws_extents = jnp.array([6.0, 5.0])

    # --- Extracting bot state ---
    pos = x[:2] 

    # ==========================================
    # Part 1: 8 Static Rectangular Obstacles (Standard SDF)
    # ==========================================
    # 计算 SDF： d = |p - c|- e
    p_rel_static = jnp.abs(pos - obs_static_centers)
    d_static = p_rel_static - obs_static_extents
    
    # Outer Distance (Vector Modulus Length) + Inner Distance (Maximum Component)
    dist_out = jnp.linalg.norm(jnp.maximum(d_static, 0.0), axis=1)
    dist_in  = jnp.minimum(jnp.max(d_static, axis=1), 0.0)
    
    # b > 0 safety, b < 0 collision
    b_static = (dist_out + dist_in) - ROBOT_RADIUS-0.01

    # ==========================================
    # Part 2: 2 Dynamic Circular Obstacles (Triangle Reciprocating Motion)
    # ==========================================
    # Calculate the virtual distance of the current moment d = v*t + phase
    dist_travel = 0.4 * t + dyn_phases
    
    # Map to the [0.5, 9.5] interval
    y_offsets = 9.0 - jnp.abs((dist_travel % 18.0) - 9.0)
    obs_dyn_y = 0.5 + y_offsets
    
    # Combine coordinates and calculate distances
    obs_dyn_pos = jnp.stack([dyn_x_fixed, obs_dyn_y], axis=1)
    dist_dyn = jnp.linalg.norm(pos - obs_dyn_pos, axis=1)
    
    # Subtract (Robot radius + Obstacle radius)
    b_dynamic = dist_dyn - (ROBOT_RADIUS + DYN_RADIUS)#-0.1

    # ==========================================
    # Part 3: Workspace Boundaries (Keep-In Constraints)
    # ==========================================
    # Calculate SDF relative to large boxes
    p_rel_ws = jnp.abs(pos - ws_center)
    d_ws = p_rel_ws - ws_extents
    sdf_ws = jnp.linalg.norm(jnp.maximum(d_ws, 0.0)) + jnp.minimum(jnp.max(d_ws), 0.0)
    
    # Reverse: positive inside (safety), negative outside (dangerous)
    b_boundary = jnp.array([-sdf_ws - ROBOT_RADIUS-0.01])


    # ==========================================
    # Part 4: Target area 
    # ==========================================
    # Objective 1: [1.0, 3.0]
    # Calculate the current Euclidean distance from the target
    t_rl_1 = jnp.round(time_vars[0]) * 0.2
    dist_goal1 = jnp.linalg.norm(pos - jnp.array([1.0, 3.0]))
    b1_hard_val = -1.5 * t + 44.99 + 0.6 - dist_goal1
    b1_rl_val = -1.0 * t + t_rl_1 * 1.0 + 0.59 - dist_goal1
    # State update: if history is already True, or this frame just triggers, update to True
    is_t1_match = jnp.abs(t - t_rl_1) < 1e-4
    flag1_new = flag1 | (is_t1_match & (b1_rl_val >= 0))

    b1_rl = jnp.where(t > t_rl_1, jnp.inf, b1_rl_val)
    b1_hard = jnp.where(t > 30.00, jnp.inf,
                jnp.where(flag1_new, jnp.inf, b1_hard_val))
    
    # Objective 2: [1.0, 9.0]
    t_rl_2 = jnp.round(time_vars[1]) * 0.2
    dist_goal2 = jnp.linalg.norm(pos - jnp.array([2.25, 9.0]))
    b2_hard_val = 100#-1.5 * t + 59.99 + 0.6 - dist_goal2
    #b2_rl_val = -2.0 * t + t_rl_2 * 2.0 + 0.59 - dist_goal2 #v3
    b2_rl_val = -1.0 * t + t_rl_2 * 1.0 + 0.59 - dist_goal2 #v2
    # State update: if history is already True, or this frame just triggers, update to True
    is_t2_match = jnp.abs(t - t_rl_2) < 1e-4
    flag2_new = flag2 | (is_t2_match & (b2_rl_val >= 0))

    b2_rl = jnp.where(t > t_rl_2, jnp.inf, b2_rl_val)
    b2_hard = jnp.where(t > 40.00, jnp.inf,
                jnp.where(flag2_new, jnp.inf, b2_hard_val))

    # ==========================================
    # Part 5: Aggregation (Strict Minimum Calculation)
    # ==========================================
    # Use atleast_1d to ensure that all scalars or arrays are stitched in the last dimension (axis=-1).
    all_b = jnp.concatenate([
        jnp.atleast_1d(b_static),       # (8,)
        jnp.atleast_1d(b_dynamic),      # (2,)
        jnp.atleast_1d(b_boundary),     # (1,)
        jnp.atleast_1d(b1_rl),          # (1,)
        jnp.atleast_1d(b1_hard),        # (1,)
        jnp.atleast_1d(b2_rl),          # (1,)
        jnp.atleast_1d(b2_hard)         # (1,)
    ], axis=-1)
    
    h = jnp.min(all_b, axis=-1)
    
    new_reach_flags = jnp.array([flag1_new, flag2_new])

    return h, new_reach_flags
