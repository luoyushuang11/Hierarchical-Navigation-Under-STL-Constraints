import jax.numpy as jnp
from config import TGPOConfig


def compute_dense_reward(aug_state, next_aug_state, next_phys_state, subgoals_array, obstacles_params):
    

    tau = aug_state[TGPOConfig.IDX_TAU]
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    p_next = next_aug_state[TGPOConfig.IDX_P]
    chi_next = next_aug_state[TGPOConfig.IDX_CHI]
    
    max_p = subgoals_array.shape[0] # 4
    safe_p = jnp.minimum(p, max_p - 1)
    
    
    current_time = (tau + 1.0) * TGPOConfig.DT
    curr_goal = subgoals_array[safe_p]
    offset_x = curr_goal[0]
    goal_y = curr_goal[1]
    ref_obs_idx = curr_goal[4].astype(jnp.int32)
    
    obs_start_x = obstacles_params[ref_obs_idx, 0]
    obs_vx = obstacles_params[ref_obs_idx, 2]
    obs_curr_x = obs_start_x + obs_vx * current_time 
    
    real_goal_x = obs_curr_x + offset_x
    real_goal_pos = jnp.array([real_goal_x, goal_y])
    
  
    is_cruise_phase = (p == max_p)
    
   
    dist = jnp.linalg.norm(next_phys_state[:2] - real_goal_pos)
    
    
    r_dist = (-dist * TGPOConfig.LAMBDA_DIST) * (1.0 - is_cruise_phase)
    
    
    
    
    y_curr = next_phys_state[1]
    theta = next_phys_state[2]
    v = next_phys_state[3]
    v_x = v * jnp.cos(theta)
    v_y = v * jnp.sin(theta)
    
   
    target_vx = 1.5
    diff_sq = (v_x - target_vx) ** 2
    r_speed = jnp.exp(-diff_sq / 0.5) * TGPOConfig.LAMBDA_VEL*10
    
    
    r_center = - ((y_curr - 1.0) ** 2) * 2.0
    
    
    r_damp = - (v_y ** 2) * 0.1
    
   
    r_cruise = is_cruise_phase * (r_speed + r_center + r_damp) * (chi_next > 0.5)
    

    r_prog = (p_next - p) * TGPOConfig.LAMBDA_PROG
    all_done = (p_next == max_p)
    r_succ = (all_done & (chi_next == 1.0)) * TGPOConfig.LAMBDA_SUCC
    is_violation = (chi_next < 0.5)
    r_inv = is_violation * TGPOConfig.LAMBDA_INV
    
  
    return r_dist + r_prog + r_succ + r_inv + r_cruise