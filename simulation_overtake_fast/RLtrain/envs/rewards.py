import jax.numpy as jnp
from config import TGPOConfig


def compute_dense_reward(aug_state, next_aug_state, next_phys_state, subgoals_array, obstacles_params):
    
    # --- 1. Legacy logic (completely untouched) ---
    tau = aug_state[TGPOConfig.IDX_TAU]
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    p_next = next_aug_state[TGPOConfig.IDX_P]
    chi_next = next_aug_state[TGPOConfig.IDX_CHI]
    
    max_p = subgoals_array.shape[0] # 4
    safe_p = jnp.minimum(p, max_p - 1)
    
    # [Coordinate Calculation: The Golden Rule - Hold Motion]
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
    
    # --- 2. Status Judgment (New) ---
    # When p == 4, indicate that the first 4 tasks (0-3) are all completed and move on to Phase 5 (cruising)
    is_cruise_phase = (p == max_p)
    
    # --- 3. Distance Bonus (Modified) ---
    dist = jnp.linalg.norm(next_phys_state[:2] - real_goal_pos)
    
    # [Logic] If you enter the cruising phase, Goal 3 is already behind you, cut off the distance bonus to prevent it from wanting to turn around
    # If you don't enter the cruise, you will be rewarded normally
    r_dist = (-dist * TGPOConfig.LAMBDA_DIST) * (1.0 - is_cruise_phase)
    
    # --- 4. Cruise Rewards (New) ---
    # Takes effect only when is_cruise_phase == 1
    
    # Unpack physical quantities
    y_curr = next_phys_state[1]
    theta = next_phys_state[2]
    v = next_phys_state[3]
    v_x = v * jnp.cos(theta)
    v_y = v * jnp.sin(theta)
    
    # A. Speed Hold (1.0 m/s)
    target_vx = 1.5
    diff_sq = (v_x - target_vx) ** 2
    r_speed = jnp.exp(-diff_sq / 0.5) * TGPOConfig.LAMBDA_VEL*10
    
    # B. Centering
    # Directs back to Lane 1 (y=1.0).
    # Note: logic.py allows 0.6~3.4, so the car will not die, but if you don't return to 1.0, you will never get this point
    r_center = - ((y_curr - 1.0) ** 2) * 2.0
    
    # C. Lateral damping
    r_damp = - (v_y ** 2) * 0.1
    
    # Combo: Given only during the cruising phase and when it is safe to do so
    r_cruise = is_cruise_phase * (r_speed + r_center + r_damp) * (chi_next > 0.5)
    
    # --- 5. Other original rewards (remain unchanged) ---
    r_prog = (p_next - p) * TGPOConfig.LAMBDA_PROG
    all_done = (p_next == max_p)
    r_succ = (all_done & (chi_next == 1.0)) * TGPOConfig.LAMBDA_SUCC
    is_violation = (chi_next < 0.5)
    r_inv = is_violation * TGPOConfig.LAMBDA_INV
    
    # --- 6. Return to sum ---
    return r_dist + r_prog + r_succ + r_inv + r_cruise