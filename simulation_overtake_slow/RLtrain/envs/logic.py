import jax
import jax.numpy as jnp
from config import TGPOConfig

@jax.jit
def update_augmented_state(aug_state, next_phys_state, time_vars, subgoals_array, obstacles_params):
    
    tau = aug_state[TGPOConfig.IDX_TAU]
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    r = aug_state[TGPOConfig.IDX_R].astype(jnp.int32)
    chi = aug_state[TGPOConfig.IDX_CHI]
    
   
    tau_next = tau + 1.0
    current_time = tau_next * TGPOConfig.DT 
    
   
    # Pos_x(t) = Start_x + v_x * t
    obs_x_curr = obstacles_params[:, 0] + obstacles_params[:, 2] * current_time
    obs_y_curr = obstacles_params[:, 1]
    obs_radii = obstacles_params[:, 3]
  
    current_obs_pos = jnp.stack([obs_x_curr, obs_y_curr], axis=-1)

   
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    
    curr_goal = subgoals_array[safe_p]
    offset_x = curr_goal[0]
    goal_y = curr_goal[1]
    goal_radius = curr_goal[2]
    t_idx = curr_goal[3].astype(jnp.int32)
    ref_obs_idx = curr_goal[4].astype(jnp.int32)
    
    
    target_obs_x = obs_x_curr[ref_obs_idx]
    goal_x_abs = target_obs_x + offset_x
    
    goal_pos_abs = jnp.array([goal_x_abs, goal_y])
    
  
    t_target = jnp.round(time_vars[t_idx])
    
    
    dist_goal = jnp.linalg.norm(next_phys_state[:2] - goal_pos_abs)
    mu_satisfied = dist_goal <= goal_radius
    
    
    cond_reset = (r == 2)
    
    time_match = (tau_next == t_target)
    
    
    cond_finish = (r != 2) & time_match & mu_satisfied
    
    r_next = jnp.where(cond_reset, 0.0,
                jnp.where(cond_finish, 2.0, r))
    
    
    p_increment = (r_next == 2.0).astype(jnp.float32)
    p_next = jnp.minimum(p + p_increment, max_p)
    p_prev_next = p
    
    
    pos = next_phys_state[:2]
    # (1, 2) - (N_obs, 2)
    diff = pos - current_obs_pos 
    dists_obs = jnp.linalg.norm(diff, axis=-1) 
    
   
    margins = dists_obs - 1.1

    
    y_curr = pos[1]
    in_bounds = (y_curr >= 0.6) & (y_curr <= 3.4)

    min_margin = jnp.min(margins)
    
    is_safe = (min_margin >= 0.0) & in_bounds
    is_safe = is_safe.astype(jnp.float32)
    
    chi_next = chi * is_safe
    
    return jnp.concatenate([
        jnp.array([tau_next, p_prev_next, p_next, r_next, chi_next])
    ])