import jax
import jax.numpy as jnp
from config import TGPOConfig

@jax.jit
def update_augmented_state(aug_state, next_phys_state, time_vars, subgoals_array, obstacles_array):
    
    tau = aug_state[TGPOConfig.IDX_TAU]
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    r = aug_state[TGPOConfig.IDX_R].astype(jnp.int32)
    chi = aug_state[TGPOConfig.IDX_CHI]
    
   
    tau_next = tau + 1.0
    
    
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    
    curr_goal = subgoals_array[safe_p]
    goal_pos = curr_goal[:2]
    goal_radius = curr_goal[2]
    
   
    t_idx = curr_goal[3].astype(jnp.int32)
    
    
    t_target = jnp.round(time_vars[t_idx])
    
  
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    mu_satisfied = dist <= goal_radius
    
   
    
    cond_reset = (r == 2)
    
    time_match = (tau_next == t_target)
    
   
    cond_finish = (r != 2) & time_match & mu_satisfied
    
 
    r_next = jnp.where(cond_reset, 0.0,
                jnp.where(cond_finish, 2.0, r))

    p_increment = (r_next == 2.0).astype(jnp.float32)
    p_next = p + p_increment
    p_next = jnp.minimum(p_next, max_p)
    

    p_prev_next = p
    
   
    pos = next_phys_state[:2]
    diff = pos - obstacles_array[:, :2] 
    dists_obs = jnp.linalg.norm(diff, axis=-1) 
    obs_margins = dists_obs - obstacles_array[:, 2]-0.2
    
    workspace_radius = 10.0 
    dist_to_origin = jnp.linalg.norm(pos) 
   
    ws_margin = workspace_radius - 0.2 - dist_to_origin
    
   
    min_obs_margin = jnp.min(obs_margins)
    
  
    total_min_margin = jnp.minimum(min_obs_margin, ws_margin)
    
   
    is_safe = (total_min_margin > 0.0).astype(jnp.float32)
    
  
    chi_next = chi * is_safe
    
    return jnp.concatenate([
        jnp.array([tau_next, p_prev_next, p_next, r_next, chi_next])
    ])