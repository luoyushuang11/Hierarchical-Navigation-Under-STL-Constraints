import jax
import jax.numpy as jnp
from config import TGPOConfig

@jax.jit
def update_augmented_state(aug_state, next_phys_state, time_vars, subgoals_array, obstacles_array):
    """
    
    
    Features:
    - Time constraints are point-wise, not intervals.
    - Require the tau to be exactly equal to the t_target sampled in the time_vars.
    - Use this logic during training to force Policy to learn precise time control.
    
    Args:
        aug_state: Current Enhanced State (9 dimensions)
        next_phys_state: Physical state of the next moment (4D)
        time_vars: Current assigned time variable (N_vars,) me variables (N_vars,)
        subgoals_array: Goal matrix
        obstacles_array: Obstacle matrix
    """
  
    # tau, p_prev, p, r, chi
    tau = aug_state[TGPOConfig.IDX_TAU]
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    r = aug_state[TGPOConfig.IDX_R].astype(jnp.int32)
    chi = aug_state[TGPOConfig.IDX_CHI]
    
    # 1. Update global time
    tau_next = tau + 1.0
    
    # 2. Get information about the current sub-target
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    
    curr_goal = subgoals_array[safe_p]
    goal_pos = curr_goal[:2]
    goal_radius = curr_goal[2]
    
    # Get Time Parameter Index (time_idx)
    t_idx = curr_goal[3].astype(jnp.int32)
    
    # [Core logic] to obtain precise target time points
    # Note: RL actions are continuous, time_vars may be floating-point numbers,
    # However, logical decisions are made on discrete time steps, so rounds must be rounded.
    t_target = jnp.round(time_vars[t_idx])
    
    # 3. Check the atomic proposition mu (if it is in the region)
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    mu_satisfied = dist <= goal_radius
    
    # 4.r (Strict)
    
    cond_reset = (r == 2)
    
    # [Strict Time Constraint]
    # tau_next Must be exactly equal to the target point in time sampled
    time_match = (tau_next == t_target)
    
    # Completion Conditions: (Not Completed r!=2) & (Time Up) & (Space Up)
    cond_finish = (r != 2) & time_match & mu_satisfied
    
    # State transition: 0 -> 2 (done directly, simplifying the intermediate state of r=1 because it is a point constraint)
    r_next = jnp.where(cond_reset, 0.0,
                jnp.where(cond_finish, 2.0, r))
    
    # 5. Update progress p
    # If r becomes 2, the current goal is complete, p plus 1
    p_increment = (r_next == 2.0).astype(jnp.float32)
    p_next = p + p_increment
    p_next = jnp.minimum(p_next, max_p)
    
    # 6. Update p_prev
    p_prev_next = p
    
    # 7. Update the immutable constraint chi (Safe Check)
    # Calculate the distance to all obstacles
    pos = next_phys_state[:2]
    diff = pos - obstacles_array[:, :2] 
    dists_obs = jnp.linalg.norm(diff, axis=-1) 
    obs_margins = dists_obs - obstacles_array[:, 2]-0.2
    
    workspace_radius = 10.0 
    dist_to_origin = jnp.linalg.norm(pos)
    
    ws_margin = workspace_radius - 0.2 - dist_to_origin
    
    # --- C. Joint Judgment ---
    # Find out which margin is the most dangerous in the obstacle
    min_obs_margin = jnp.min(obs_margins)
    
    # The final safety margin is: (Most Dangerous Obstacle Allowance) and (Workspace Boundary Allowance), the lesser of the two
    total_min_margin = jnp.minimum(min_obs_margin, ws_margin)
    
    # As long as this minimum value is > 0, it means that all conditions are met
    is_safe = (total_min_margin > 0.0).astype(jnp.float32)
    
    # chi is cumulative: once it becomes 0, it is always 0
    chi_next = chi * is_safe
    
    return jnp.concatenate([
        jnp.array([tau_next, p_prev_next, p_next, r_next, chi_next])
    ])