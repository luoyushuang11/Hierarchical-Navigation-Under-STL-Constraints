import jax
import jax.numpy as jnp
from config import TGPOConfig


DYN_OBS_PARAMS = jnp.array([
    [6000.0, 0.5, 9.5, 0.4, 0.5, 0.0]
    
])




def sd_box(p, box_center, box_size):
    d = jnp.abs(p - box_center) - (box_size / 2.0)

    return jnp.linalg.norm(jnp.maximum(d, 0.0)) + jnp.minimum(jnp.maximum(d[0], d[1]), 0.0)


def compute_single_dyn_pos(params, t):

    x_fixed, y_min, y_max, speed, _, offset = params
    
    length = y_max - y_min
    cycle = 2.0 * length
    

    dist = t * speed + offset
    
    mod = dist % cycle

    y_pos = jnp.where(mod <= length, y_min + mod, y_max - (mod - length))
    return jnp.array([x_fixed, y_pos])

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
    

    dist_to_goal = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    mu_satisfied = dist_to_goal <= goal_radius
    

    time_match = (tau_next == t_target)
    
    cond_reset = (r == 2)
    cond_finish = (r != 2) & time_match & mu_satisfied
    
    r_next = jnp.where(cond_reset, 0.0,
                jnp.where(cond_finish, 2.0, r))
    

    p_increment = (r_next == 2.0).astype(jnp.float32)
    p_next = jnp.minimum(p + p_increment, max_p)
    p_prev_next = p
    

    pos = next_phys_state[:2]
    

    def check_single_obs(obs_params):
        return sd_box(pos, obs_params[:2], obs_params[2:])
    

    dists_static = jax.vmap(check_single_obs)(obstacles_array)
    min_dist_static = jnp.min(dists_static)
    

    current_time = tau_next * TGPOConfig.DT
    

    dyn_positions = jax.vmap(compute_single_dyn_pos, in_axes=(0, None))(DYN_OBS_PARAMS, current_time)
    

    dists_dyn = jnp.linalg.norm(pos - dyn_positions, axis=1) - DYN_OBS_PARAMS[:, 4]
    

    min_dist_dyn = jnp.min(dists_dyn)
    

    total_min_dist = jnp.minimum(min_dist_static, min_dist_dyn)
    rect_safe = total_min_dist > 0.3
    #rect_safe = min_dist_static > 0.3
    x, y = pos[0], pos[1]
    ws_safe_x = (x >= 0.3) & (x <= 11.7)
    ws_safe_y = (y >= 0.3) & (y <= 9.7)
    workspace_safe = ws_safe_x & ws_safe_y
    

    is_safe = (rect_safe & workspace_safe).astype(jnp.float32)
    

    chi_next = chi * is_safe
    

    return jnp.concatenate([
        jnp.array([tau_next, p_prev_next, p_next, r_next, chi_next])
    ])