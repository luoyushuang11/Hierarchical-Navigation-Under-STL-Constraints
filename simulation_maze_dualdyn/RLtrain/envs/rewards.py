import jax.numpy as jnp
from config import TGPOConfig

def compute_dense_reward(aug_state, next_aug_state, next_phys_state, subgoals_array):
    """
    Compute the dense reward based on the augmented state transition and the next physical state.
    """
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    p_next = next_aug_state[TGPOConfig.IDX_P]
    chi = aug_state[TGPOConfig.IDX_CHI]
    chi_next = next_aug_state[TGPOConfig.IDX_CHI]
    
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    curr_goal = subgoals_array[safe_p]
    goal_pos = curr_goal[:2]
    goal_radius = curr_goal[2]
    
    # 1. Distance Bonus (directly using Euclidean distance, no detour required)
    # When there are no walls at the beginning of the course, the Euclidean distance is the optimal solution
    # As the wall grows, the Agent will automatically sacrifice a distance bonus to avoid the wall penalty
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    r_dist = -dist * TGPOConfig.LAMBDA_DIST
    
    # 2. Progress rewardsress Reward
    r_prog = (p_next - p) * TGPOConfig.LAMBDA_PROG
    
    # 3. Success Reward
    all_done = (p_next == max_p)
    r_succ = (all_done & (chi_next == 1.0)) * TGPOConfig.LAMBDA_SUCC
    
    # 4. Violation Penalty
    is_violation = (chi_next < 0.5)
    r_inv = is_violation * TGPOConfig.LAMBDA_INV
    
   

    return r_dist + r_prog + r_succ + r_inv 