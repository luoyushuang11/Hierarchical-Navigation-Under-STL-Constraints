import jax.numpy as jnp
from config import TGPOConfig

def compute_dense_reward(aug_state, next_aug_state, next_phys_state, subgoals_array):
    """
    [课程学习版] 纯净奖励函数
    不需要任何人工引导，依靠环境难度的渐进变化自然学会避障
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
    
    
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    r_dist = -dist * TGPOConfig.LAMBDA_DIST
    

    r_prog = (p_next - p) * TGPOConfig.LAMBDA_PROG
    

    all_done = (p_next == max_p)
    r_succ = (all_done & (chi_next == 1.0)) * TGPOConfig.LAMBDA_SUCC
    
  
    is_violation = (chi_next < 0.5)
    r_inv = is_violation * TGPOConfig.LAMBDA_INV
    
   

    return r_dist + r_prog + r_succ + r_inv 