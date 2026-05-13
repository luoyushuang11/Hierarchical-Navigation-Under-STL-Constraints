import jax.numpy as jnp
from config import TGPOConfig

def compute_dense_reward(aug_state, next_aug_state, next_phys_state, subgoals_array):
    """
    Strict implementation of Eq. (6) [cite: 180]
    """
    p = aug_state[TGPOConfig.IDX_P].astype(jnp.int32)
    p_next = next_aug_state[TGPOConfig.IDX_P]
    chi_next = next_aug_state[TGPOConfig.IDX_CHI]
    
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    
    # 1.  R_dist
    goal_pos = subgoals_array[safe_p][:2]
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    r_dist = -dist * TGPOConfig.LAMBDA_DIST
    
    # 2. R_progress (Triggered when p increases)
    r_prog = (p_next - p) * TGPOConfig.LAMBDA_PROG
    
    # 3. R_success (All Done and Secure)
    all_done = (p_next == max_p)
    r_succ = (all_done & (chi_next == 1.0)) * TGPOConfig.LAMBDA_SUCC
    
    # 4. Penalty R_inv for violations (triggered at chi=0)
    is_violation = (chi_next < 0.5)
    r_inv = is_violation * TGPOConfig.LAMBDA_INV
    
    return r_dist + r_prog + r_succ + r_inv