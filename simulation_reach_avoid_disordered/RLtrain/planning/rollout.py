import jax
import jax.numpy as jnp
import distrax
from functools import partial
from config import TGPOConfig
from envs.dynamics import unicycle_step
from envs.logic import update_augmented_state
from envs.rewards import compute_dense_reward

@partial(jax.jit, static_argnames=['actor_apply_fn', 'scenario'])
def rollout_episode(
    rng_key,
    actor_apply_fn,
    actor_params,
    init_aug_states,
    time_vars,
    scenario
):
    """
    Execute a full episode
    
    Args:
        rng_key： Random key
        actor_apply_fn： Actor network's apply function
        actor_params： Actor network's parameters
        init_aug_states： Initial enhanced state
        time_vars： （Batch， N_vars） Suggested time parameters
        scenario： Objects containing oblesles and subgoals (from scenarios.py)

    Returns:
        traj_data: Contains physical tracks for STL calculations (Batch, T, 4)
        transition_data: Contains data for PPO updates (states, actions, rewards...)
    """
    
    batch_size = init_aug_states.shape[0]
    
    # Extracting static environment data (for JIT)
    obstacles = scenario.obstacles
    subgoals = scenario.subgoals

    # --- Single-step loop function （For lax.scan） ---
    def env_step(carry, _):
        # Unpacking Carry
        current_aug_state, key = carry
        
        # 1. Actor decision-making
        key, subkey = jax.random.split(key)
        # Actor Input: (aug_state, time_vars)
        # output: mean, log_std (We only sample here action)
        mean, log_std = actor_apply_fn(actor_params, current_aug_state, time_vars)
        std = jnp.exp(log_std)
        noise = jax.random.normal(subkey, mean.shape)
        action = jnp.tanh(mean + noise * std) # Sampling
        # mean, log_std = actor_apply_fn(actor_params, current_aug_state, time_vars)
        # pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        # action = jnp.tanh(pi.sample(seed=subkey)) # sampled and limited to [-1, 1]
        
        # 2. Environmental evolution
        # (A) Physical step
        current_phys = current_aug_state[:, TGPOConfig.IDX_PHYS]
        next_phys = jax.vmap(unicycle_step)(current_phys, action)
        
        # (B) Logical step
       
        next_aug_logic_part = jax.vmap(update_augmented_state, in_axes=(0, 0, 0, None, None))(
            current_aug_state, next_phys, time_vars, subgoals, obstacles
        )
        
        # Assembly Next Aug State
        # update_augmented_state The return is (tau, p_prev, p, r, chi)，Need to fight with next_phys
      
        next_aug_state = jnp.concatenate([next_phys, next_aug_logic_part], axis=-1)
        
        # 3. Dense Reward
        # (Batch, 9), (Batch, 9), (Batch, 4), (N_goals, 4)
        reward = jax.vmap(compute_dense_reward, in_axes=(0, 0, 0, None))(
            current_aug_state, next_aug_state, next_phys, subgoals
        )
        
        # 4. Collect data (for return)
        # We need to save (s, a, r, s', done)
        # done In a fixed timestep task, only the last step is done, where mask=1 is staged
        transition = {
            'obs': current_aug_state,
            'action': action,
            'reward': reward,
            'next_obs': next_aug_state,
            'phys_state': current_phys # Specifically for STL calculations
        }
        
        return (next_aug_state, key), transition

    
    # carry: (init_aug_states, rng_key)
    scan_len = int(TGPOConfig.TIME_SCALE)
    
    (final_aug_state, _), transitions = jax.lax.scan(
        env_step,
        (init_aug_states, rng_key),
        None, # xs=None 
        length=scan_len
    )
    
    
    def transpose_batch(x):
        return jnp.swapaxes(x, 0, 1)
    

    batch_transitions = jax.tree_util.tree_map(transpose_batch, transitions)
    

    phys_traj = batch_transitions['phys_state']
    
    return phys_traj, batch_transitions