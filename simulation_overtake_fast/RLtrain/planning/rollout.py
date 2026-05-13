import jax
import jax.numpy as jnp
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

    obs_params = scenario.dynamic_obs_params
    subgoals = scenario.subgoals

    def env_step(carry, _):
        current_aug_state, key = carry
        
        # 1. Actor
        key, subkey = jax.random.split(key)
        mean, log_std = actor_apply_fn(actor_params, current_aug_state, time_vars)
        std = jnp.exp(log_std)
        noise = jax.random.normal(subkey, mean.shape)
        action = jnp.tanh(mean + noise * std)
        
        # 2. Physics
        current_phys = current_aug_state[:, TGPOConfig.IDX_PHYS]
        next_phys = jax.vmap(unicycle_step)(current_phys, action)
        
        # 3. Logic
        next_aug_logic_part = jax.vmap(update_augmented_state, in_axes=(0, 0, 0, None, None))(
            current_aug_state, next_phys, time_vars, subgoals, obs_params
        )
        next_aug_state = jnp.concatenate([next_phys, next_aug_logic_part], axis=-1)
        
        # 4. Reward 
        reward = jax.vmap(compute_dense_reward, in_axes=(0, 0, 0, None, None))(
            current_aug_state, next_aug_state, next_phys, subgoals, obs_params
        )
        
        transition = {
            'obs': current_aug_state,
            'action': action,
            'reward': reward,
            'next_obs': next_aug_state,
            'phys_state': current_phys
        }
        return (next_aug_state, key), transition

    scan_len = int(TGPOConfig.TIME_SCALE)
    (final_aug_state, _), transitions = jax.lax.scan(
        env_step, (init_aug_states, rng_key), None, length=scan_len
    )
    
    def transpose_batch(x): return jnp.swapaxes(x, 0, 1)
    batch_transitions = jax.tree_util.tree_map(transpose_batch, transitions)
    phys_traj = batch_transitions['phys_state']
    
    return phys_traj, batch_transitions