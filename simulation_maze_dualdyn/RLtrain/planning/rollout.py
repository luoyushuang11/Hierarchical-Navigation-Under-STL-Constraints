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
    Perform a full episode
    
Args:
        rng_key: Random key
        actor_apply_fn: The apply function of the Actor network
        actor_params: Actor network parameters
        init_aug_states: (Batch, 9) Initial enhanced state
        time_vars: (Batch, N_vars) Suggested time parameters
        scenario: Objects containing obstacles and subgoals (from scenarios.py)

Returns:
        traj_data: Contains physical tracks for STL calculations (Batch, T, 4)
        transition_data: Contains data for PPO updates (states, actions, rewards...)
    """
    
    batch_size = init_aug_states.shape[0]
    
    # Extracting Static Environment Data (for JIT)
    obstacles = scenario.obstacles
    subgoals = scenario.subgoals

    # ---  (For lax.scan) ---
    def env_step(carry, _):
        #  Carry
        current_aug_state, key = carry
        
        # 1. Actor 
        key, subkey = jax.random.split(key)
        # Actor 输入: (aug_state, time_vars)
        # 输出: mean, log_std (action)
        mean, log_std = actor_apply_fn(actor_params, current_aug_state, time_vars)
        std = jnp.exp(log_std)
        noise = jax.random.normal(subkey, mean.shape)
        action = jnp.tanh(mean + noise * std) 
        # mean, log_std = actor_apply_fn(actor_params, current_aug_state, time_vars)
        # pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        # action = jnp.tanh(pi.sample(seed=subkey)) 
        
        # 2. Environmental evolution
        # (A) Physical step
        current_phys = current_aug_state[:, TGPOConfig.IDX_PHYS]
        next_phys = jax.vmap(unicycle_step)(current_phys, action)
        
        # (B) Logical step
        # Note: update_augmented_state does not have a vmap internally, you need to use it here
        # Input Shapes: (Batch, 9), (Batch, 4), (Batch, N_vars), (N_goals, 4), (N_obs, 3)
        # [Modify point] Added the incoming obstacles parameter, and added a None (static broadcast) to the corresponding in_axes.
        next_aug_logic_part = jax.vmap(update_augmented_state, in_axes=(0, 0, 0, None, None))(
            current_aug_state, next_phys, time_vars, subgoals, obstacles
        )
        
        # Assemble Next Aug State
        # update_augmented_state returns (tau, p_prev, p, r, chi), which needs to be spelled out with next_phys
        # The next_aug_logic_part here is already (Batch, 5)
        next_aug_state = jnp.concatenate([next_phys, next_aug_logic_part], axis=-1)
        
        # 3.  Dense Reward
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
            'phys_state': current_phys 
        }
        
        return (next_aug_state, key), transition

    # ---  Scan 
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
    
    #  jax.tree_util.tree_map alternative jax.tree_map
    batch_transitions = jax.tree_util.tree_map(transpose_batch, transitions)
    
    # Extract the physical trajectory used for STL (Batch, T, 4)
    phys_traj = batch_transitions['phys_state']
    
    return phys_traj, batch_transitions