import jax
import jax.numpy as jnp
from config import TGPOConfig

def metropolis_hastings_sampling(rng_key, critic_apply_fn, critic_params, initial_states, current_times):
   
    
    batch_size, n_vars = current_times.shape
    
  
    windows = jnp.array(TGPOConfig.SAMPLING_WINDOWS) # Shape: (N_vars, 2)
    
    t_min_constraints = windows[:, 0] # [120, 200, 280]
    t_max_constraints = windows[:, 1] # [140, 220, 300]
    

    current_values = critic_apply_fn(critic_params, initial_states, current_times)

   
    def mcmc_step(carry, _):
        times, values, key = carry
        
      
        key, k1, k2, k3 = jax.random.split(key, 4)
        noise = jax.random.normal(k1, shape=times.shape) * 2.0 
        proposed_times = times + noise
        
       
        mask_min = jnp.all(proposed_times >= t_min_constraints, axis=-1)
        mask_max = jnp.all(proposed_times <= t_max_constraints, axis=-1)
        
        
        diffs = jnp.diff(proposed_times, axis=-1)
        mask_order = jnp.all(diffs >= 0, axis=-1)
        
    
        mask_zero = proposed_times[:, 0] >= 0
        
       
        valid_mask = mask_min & mask_max & mask_order & mask_zero
        
        
        v_new = critic_apply_fn(critic_params, initial_states, proposed_times)
        
        
        temperature = 1.0
        log_alpha = (v_new - values) / temperature
        
     
        log_alpha = jnp.where(valid_mask, log_alpha, -jnp.inf)
        
        u = jax.random.uniform(k3, (batch_size,))
        accept_mask = jnp.log(u) < log_alpha
        
       
        next_times = jnp.where(accept_mask[:, None], proposed_times, times)
        next_values = jnp.where(accept_mask, v_new, values)
        
        return (next_times, next_values, key), (next_times, next_values)

   
    init_carry = (current_times, current_values, rng_key)
    
   
    _, (history_times, history_values) = jax.lax.scan(
        mcmc_step, 
        init_carry, 
        None, 
        length=TGPOConfig.N_MCMC
    )


    valid_times = history_times[TGPOConfig.N_WARMUP:]   
    valid_values = history_values[TGPOConfig.N_WARMUP:] 
    
  
    best_indices = jnp.argmax(valid_values, axis=0) # (Batch,)
    
  
    def get_best(times_history, idx):
        return times_history[idx]
        
    valid_times_T = jnp.transpose(valid_times, (1, 0, 2))
    final_best_times = jax.vmap(get_best)(valid_times_T, best_indices)
    
    return final_best_times