import jax
import jax.numpy as jnp
from functools import partial
from config import TGPOConfig

class EliteBuffer:
    def __init__(self, n_vars, capacity=TGPOConfig.BATCH_SIZE):
       
        self.capacity = capacity
        self.n_vars = n_vars
        
        
        self.times = jnp.zeros((capacity, n_vars))
        
        
        self.scores = jnp.full((capacity,), -1e9)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, current_times, current_scores, batch_new_times, batch_new_scores):
        
        all_times = jnp.concatenate([current_times, batch_new_times], axis=0)
        # shape: (1024,)
        all_scores = jnp.concatenate([current_scores, batch_new_scores], axis=0)
        
       
        sort_idx = jnp.argsort(all_scores)
        
      
        top_k_idx = sort_idx[-self.capacity:]
        
    
        new_times = all_times[top_k_idx]
        new_scores = all_scores[top_k_idx]
        
        return new_times, new_scores

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, current_times, n_sample, rng_key):
        

        idx = jax.random.randint(
            rng_key,
            shape=(n_sample,),
            minval=0,
            maxval=self.capacity
        )

   

        sampled_times = current_times[idx] 
        return sampled_times