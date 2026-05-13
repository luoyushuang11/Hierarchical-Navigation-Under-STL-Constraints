import jax
import jax.numpy as jnp
from config import TGPOConfig

def sample_uniform_times(rng_key, n_samples, n_vars):
  
    sampling_windows = TGPOConfig.SAMPLING_WINDOWS

    keys = jax.random.split(rng_key, n_vars)
    samples = []
    
    for i in range(n_vars):
    
        t_min, t_max = sampling_windows[i]
        
        t_i = jax.random.uniform(
            keys[i], 
            shape=(n_samples, 1), 
            minval=t_min, 
            maxval=t_max
        )
        samples.append(t_i)
        
    times = jnp.concatenate(samples, axis=-1)
    return times