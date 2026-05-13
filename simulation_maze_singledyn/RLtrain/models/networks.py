import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from config import TGPOConfig

#  (Orthogonal Initialization) 
kernel_init = nn.initializers.orthogonal(np.sqrt(2))
action_init = nn.initializers.orthogonal(0.01)

class Actor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, aug_state, time_vars):
        """
        Actor Network: Input (s, t) -> Output (action_mean, log_std)
        """
        # 1. Strict Normalization
        #  PHYS_SCALES (5.0, 5.0, pi, 5.0) -> [-1, 1]
        phys_state = aug_state[..., TGPOConfig.IDX_PHYS]
        norm_phys = phys_state / TGPOConfig.PHYS_SCALES
        
        #  TIME_SCALE -> [0, 1]
        tau = aug_state[..., TGPOConfig.IDX_TAU:TGPOConfig.IDX_TAU+1]
        norm_tau = tau / TGPOConfig.TIME_SCALE
        
        # p_prev, p, r, chi
        logic_state = aug_state[..., TGPOConfig.IDX_P_PREV:]
        
        #  TIME_SCALE -> [0, 1]
        
        time_flat = time_vars.reshape(time_vars.shape[0], -1)
        norm_time = time_flat / TGPOConfig.TIME_SCALE
        
       
        x = jnp.concatenate([norm_phys, norm_tau, logic_state, norm_time], axis=-1)
        
        # 3. MLP Backbone (3 tiers x 512 units)
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        
        # 4. Output layer
        # mean: Use tanh to limit to [-1, 1], corresponding to ACTION_SCALE in dynamics
        mean = nn.Dense(self.action_dim, kernel_init=action_init)(x)
        mean = nn.tanh(mean)
        
        # Log Std
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        

        return mean, log_std

class Critic(nn.Module):
    @nn.compact
    def __call__(self, aug_state, time_vars):
        """
        Critic Network: Input (s, t) -> Output (Value)
        """
        
        phys_state = aug_state[..., TGPOConfig.IDX_PHYS]
        norm_phys = phys_state / TGPOConfig.PHYS_SCALES
        
        tau = aug_state[..., TGPOConfig.IDX_TAU:TGPOConfig.IDX_TAU+1]
        norm_tau = tau / TGPOConfig.TIME_SCALE
        
        logic_state = aug_state[..., TGPOConfig.IDX_P_PREV:]
        
        time_flat = time_vars.reshape(time_vars.shape[0], -1)
        norm_time = time_flat / TGPOConfig.TIME_SCALE
        
        x = jnp.concatenate([norm_phys, norm_tau, logic_state, norm_time], axis=-1)
        
        
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=kernel_init)(x)
        x = nn.relu(x) # Critic 通常用 ReLU
        
        
        value = nn.Dense(1, kernel_init=kernel_init)(x)
        
        return jnp.squeeze(value, -1)