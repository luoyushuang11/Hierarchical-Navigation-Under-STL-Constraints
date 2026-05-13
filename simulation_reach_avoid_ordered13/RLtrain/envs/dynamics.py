import jax
import jax.numpy as jnp
from config import TGPOConfig

@jax.jit
def unicycle_step(state, action):
    """
    Strict implementation of Unicycle Dynamics (Eq. 8)
    state: [x, y, theta, v]
    action: [omega, a] (normalized to [-1, 1])
    """
    x, y, theta, v = state[0], state[1], state[2], state[3]
    
   
    
    omega = action[0] * TGPOConfig.ACTION_SCALE[0]
    accel = action[1] * TGPOConfig.ACTION_SCALE[1]
    
    x_new = x + v * jnp.cos(theta) * TGPOConfig.DT
    y_new = y + v * jnp.sin(theta) * TGPOConfig.DT
    theta_new = theta + omega * TGPOConfig.DT
    v_new = v + accel * TGPOConfig.DT
    
    
    

    x_new = jnp.clip(x_new, -TGPOConfig.X_LIMIT, TGPOConfig.X_LIMIT)
    y_new = jnp.clip(y_new, -TGPOConfig.X_LIMIT, TGPOConfig.X_LIMIT)
    

    theta_new = (theta_new + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
   
    v_limit = TGPOConfig.PHYS_SCALES[3]
    v_new = jnp.clip(v_new, -v_limit, v_limit)
    
    return jnp.array([x_new, y_new, theta_new, v_new])