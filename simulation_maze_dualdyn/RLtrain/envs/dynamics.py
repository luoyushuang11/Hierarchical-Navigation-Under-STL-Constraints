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
    
    
    # omega: [-1, 1] rad/s
    # accel: [-2, 2] m/s^2
    
    omega = action[0] * TGPOConfig.ACTION_SCALE[0]
    accel = action[1] * TGPOConfig.ACTION_SCALE[1]
    
    # [cite_start]2. 
    # x_{t+1} = x_t + v_t * cos(theta_t) * dt
    x_new = x + v * jnp.cos(theta) * TGPOConfig.DT
    y_new = y + v * jnp.sin(theta) * TGPOConfig.DT
    theta_new = theta + omega * TGPOConfig.DT
    v_new = v + accel * TGPOConfig.DT
    
   
    
   
    x_new = jnp.clip(x_new, -3, 15)
    y_new = jnp.clip(y_new, -3, 13)
    
    # (B) Angle normalization (Wrap to [-pi, pi])
    # The neural network cannot understand the difference between 3pi and pi and has to wrap back
    theta_new = (theta_new + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Clip to V_LIMIT
    v_limit = TGPOConfig.PHYS_SCALES[3]
    v_new = jnp.clip(v_new, -v_limit, v_limit)
    
    return jnp.array([x_new, y_new, theta_new, v_new])