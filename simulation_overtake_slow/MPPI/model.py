# MPPI/model.py
import jax
import jax.numpy as jnp
from config import dt as default_dt, V_LIMIT

@jax.jit
def jax_dynamics(state, control, dt=default_dt):
    
    theta = state[2]
    v = state[3]
    omega = control[0]
    accel = control[1]
    

    state_dot = jnp.array([
        v * jnp.cos(theta),  # dx
        v * jnp.sin(theta),  # dy
        omega,               # dtheta
        accel                # dv
    ])
    
   
    next_state = state + state_dot * dt
    
 
    next_state = next_state.at[3].set(
        jnp.clip(next_state[3], -V_LIMIT, V_LIMIT)
    )
    

    
    return next_state


def dynamics(state, control, dt=default_dt):
    import numpy as np
    x, y, theta, v = state
    omega, accel = control
    
    state_dot = np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        omega,
        accel
    ])
    
    next_state = state + state_dot * dt

    next_state[3] = np.clip(next_state[3], -V_LIMIT, V_LIMIT)
    
    return next_state