# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t, time_vars, reach_flags):
    

    


    
    
    flag1, flag2, flag3 = reach_flags[0], reach_flags[1], reach_flags[2]
    x1 = x[0]
    x2 = x[1]

  
    t_rl_1 = jnp.round(time_vars[0]) * 0.2
    b1_hard_val = -2.0 * t + 55.99 + 1.0 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 + 2.5, 2))
    b1_rl_val = -2.0 * t + t_rl_1 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 + 2.5, 2))
    
   
    is_t1_match = jnp.abs(t - t_rl_1) < 1e-4
    flag1_new = flag1 | (is_t1_match & (b1_rl_val >= 0))

    b1_rl = jnp.where(t > t_rl_1, jnp.inf, b1_rl_val)
    b1_hard = jnp.where(t > 28.00, jnp.inf,
                jnp.where(flag1_new, jnp.inf, b1_hard_val))



 
    t_rl_2 = jnp.round(time_vars[1]) * 0.2
    b12_hard_val = -2.0 * t + 87.99 + 1.0 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 2.5, 2))
    b12_rl_val = -2.0 * t + t_rl_2 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 2.5, 2))
    
    is_t2_match = jnp.abs(t - t_rl_2) < 1e-4
    flag2_new = flag2 | (is_t2_match & (b12_rl_val >= 0))

    b12_rl = jnp.where(t > t_rl_2, jnp.inf, b12_rl_val)
    b12_hard = jnp.where(t > 44.00, jnp.inf,
                jnp.where(flag2_new, jnp.inf, b12_hard_val))
    
    
    
  
    t_rl_3 = jnp.round(time_vars[2]) * 0.2
    b13_hard_val = -2.0 * t + 119.99 + 1.0 - jnp.sqrt(jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 5.0, 2))
    b13_rl_val = -2.0 * t + t_rl_3 * 2.0 + 0.99 - jnp.sqrt(jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 5.0, 2))
    
    is_t3_match = jnp.abs(t - t_rl_3) < 1e-4
    flag3_new = flag3 | (is_t3_match & (b13_rl_val >= 0))

    b13_rl = jnp.where(t > t_rl_3, jnp.inf, b13_rl_val)

    b13_hard = jnp.where(t > 60.00, jnp.inf,
                jnp.where(flag3_new, jnp.inf, b13_hard_val))



    b2 = - 1.94 + jnp.power(x1 + 2.5, 2) + jnp.power(x2, 2)
    # b2 = - 2.89 + jnp.power(x1 + 2.5, 2) + jnp.power(x2, 2)
    
    b3 = - 1.94 + jnp.power(x1 - 6.0, 2) + jnp.power(x2 + 2.5, 2)
    # b3 = - 1.44 + jnp.power(x1, 2) + jnp.power(x2 + 5.0, 2)
    
    
    b4 = - 1.94 + jnp.power(x1 - 2.5, 2) + jnp.power(x2, 2)
    
    b5 = - 1.94 + jnp.power(x1 - 2.5, 2) + jnp.power(x2 - 5.0, 2)
    # b5 = - 1.00 + jnp.power(x1, 2) + jnp.power(x2 - 5.0, 2)
    
    
    b6 = - 1.94 + jnp.power(x1 + 2.5, 2) + jnp.power(x2 + 2.5, 2)
    # b6 = - 1.00 + jnp.power(x1 + 2.5, 2) + jnp.power(x2 + 3.0, 2)
    
    b7 = - 1.94 + jnp.power(x1 - 6.0, 2) + jnp.power(x2 - 2.5, 2)
    # b7 = - 1.00 + jnp.power(x1 + 1.0, 2) + jnp.power(x2 - 2.5, 2)
    
    
    b8 = - 1.94 + jnp.power(x1 - 6.0, 2) + jnp.power(x2 - 5.0, 2)
    # b8 = - 1.00 + jnp.power(x1 - 5.0, 2) + jnp.power(x2 - 5.0, 2)
    
    
    
    b9 = - 1.94 + jnp.power(x1 - 6.0, 2) + jnp.power(x2, 2)
    
    
    # b10 = - 1.00 + jnp.power(x1 + 5.0, 2) + jnp.power(x2 - 2.5, 2)
    b10 = - 1.94 + jnp.power(x1 + 2.5, 2) + jnp.power(x2 - 2.5, 2)

   
    b11 = 99.10 - jnp.power(x1, 2) - jnp.power(x2, 2)

    all_b = jnp.concatenate([
        jnp.atleast_1d(b1_rl), jnp.atleast_1d(b1_hard),      
        jnp.atleast_1d(b2), jnp.atleast_1d(b3), jnp.atleast_1d(b4), 
        jnp.atleast_1d(b5), jnp.atleast_1d(b6), jnp.atleast_1d(b7), 
        jnp.atleast_1d(b8), jnp.atleast_1d(b9), jnp.atleast_1d(b10), 
        jnp.atleast_1d(b11),          
        jnp.atleast_1d(b12_rl), jnp.atleast_1d(b12_hard),     
        jnp.atleast_1d(b13_rl), jnp.atleast_1d(b13_hard),     
    ], axis=-1)

    

    h = jnp.min(all_b, axis=-1)
    

    new_reach_flags = jnp.array([flag1_new, flag2_new, flag3_new])
    
    

    
    return h, new_reach_flags
    
    
    


    
    
    
    
    
