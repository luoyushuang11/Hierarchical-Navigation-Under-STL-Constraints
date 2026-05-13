# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t):
    
    
    # reach_avoid 9 Obstacles are out of order
    x1 = x[0]
    x2 = x[1]

    # Target region constraints Time-varying CBF
    b1_original = -4.0 * t + 111.7 + 1.0 - jnp.power(x1 - 2.5, 2) - jnp.power(x2 + 2.5, 2) # 0到28s
    b1 = jnp.where(t > 28.00, jnp.inf, b1_original)
    b12_original = -4.0 * t + 175.7 + 1.0 - jnp.power(x1 - 2.0, 2) - jnp.power(x2 - 3.0, 2) # 28到44s
    b12 = jnp.where(t > 44.00, jnp.inf, b12_original)
    b13 = -4.0 * t + 239.7 + 1.0 - jnp.power(x1 + 2.5, 2) - jnp.power(x2 - 5.0, 2) # 44到60s

    # Indicates obstacles
    b2 = - 2.89 + jnp.power(x1 + 2.5, 2) + jnp.power(x2, 2)
    b3 = - 1.44 + jnp.power(x1, 2) + jnp.power(x2 + 5.0, 2)
    b4 = - 1.44 + jnp.power(x1 - 2.5, 2) + jnp.power(x2, 2)
    b5 = - 1.00 + jnp.power(x1, 2) + jnp.power(x2 - 5.0, 2)
    b6 = - 1.00 + jnp.power(x1 + 2.5, 2) + jnp.power(x2 + 3.0, 2)
    b7 = - 1.00 + jnp.power(x1 + 1.0, 2) + jnp.power(x2 - 2.5, 2)
    b8 = - 1.00 + jnp.power(x1 - 5.0, 2) + jnp.power(x2 - 5.0, 2)
    b9 = - 1.44 + jnp.power(x1 - 6.0, 2) + jnp.power(x2, 2)
    b10 = - 1.00 + jnp.power(x1 + 5.0, 2) + jnp.power(x2 - 2.5, 2)

    # Work area
    b11 = 99.10 - jnp.power(x1, 2) - jnp.power(x2, 2)

    # Use jax.nn.logsumexp for numerical stability
    neg_b = jnp.array([-b1, -b2, -b3, -b4, -b5, -b6, -b7, -b8, -b9, -b10, -b11, -b12, -b13])
    alpha = 100.0
    h = -jax_nn.logsumexp(neg_b * alpha) / alpha
    
    




    
    
    
    
    
    
    

    return h

def h_x_numpy(x, t):
    
    
    # Disorganized
    x1 = x[0]
    x2 = x[1]

    # Target area
    b1_original = -4.0 * t + 111.7 + 1.0 - np.power(x1 - 2.5, 2) - np.power(x2 + 2.5, 2) # 0到28s
    b1 = np.where(t > 28.00, np.inf, b1_original)
    b12_original = -4.0 * t + 175.7 + 1.0 - np.power(x1 - 2.0, 2) - np.power(x2 - 3.0, 2) # 28到44s
    b12 = np.where(t > 44.00, np.inf, b12_original)
    b13 = -4.0 * t + 239.7 + 1.0 - np.power(x1 + 2.5, 2) - np.power(x2 - 5.0, 2) # 44到60s

    # Obstacles
    b2 = - 2.89 + np.power(x1 + 2.5, 2) + np.power(x2, 2)
    b3 = - 1.44 + np.power(x1, 2) + np.power(x2 + 5.0, 2)
    b4 = - 1.44 + np.power(x1 - 2.5, 2) + np.power(x2, 2)
    b5 = - 1.00 + np.power(x1, 2) + np.power(x2 - 5.0, 2)
    b6 = - 1.00 + np.power(x1 + 2.5, 2) + np.power(x2 + 3.0, 2)
    b7 = - 1.00 + np.power(x1 + 1.0, 2) + np.power(x2 - 2.5, 2)
    b8 = - 1.00 + np.power(x1 - 5.0, 2) + np.power(x2 - 5.0, 2)
    b9 = - 1.44 + np.power(x1 - 6.0, 2) + np.power(x2, 2)
    b10 = - 1.00 + np.power(x1 + 5.0, 2) + np.power(x2 - 2.5, 2)

    # Work area
    b11 = 99.10 - np.power(x1, 2) - np.power(x2, 2)

    # Implement numerically stable logsumexp for numpy
    neg_b = np.array([-b1, -b2, -b3, -b4, -b5, -b6, -b7, -b8, -b9, -b10, -b11, -b12, -b13])
    alpha = 100.0
    scaled_neg_b = neg_b * alpha

    m = np.max(scaled_neg_b)
    h = -(np.log(np.sum(np.exp(scaled_neg_b - m))) + m) / alpha
   
   
    
    
   
    


    
    return h
    