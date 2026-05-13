# cbf_functions.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn

@jax.jit
def h_x_jax(x, t):
    
    
   
    


    
    
    
    
    
    
    
    
    
    
    
    """
    

    x1 = x[0]
    x2 = x[1]

    
    obs_x_init = jnp.array([4.0, 8.0, 12.0, 16.0])
    
   
    obs_y      = jnp.array([1.0, 3.0, 1.0, 3.0])
    target_y      = jnp.array([3.0, 1.0, 3.0, 1.0])

    
    #obs_vx     = jnp.array([0.4, 0.6, 0.8, 1.0])
    obs_vx     = jnp.array([0.5, 0.5, 0.8, 1.0])


    x_obs_t = obs_x_init + obs_vx * t

  
    b_safe = -1.22 + jnp.square(x1 - x_obs_t) + jnp.square(x2 - obs_y)
    
   
    overtake_offsets = jnp.array([7.49, 14.99, 29.99, 44.99]) # 对应 b6, b7, b8, b9 的常数项
    time_thresholds  = jnp.array([5.0, 10.0, 20.0, 30.0])    # 对应 5s, 10s...
    

    b_overtake_raw = overtake_offsets + 0.45 - 1.5 * t - jnp.sqrt(jnp.square(x1 - (x_obs_t + 2.0)) + jnp.square(x2 - target_y))

 
    b_overtake = jnp.where(t > time_thresholds, jnp.inf, b_overtake_raw)


    b_lane_raw = 1.95 - jnp.square(x2 - 2.0) # b5
    b_lane = jnp.where(t >= 30.0, jnp.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - jnp.square(x2 - 1.0)

    b_lane_final = jnp.where(t < 30.0, jnp.inf, b_lane_final_raw) # b10

 
    all_neg_b = jnp.concatenate([
        -b_safe, 
        jnp.array([-b_lane]), 
        -b_overtake, 
        jnp.array([-b_lane_final])
    ])
 
    alpha = 500.0 
    

   
    h = -jax_nn.logsumexp(all_neg_b * alpha) / alpha
    """



    x1 = x[0]
    x2 = x[1]

 
    obs_x_init = jnp.array([4.0, 8.0, 12.0, 16.0])

    obs_y      = jnp.array([1.0, 3.0, 1.0, 3.0])
    target_y      = jnp.array([3.0, 1.0, 3.0, 1.0])

    # --- 2. Vectorized Computation Obstacle Current Location ---
    # Utilize the broadcast mechanism: (4,) + scalar * scalar -> (4,)
    # x_obs_t is now a vector containing the current x coordinates of 4 obstacles
    # For example: the first 0.5, the second 0.6, the third 0.7, the fourth 0.8
    obs_vx     = jnp.array([0.5, 0.5, 0.5, 0.5])
    x_obs_t = obs_x_init + obs_vx * t

    # --- 3. Vectorized Computation Safe Distance Constraints (B1 - B4) ---
    # Formula: -2.26 + (x1 - x_obs)^2 + (x2 - y_obs)^2
    # JAX will automatically broadcast scalar x1, x2 to match the shape of the obs array
    # b_safe is an array with a shape of (4,).
    b_safe = -1.22 + jnp.square(x1 - x_obs_t) + jnp.square(x2 - obs_y)
    
    # --- 4. Vectorized Computation Overtaking Constraints (B6 - B9) ---
    # Define the overtaking parameters corresponding to each obstacle
    overtake_offsets = jnp.array([9.99, 19.99, 29.99, 39.99]) # Corresponding to the constant terms of b6, b7, b8, b9
    time_thresholds  = jnp.array([5.0, 10.0, 15.0, 20.0])    # Corresponding to 5s, 10s... to 5s, 10s...
    
    # Calculate the original constraint value
    # Corresponding original code: const - 1.5 - t + x1 - x_obs
    b_overtake_raw = overtake_offsets + 0.45 - 2.0 * t - jnp.sqrt(jnp.square(x1 - (x_obs_t + 2.0)) + jnp.square(x2 - target_y))

    # Time judgment using jnp.where (vectorized judgment)
    # If t > threshold, set it to inf, otherwise use the raw value
    b_overtake = jnp.where(t > time_thresholds, jnp.inf, b_overtake_raw)

    # --- 5. Other individual constraints (B5, B10) ---
    b_lane_raw = 1.95 - jnp.square(x2 - 2.0) # b5
    b_lane = jnp.where(t >= 20.0, jnp.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - jnp.square(x2 - 1.0)

    b_lane_final = jnp.where(t < 20.0, jnp.inf, b_lane_final_raw) # b10

    # --- 6. Merge all constraints and compute LogSumExp ---
    # Stitch all the parts into one large vector: [b_safe(4), b_lane(1), b_overtake(4), b_lane_final(1)]
    # A total of 4+1+4+1 = 10 constraints
    all_neg_b = jnp.concatenate([
        -b_safe, 
        jnp.array([-b_lane]), 
        -b_overtake, 
        jnp.array([-b_lane_final])
    ])
    # Using the same large coefficients as the NumPy version, approximating the real min()
    alpha = 500.0 
    

    # LogSumExp trick: 
    # min(b) ≈ - (1/alpha) * log( sum( exp( -alpha * b ) ) )
    # Corresponding code: - (1/alpha) * logsumexp( alpha * -b )
    
    # Note: jax.nn.logsumexp already has numerical stability handling (subtracting max), so just call it directly
    h = -jax_nn.logsumexp(all_neg_b * alpha) / alpha
    


    



    return h

def h_x_numpy(x, t):
    
    
    
    
    
    """
    # Dynamic overtaking, but speed is defined as high speed and normal version but speed defined as high-speed and normal versions
    x1 = x[0]
    x2 = x[1]

    # --- 1. Define Obstacle Parameters (Arrays) ---
    # 初始 X 位置 [o1， o2， o3， o4]
    obs_x_init = np.array([4.0, 8.0, 12.0, 16.0])

    # 固定 Y 位置 [o1， o2， o3， o4]
    obs_y      = np.array([1.0, 3.0, 1.0, 3.0])
    target_y      = np.array([3.0, 1.0, 3.0, 1.0])

    
    # --- 2. Vectorized Computation Obstacle Current Location ---
    # Numpy broadcast mechanism: (4,) + scalar * scalar -> (4,)
    #obs_vx     = np.array([0.4, 0.6, 0.8, 1.0])
    obs_vx     = np.array([0.5, 0.5, 0.8, 1.0])



    x_obs_t = obs_x_init + obs_vx * t

    # --- 3. Vectorized Computation Safety Distance Constraints (b1 - b4) ---
    # Formula: -2.26 + (x1 - x_obs)^2 + (x2 - y_obs)^2
    # Numpy will automatically broadcast scalars x1, x2 to match the shape of the obs array
    b_safe = -1.22 + np.square(x1 - x_obs_t) + np.square(x2 - obs_y)
    
    # --- 4. Vectorized Computation Overtake Constraints (b6 - b9) ---
    # Define overtaking parameters for each obstacle
    overtake_offsets = np.array([7.49, 14.99, 29.99, 44.99]) # Corresponding constants for b6, b7, b8, b9
    time_thresholds  = np.array([5.0, 10.0, 20.0, 30.0])    
    
    # Calculate the original constraint value
    # Corresponding original code: const - 1.5 - t + x1 - x_obs
    b_overtake_raw = overtake_offsets + 0.45 - 1.5 * t - np.sqrt(np.square(x1 - (x_obs_t + 2.0)) + np.square(x2 - target_y))

    # Time judgment using np.where (vectorized judgment)
    # If t > threshold, set it to inf (not active), otherwise use raw value
    b_overtake = np.where(t > time_thresholds, np.inf, b_overtake_raw)

    # --- 5. Other individual constraints (B5, B10) ---on Other Constraints (b5, b10) ---
    b_lane_raw = 1.95 - np.square(x2 - 2.0) # b5
    b_lane = np.where(t >= 30.0, np.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - np.square(x2 - 1.0)

    # Logic: When t < 30, set to inf (not active); when t >= 30, use the original constraint
    b_lane_final = np.where(t < 30.0, np.inf, b_lane_final_raw) # b10

    
    all_neg_b = np.concatenate([
        -b_safe, 
        np.array([-b_lane]), 
        -b_overtake, 
        np.array([-b_lane_final])
    ])

    
    alpha = 500.0 
    

  
    scaled_neg_b = all_neg_b * alpha
    
    m = np.max(scaled_neg_b)
  
    h = -(np.log(np.sum(np.exp(scaled_neg_b - m))) + m) / alpha
    """



    
    # Dynamic overtaking for low-speed versions
    # Dynamic overtaking, but speed is defined as high speed and normal version
    x1 = x[0]
    x2 = x[1]

   
    obs_x_init = np.array([4.0, 8.0, 12.0, 16.0])

    obs_y      = np.array([1.0, 3.0, 1.0, 3.0])
    target_y      = np.array([3.0, 1.0, 3.0, 1.0])

    

    obs_vx     = np.array([0.5, 0.5, 0.5, 0.5])
    


    x_obs_t = obs_x_init + obs_vx * t


    b_safe = -1.22 + np.square(x1 - x_obs_t) + np.square(x2 - obs_y)
    

    overtake_offsets = np.array([9.99, 19.99, 29.99, 39.99])
    time_thresholds  = np.array([5.0, 10.0, 15.0, 20.0])
    

    b_overtake_raw = overtake_offsets + 0.45 - 2.0 * t - np.sqrt(np.square(x1 - (x_obs_t + 2.0)) + np.square(x2 - target_y))

 
    b_overtake = np.where(t > time_thresholds, np.inf, b_overtake_raw)


    b_lane_raw = 1.95 - np.square(x2 - 2.0) 
    b_lane = np.where(t >= 20.0, np.inf, b_lane_raw)

    b_lane_final_raw = 0.15 - np.square(x2 - 1.0)


    b_lane_final = np.where(t < 20.0, np.inf, b_lane_final_raw) 


    all_neg_b = np.concatenate([
        -b_safe, 
        np.array([-b_lane]), 
        -b_overtake, 
        np.array([-b_lane_final])
    ])


    alpha = 500.0 
    


    scaled_neg_b = all_neg_b * alpha
    
    m = np.max(scaled_neg_b)

    h = -(np.log(np.sum(np.exp(scaled_neg_b - m))) + m) / alpha
    
    

    
    



    


   
    

    
    return h
    