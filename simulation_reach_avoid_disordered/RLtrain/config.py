import jax.numpy as jnp

class TGPOConfig:
    # === 1. Environment and dynamics  ===
    DT = 0.2
    X_LIMIT = 10.0
    ACTION_SCALE = jnp.array([1.0, 4.0]) 
    
    # Normalize parameters
    PHYS_SCALES = jnp.array([5.0, 5.0, jnp.pi, 2.0])
    
    # === 2. Training hyperparameters  ===
    
    BATCH_SIZE = 512
    HIDDEN_SIZES = (512, 512, 512)
    LR = 1e-5                # Learning rate
    WEIGHT_DECAY = 0.1        # Weight decay 
    GRAD_NORM_CLIP = 0.5      # Grad norm clip 
    PPO_EPOCHS = 10
 
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.005

   
    TIME_SCALE = 300.0 
    CUSTOM_TIME_WINDOWS = [
        (0.0, 140.0),
        (140.0, 220.0),
        (220.0, 300.0)
    ]
   
    """
    SAMPLING_WINDOWS = [
        (135.0, 140.0), # Task 0
        (215.0, 220.0), # Task 1
        (295.0, 300.0)  # Task 2
    ]"""
    #使用采样出来的最优时间点来构造CBF
    SAMPLING_WINDOWS = [
        (50.0, 140.0), # Task 0
        (140.0, 220.0), # Task 1
        (220.0, 300.0)  # Task 2
    ]
    
    # === 4. Reward weight  ===
    # 
    LAMBDA_DIST = 0.3
    LAMBDA_PROG = 20.0
    LAMBDA_SUCC = 20.0
    LAMBDA_INV = -3.5 
    
    # === 5. MCMC  ===
    N_MCMC = 500
    N_WARMUP = 200
    
    
    # Ratio of Random (Uniform): 0.5
    # Ratio of MCMC: 0.4
    # Ratio of Elite: 0.1
    RATIO_UNI = 0.50
    RATIO_MCMC = 0.4
    RATIO_ELITE = 0.1

    # === 6. Status index ===
    IDX_PHYS = slice(0, 4)
    IDX_TAU = 4
    IDX_P_PREV = 5
    IDX_P = 6
    IDX_R = 7
    IDX_CHI = 8
    
    AUG_STATE_DIM = 9
    PHYS_STATE_DIM = 4