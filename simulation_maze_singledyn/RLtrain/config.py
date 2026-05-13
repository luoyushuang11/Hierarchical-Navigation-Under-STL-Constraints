import jax.numpy as jnp

class TGPOConfig:
    # === 1. Environment and dynamics (Unicycle) ===
    DT = 0.2
    X_LIMIT = 12.0
    # Action space: omega [-1, 1], accel [-4, 4] 
    ACTION_SCALE = jnp.array([1.0, 4.0]) 
    
    # Normalization parameters
    PHYS_SCALES = jnp.array([5.0, 5.0, jnp.pi, 2.0])
    
    # === 2. Training Hyperparameters===
    #  Table 2
    BATCH_SIZE = 512
    HIDDEN_SIZES = (512, 512, 512)
    LR = 3e-5                # Learning rate
    WEIGHT_DECAY = 0.1        # Weight decay 
    GRAD_NORM_CLIP = 0.5      # Grad norm clip 
    PPO_EPOCHS = 10
    
    
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.015

    # === 3. Task and Time Parameters ===
    # [New] Custom Time Window Configuration (Start, End)
    # Unified Management: Sampling(Uniform) and Evaluation(STL) both read from here
    # Task 0: [0, 140] (Longest, 140 steps)
    # Task 1: [140, 220] (80 steps)
    # Task 2: [220, 300] (80 steps)
    TIME_SCALE = 200
    CUSTOM_TIME_WINDOWS = [
        (0.0, 150.0),
        (150.0, 200.0)
    ]
    # [New] Training Sampling Windows
    # Force Policy to learn within the "last 20 steps" of each window
    # This corresponds to the Forward Invariant Set property of CBF-MPPI
    """SAMPLING_WINDOWS = [
        (145.0, 150.0),
        (195.0, 200.0)
    ]"""

    SAMPLING_WINDOWS = [
        (145.0, 150.0),
        (195.0, 200.0)
    ] # For v2 155-200 100-150, if v3 is changed to 150-200 100-150 v1 125-150 175 -200
    
   
    LAMBDA_DIST = 0.3
    LAMBDA_PROG = 20.0
    LAMBDA_SUCC = 20.0
    LAMBDA_INV = -3.5  
    
    # ===  MCMC 
    N_MCMC = 500
    N_WARMUP = 200
    
    
    # Ratio of Random (Uniform): 0.5
    # Ratio of MCMC: 0.4
    # Ratio of Elite: 0.1
    RATIO_UNI = 0.50
    RATIO_MCMC = 0.40
    RATIO_ELITE = 0.10

    
    IDX_PHYS = slice(0, 4)
    IDX_TAU = 4
    IDX_P_PREV = 5
    IDX_P = 6
    IDX_R = 7
    IDX_CHI = 8
    
    AUG_STATE_DIM = 9
    PHYS_STATE_DIM = 4