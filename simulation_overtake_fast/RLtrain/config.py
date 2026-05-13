import jax.numpy as jnp

class TGPOConfig:

    """"Speed Training"""
    # === 1. Environment and dynamics (Unicycle) ===
    DT = 0.2
    X_LIMIT = 70.0
   
    ACTION_SCALE = jnp.array([1.0, 4.0]) 
    
    
    PHYS_SCALES = jnp.array([70.0, 4.0, jnp.pi/4.0, 3.0])
    
    
   
    BATCH_SIZE = 512
    HIDDEN_SIZES = (512, 512, 512)
    LR = 1e-4               # Learning rate
    WEIGHT_DECAY = 0.1        # Weight decay 
    GRAD_NORM_CLIP = 0.5      # Grad norm clip 
    PPO_EPOCHS = 10
    
   
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.01
    


    


    
    # 针对fast，normal。
    TIME_SCALE = 175.0 
    # [Modified] 4-stage target window (Start, End)
    # Goal 0 (0-25), Goal 1 (25-50), Goal 2 (50-75), Goal 3 (75-100)
    CUSTOM_TIME_WINDOWS = [
        (0.0, 25.0),
        (25.0, 50.0),
        (50.0, 100.0),
        (100.0, 150.0)
    ]
    LANE_KEEP_WINDOW = (150.0, 175.0)
    
    
    SAMPLING_WINDOWS = [
        (20.0, 25.0),
        (45.0, 50.0),
        (70.0, 100.0),
        (110.0, 150.0)
    ]


    
    """
    # For slow
    
    TIME_SCALE = 125.0 
    
    # Goal 0 (0-25), Goal 1 (25-50), Goal 2 (50-75), Goal 3 (75-100)
    CUSTOM_TIME_WINDOWS = [
        (0.0, 25.0),
        (25.0, 50.0),
        (50.0, 75.0),
        (75.0, 100.0)
    ]
    LANE_KEEP_WINDOW = (100.0, 125.0)

    
    SAMPLING_WINDOWS = [
        (18.0, 25.0),
        (40.0, 50.0),
        (65.0, 75.0),
        (90.0, 100.0)
    ]
    """
    



    
    
    
    # 
    LAMBDA_DIST = 0.3
    LAMBDA_PROG = 20.0
    LAMBDA_SUCC = 20.0
    LAMBDA_INV = -3.5  
    LAMBDA_VEL = 0.5

    
    N_MCMC = 500
    N_WARMUP = 200
    
   
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