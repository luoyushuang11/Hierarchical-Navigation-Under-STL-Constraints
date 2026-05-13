import jax.numpy as jnp
from config import TGPOConfig

class TaskScenario:
    def __init__(self, name="default"):
        self.name = name
        self.obstacles = None
        self.subgoals = None
        
        self.dynamic_obs_params = None 
        
        if name == "reach_avoid_simple":
            self._setup_reach_avoid_simple()
        elif name == "dynamic_overtaking":
            self._setup_dynamic_overtaking()
        else:
            raise ValueError(f"Unknown scenario: {name}")

    def _setup_reach_avoid_simple(self):
      
        self.obstacles = jnp.array([
            [4.0,  1.0, 0.5], 
            [8.0,  3.0, 0.5], 
            [12.0, 1.0, 0.5], 
            [16.0, 3.0, 0.5], 
        ])
        # [x, y, r, time_idx]
        self.subgoals = jnp.array([
            [6.0, 3.0, 0.5, 0], 
            [10.0, 1.0, 0.5, 1], 
            [14.0, 3.0, 0.5, 2],
            [18.0, 1.0, 0.5, 3]   
        ])

    def _setup_dynamic_overtaking(self):
        """
        
        [start_x, start_y, v_x, radius]
        """


        
        self.dynamic_obs_params = jnp.array([
            [4.0,  1.0, 0.4, 0.5], # Obs 1
            [8.0,  3.0, 0.6, 0.5], # Obs 2
            [12.0, 1.0, 0.8, 0.5], # Obs 3
            [16.0, 3.0, 1.0, 0.5], # Obs 4
        ]) # for fastest
        


        
        """
        self.dynamic_obs_params = jnp.array([
            [4.0,  1.0, 0.5, 0.5], # Obs 1
            [8.0,  3.0, 0.5, 0.5], # Obs 2
            [12.0, 1.0, 0.8, 0.5], # Obs 3
            [16.0, 3.0, 1.0, 0.5], # Obs 4
        ]) # for normal
        """
        

        
        """
        self.dynamic_obs_params = jnp.array([
            [4.0,  1.0, 0.5, 0.5], # Obs 1
            [8.0,  3.0, 0.5, 0.5], # Obs 2
            [12.0, 1.0, 0.5, 0.5], # Obs 3
            [16.0, 3.0, 0.5, 0.5], # Obs 4
        ]) # for slow
        """
        
        
        self.obstacles = self.dynamic_obs_params[:, [0, 1, 3]] 

        
        self.subgoals = jnp.array([
            [2.0, 3.0, 0.5, 0, 0], # Goal 0: Obs 1 + 2m
            [2.0, 1.0, 0.5, 1, 1], # Goal 1: Obs 2 + 2m
            [2.0, 3.0, 0.5, 2, 2], # Goal 2: Obs 3 + 2m
            [2.0, 1.0, 0.5, 3, 3]  # Goal 3: Obs 4 + 2m
        ])

def get_scenario(name="reach_avoid_simple"):
    return TaskScenario(name)