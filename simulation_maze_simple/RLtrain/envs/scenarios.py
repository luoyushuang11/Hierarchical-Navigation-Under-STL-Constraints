import jax.numpy as jnp
from config import TGPOConfig

class TaskScenario:
    def __init__(self, name="default"):
        self.name = name
        self.obstacles = None
        self.subgoals = None
        self.map_size = TGPOConfig.X_LIMIT

        if name == "reach_avoid_simple":
            self._setup_reach_avoid_simple()
        elif name == "multi_goal_sequence":
            self._setup_multi_goal_sequence()
        elif name == "dynamic_overtaking":
            self._setup_dynamic_overtaking()
        elif name == "local_minima_trap":
            self._setup_local_minima_trap()
        else:
            raise ValueError(f"Unknown scenario: {name}")

    def _setup_reach_avoid_simple(self):
       
        self.obstacles = jnp.array([
            [-2.5, 0.0, 1.5],
            [ 0.0, -5.0, 1.0],
            [ 2.5, 0.0, 1.0],
            [ 0.0, 5.0, 0.75], 
            [ -2.5, -3.0, 0.75],
            [ -1.0, 2.5, 0.75],
            [ 5.0, 5.0, 0.75],
            [ 6.0, 0.0, 1.0],
            [ -5.0, 2.5, 0.75],
        ])
        

        self.subgoals = jnp.array([
            [2.5,  -2.5, 1.0, 0],  # Goal 0
            [2.0,  3.0, 1.0, 1],   # Goal 1
            [-2.5, 5.0, 1.0, 2]    # Goal 2
        ])

    def _setup_multi_goal_sequence(self):
      
        self.obstacles = jnp.array([
            [-3.0, -3.0, 0.5], [-3.0, 0.0, 0.5], [-3.0, 3.0, 0.5],
            [ 0.0, -3.0, 0.5], [ 0.0, 0.0, 0.5], [ 0.0, 3.0, 0.5],
            [ 3.0, -3.0, 0.5], [ 3.0, 0.0, 0.5], [ 3.0, 3.0, 0.5],
        ])
        

        self.subgoals = jnp.array([
            [-4.0, 4.0, 0.6, 0], 
            [ 4.0, -4.0, 0.6, 1], 
            [ 0.0, 4.2, 0.6, 2]  
        ])

    def _setup_dynamic_overtaking(self):
       
        self.obstacles = jnp.array([
            [4.0, 1.0, 0.5],
            [8.0, 3.0, 0.5],
            [12.0, 1.0, 0.5],
            [16.0, 3.0, 0.5]
        ])
        
  
        self.dynamic_obs_params = jnp.array([
            [4.0, 1.0, 0.5, 0.5],
            [8.0, 3.0, 0.6, 0.5],
            [12.0, 1.0, 0.7, 0.5],
            [16.0, 3.0, 0.8, 0.5]
        ])
        

        self.subgoals = jnp.array([
            [2.0, 3.0, 0.5, 0, 0],
            [2.0, 1.0, 0.5, 1, 1],
            [2.0, 3.0, 0.5, 2, 2],
            [2.0, 1.0, 0.5, 3, 3]
        ])

    def _setup_local_minima_trap(self):
       
        self.obstacles = jnp.array([
            [2.0,  1.0, 2.0, 2.0],     # Obs 1
            [1.5,  4.25, 3.0, 0.5],    # Obs 2
            [2.0, 7.0, 2.0, 2.0],      # Obs 3
            [4.75, 6.0, 0.5, 8.0],     # Obs 4 
            [8.25, 4.0, 0.5, 8.0],     # Obs 5 
            [11.25, 3.5, 1.5, 1.0],    # Obs 6
            [9.25,  5.5, 1.5, 1.0],    # Obs 7
            [11.5, 9.0, 1.0, 2.0],      # Obs 8
        ])
        
        
        self.subgoals = jnp.array([
            [1.0, 3.0, 0.6, 0],  
            [2.25, 9.0, 0.6, 1]   
        ])


def get_scenario(name="reach_avoid_simple"):
    return TaskScenario(name)