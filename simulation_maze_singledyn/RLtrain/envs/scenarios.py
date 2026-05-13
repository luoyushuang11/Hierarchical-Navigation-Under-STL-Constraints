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
        """
        Advanced Test Scenarios:
        - Barrier-free (safe area)
        - 3 sub-objectives (sequence tasks)
        """
        # 1. False obstacles (placed far away)
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
        
        # 2. Set 3 sub-goals
        self.subgoals = jnp.array([
            [2.5,  -2.5, 1.0, 0],  # Goal 0
            [2.0,  3.0, 1.0, 1],   # Goal 1
            [-2.5, 5.0, 1.0, 2]    # Goal 2
        ])

    def _setup_multi_goal_sequence(self):
        """
        Scenario V4: Wide-area nine-grid blockade
        """
        # 1. Obstacles: 3x3 wide area
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
        """
        Dynamic Overtaking Scenario (保留以兼容之前的训练)
        """
        # Static obstacle placeholders (actual dynamic parameters in dynamic_obs_params)
        # Here we only define initial positions to prevent code errors
        self.obstacles = jnp.array([
            [4.0, 1.0, 0.5],
            [8.0, 3.0, 0.5],
            [12.0, 1.0, 0.5],
            [16.0, 3.0, 0.5]
        ])
        
        # Dynamic Obstacle Parameters: [start_x, start_y, vx, r]t_x, start_y, vx, r]
        self.dynamic_obs_params = jnp.array([
            [4.0, 1.0, 0.5, 0.5],
            [8.0, 3.0, 0.6, 0.5],
            [12.0, 1.0, 0.7, 0.5],
            [16.0, 3.0, 0.8, 0.5]
        ])
        
        # Objectives: [offset_x, abs_y, r, t_idx, ref_obs_idx]
        self.subgoals = jnp.array([
            [2.0, 3.0, 0.5, 0, 0],
            [2.0, 1.0, 0.5, 1, 1],
            [2.0, 3.0, 0.5, 2, 2],
            [2.0, 1.0, 0.5, 3, 3]
        ])

    def _setup_local_minima_trap(self):
        """
        [New] Local Minimum Trap Scene (Cul-de-sac)
        
Description:
        - Obstacles: A large static rectangular wall that stands between the start and end.
        - Purpose: To verify that the algorithm can plan a path to detour and reach instead of hitting the wall.
        """
        # 1. Rectangular obstacles
        # Format changed to: [center_x, center_y, width, height]
        # The center is at the origin, 15 meters wide and 5 meters high
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