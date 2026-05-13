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
            [2.5,  -2.5, 1.0, 0],  
            [2.0,  3.0, 1.0, 1],  
            [-2.5, 5.0, 1.0, 2]  
        ])

        

        """
        # 9. Obstacle orderly distribution scene
        self.obstacles = jnp.array([
        [-2.5, 0.0, 1.0], [ 6.0, -2.5, 1.0], [ 2.5, 0.0, 1.0],
        [ 2.5, 5.0, 1.0], [ -2.5, -2.5, 1.0], [ 6.0, 2.5, 1.0],
        [ 6.0, 5.0, 1.0], [ 6.0, 0.0, 1.0], [ -2.5, 2.5, 1.0],

        ])"""
        



        """
        # 13 Scenes of orderly distribution of obstacles
        
        self.obstacles = jnp.array([
        [-2.5, 0.0, 0.75], [ 6.0, -2.5, 0.75], [ 2.5, 0.0, 0.75],
        [ 2.5, 5.0, 0.75], [ -2.5, -2.5, 0.75], [ 6.0, 2.5, 0.75],
        [ 6.0, 5.0, 0.75], [ 6.0, 0.0, 0.75], [ -2.5, 2.5, 0.75],
        [ 0.0, -2.5, 0.75], [ 0.0, 0.0, 0.75], [ 0.0, 2.5, 0.75], [ 0.0, 5.0, 0.75],

        ])"""
        
        '''
        self.subgoals = jnp.array([
            [2.5,  -2.5, 1.0, 0], 
            [2.5,  2.5, 1.0, 1],  
            [-2.5, 5.0, 1.0, 2]   
        ])'''

    def _setup_multi_goal_sequence(self):
       
        self.obstacles = jnp.array([
            [-3.0, -3.0, 0.5], [-3.0, 0.0, 0.5], [-3.0, 3.0, 0.5],
            [ 0.0, -3.0, 0.5], [ 0.0, 0.0, 0.5], [ 0.0, 3.0, 0.5],
            [ 3.0, -3.0, 0.5], [ 3.0, 0.0, 0.5], [ 3.0, 3.0, 0.5],
        ])
        '''self.obstacles = jnp.array([
            [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08],
            [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08],
            [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08], [ 0.0, 0.0, 0.08],
        ])'''
        
       
        self.subgoals = jnp.array([
           
            [-4.0, 4.0, 0.6, 0], 
            
           
            [ 4.0, -4.0, 0.6, 1], 
            
  
            [ 0.0, 4.2, 0.6, 2]  
        ])


def get_scenario(name="reach_avoid_simple"):
    return TaskScenario(name)