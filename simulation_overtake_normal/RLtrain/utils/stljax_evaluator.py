import jax
import jax.numpy as jnp
from stljax.formula import Predicate, DifferentiableAlways, DifferentiableEventually
from functools import partial
from config import TGPOConfig

class STLEvaluator:
    def __init__(self, obstacles_params, subgoals):
        
        self.obs_params = obstacles_params
        self.subgoals = subgoals
        self.T = int(TGPOConfig.TIME_SCALE) 
        self.num_goals = subgoals.shape[0]
        self.dt = TGPOConfig.DT
        
        
        times = jnp.arange(self.T) * self.dt # (T,) -> [0.0, 0.2, ..., 19.8]
        
        # x(t) = x0 + vx * t
        obs_x_traj = self.obs_params[None, :, 0] + times[:, None] * self.obs_params[None, :, 2]
        obs_y_traj = self.obs_params[None, :, 1] 
        
        obs_y_traj = jnp.tile(obs_y_traj, (self.T, 1))
        
        
        self.obs_traj = jnp.stack([obs_x_traj, obs_y_traj], axis=-1)
        
        
        def dynamic_safety(states):
            """states: (T, 4)"""
            pos = states[:, :2] # (T, 2)
            
            diff = pos[:, None, :] - self.obs_traj
            dist = jnp.linalg.norm(diff, axis=-1)
            
            return jnp.min(dist - 1.1, axis=-1)

        def workspace_margin(states):
            
            y = states[:, 1]
            return jnp.minimum(y - 0.6, 3.4 - y)

        self.safe_pred = Predicate("safety", predicate_function=dynamic_safety)
        self.safety_formula = DifferentiableAlways(self.safe_pred > 0.0)
        
        self.boundary_pred = Predicate("boundary", predicate_function=workspace_margin)
        self.boundary_formula = DifferentiableAlways(self.boundary_pred > 0.0)

        
        def lane_keeping(states):
            
            y = states[:, 1]
            return jnp.minimum(y - 0.6, 1.4 - y)

        self.lane_pred = Predicate("lane_keep", predicate_function=lane_keeping)
        self.lane_formula = DifferentiableAlways(self.lane_pred > 0.0)
        
        
        lk_start_abs, lk_end_abs = TGPOConfig.LANE_KEEP_WINDOW
        self.lane_t_start = lk_start_abs / self.T
        self.lane_t_end = lk_end_abs / self.T

        
        self.goal_formulas = []
        self.time_windows = [] 
        
        custom_windows_abs = TGPOConfig.CUSTOM_TIME_WINDOWS
        
        for i in range(self.num_goals):
            
            offset_x = subgoals[i, 0]
            goal_y = subgoals[i, 1]
            goal_r = subgoals[i, 2]
            ref_idx = int(subgoals[i, 4])
            
            
            ref_traj = self.obs_traj[:, ref_idx, :]
            
            
            def goal_pred_func(states, ox=offset_x, oy=goal_y, r=goal_r, rt=ref_traj):
               
                target_x_t = rt[:, 0] + ox
                
                dx = states[:, 0] - target_x_t
                dy = states[:, 1] - oy
                dist = jnp.sqrt(dx**2 + dy**2)
                return r - dist
                
            pred = Predicate(f"goal_{i}", predicate_function=goal_pred_func)
            
           
            formula = DifferentiableEventually(pred > 0.0)
            self.goal_formulas.append(formula)
            
            
            start_abs, end_abs = custom_windows_abs[i]
            
            self.time_windows.append((start_abs / self.T, end_abs / self.T))

    def evaluate_single(self, trajectory):
        
        safe_score = self.safety_formula.robustness(trajectory, t_start=0.0, t_end=1.0, scale=10.0)
        bound_score = self.boundary_formula.robustness(trajectory, t_start=0.0, t_end=1.0, scale=10.0)
        
        
        lane_score = self.lane_formula.robustness(
            trajectory, 
            t_start=self.lane_t_start, 
            t_end=self.lane_t_end, 
            scale=10.0
        )
        
        
        goal_scores = []
        for i, formula in enumerate(self.goal_formulas):
            t_start, t_end = self.time_windows[i]
            
            score = formula.robustness(
                trajectory, 
                t_start=t_start, 
                t_end=t_end, 
                scale=10.0
            )
            goal_scores.append(score)
            
        
        all_scores = jnp.array([safe_score, bound_score, lane_score] + goal_scores)
        return jnp.min(all_scores)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, batch_trajectories):
        """
        
        Args:
            batch_trajectories: (Batch, T, 4)
        """
        return jax.vmap(self.evaluate_single)(batch_trajectories)