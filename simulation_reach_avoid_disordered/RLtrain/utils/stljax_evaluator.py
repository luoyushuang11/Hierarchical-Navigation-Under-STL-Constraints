import jax
import jax.numpy as jnp
from stljax.formula import Predicate, DifferentiableAlways, DifferentiableEventually
from functools import partial
from config import TGPOConfig

class STLEvaluator:
    def __init__(self, obstacles, subgoals):
        """
        Batch Robustness Calculator Based on Stljax (Using the Differentiable Operator)
        
        Args:
            obstacles: (N_obs, 3) [x, y, r]
            subgoals: (N_goals, 4) [x, y, r, time_idx]
        """
        self.obstacles = obstacles
        self.subgoals = subgoals
        self.workspace_radius = TGPOConfig.X_LIMIT
        self.T = int(TGPOConfig.TIME_SCALE)
        self.num_goals = subgoals.shape[0]
        
        print(f"[STL Init] T={self.T}, Goals={self.num_goals}, Obstacles={obstacles.shape[0]}")
        
        # ==========================================
        # 1. Define predicate logic
        # ==========================================
        
        # --- Predicate A： （Safety） ---
        def min_obstacle_margin(states):
            """states: (T, state_dim) -> (T,)"""
            pos = states[:, :2]
            diff = pos[:, None, :] - self.obstacles[None, :, :2]
            dist = jnp.linalg.norm(diff, axis=-1)
            margin = dist - self.obstacles[None, :, 2]-0.2
            return jnp.min(margin, axis=-1)
        # --- [NEW] Predicate B: Workspace Boundary ---
        def workspace_margin(states):
            
            pos = states[:, :2]
       
            dist_to_origin = jnp.linalg.norm(pos, axis=-1)
           
            return self.workspace_radius - dist_to_origin-0.2

        # --- Predicate B: Reach the target ---
        self.goal_predicates = []
        for i in range(self.num_goals):
            gx, gy, gr = self.subgoals[i, :3]
            def goal_pred(states, gx=gx, gy=gy, gr=gr):
                pos = states[:, :2]
                dist = jnp.linalg.norm(pos - jnp.array([gx, gy]), axis=-1)
                return gr - dist
            pred = Predicate(f"goal_{i}", predicate_function=goal_pred)
            self.goal_predicates.append(pred)

        # ==========================================
        # 2. Build STL formulas (with the Differentiable version!)
        # ==========================================
        
        # Safety: DifferentiableAlways(Safe > 0) for entire trajectory
        self.safe_pred = Predicate("safety", predicate_function=min_obstacle_margin)
        self.safety_formula = DifferentiableAlways(self.safe_pred > 0.0)
        self.workspace_pred = Predicate("workspace_safety", predicate_function=workspace_margin)
        self.workspace_formula = DifferentiableAlways(self.workspace_pred > 0.0)
        
        # Reachability: DifferentiableEventually for each goal
        self.goal_formulas = []
        for i in range(self.num_goals):
            formula = DifferentiableEventually(self.goal_predicates[i] > 0.0)
            self.goal_formulas.append(formula)
        
        # Calculate the time window
        custom_windows_abs = TGPOConfig.CUSTOM_TIME_WINDOWS
        self.time_windows = []
        for i in range(self.num_goals):
            start_abs, end_abs = custom_windows_abs[i]
            # Normalization time required to convert to stljax [0, 1]
            t_start_norm = start_abs / self.T
            t_end_norm = end_abs / self.T
            self.time_windows.append((t_start_norm, t_end_norm))
            #t_start_abs = int(t_start_norm * self.T)
            #t_end_abs = int(t_end_norm * self.T)
            #print(f"  Goal {i}: window [{t_start_abs}, {t_end_abs}), target=({self.subgoals[i, 0]:.1f}, {self.subgoals[i, 1]:.1f})")
        
       

    def evaluate_single(self, trajectory):
        """
        Evaluating a single trajectory (T, state_dim)
        Use the t_start, t_end parameter of DifferentiableAlways/Eventual
        """
        # 1. Full safety check
        safety_score = self.safety_formula.robustness(
            trajectory,
            t_start=0.0,
            t_end=1.0,
            scale=10.0  # Smooth parameters, the larger the closer it is to the true min/max
        )
        # [NEW] Workspace Boundary Check (Full)
        boundary_score = self.workspace_formula.robustness(
            trajectory, t_start=0.0, t_end=1.0, scale=10.0
        )
        
        # 2. Check the target arrival in different periods
        goal_scores = []
        for i, (t_start, t_end) in enumerate(self.time_windows):
            score = self.goal_formulas[i].robustness(
                trajectory,
                t_start=t_start,
                t_end=t_end,
                scale=10.0
            )
            goal_scores.append(score)
        
        # 3. min(safety, goal1, goal2, ...)
        all_scores = jnp.array([safety_score, boundary_score] + goal_scores)
        return jnp.min(all_scores)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, batch_trajectories):
        """
        Batch evaluation (Batch, T, state_dim)
        """
        return jax.vmap(self.evaluate_single)(batch_trajectories)