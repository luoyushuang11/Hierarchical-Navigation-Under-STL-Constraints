import jax
import jax.numpy as jnp
from stljax.formula import Predicate, DifferentiableAlways, DifferentiableEventually
from functools import partial
from config import TGPOConfig



# --- 1. Define dynamic obstacle parameters (consistent with logic.py) ---
# Format: [x_fixed, y_min, y_max, speed, radius, phase_offset]
# One way L = 9.0
# Obs 1: offset=0.0 (from bottom 0.5)
# Obs 2: offset=9.0 (from top 9.5)
DYN_OBS_PARAMS = jnp.array([
    
    [6.0, 0.5, 9.5, 0.4, 0.5, 0.0]

])

class STLEvaluator:
    def __init__(self, obstacles, subgoals):
        """
        [Pre-calculated version] STLEvaluator
        All trajectories of dynamic obstacles are directly calculated during the initialization stage, greatly improving the evaluation efficiency.
        """
        self.obstacles = obstacles
        self.subgoals = subgoals
        self.T = int(TGPOConfig.TIME_SCALE)
        self.dt = TGPOConfig.DT
        self.num_goals = subgoals.shape[0]
        
        self.ws_limits = jnp.array([12.0, 10.0])
        self.robot_radius = 0.2
        
        
        times = (jnp.arange(self.T) * self.dt)[:, None] 
        
       
        # DYN_OBS_PARAMS: (N, 6)
        x_fixed = DYN_OBS_PARAMS[None, :, 0]
        y_min   = DYN_OBS_PARAMS[None, :, 1]
        y_max   = DYN_OBS_PARAMS[None, :, 2]
        speed   = DYN_OBS_PARAMS[None, :, 3]
       
        offset  = DYN_OBS_PARAMS[None, :, 5]
        
        # C. (Ping-Pong Logic)
        length = y_max - y_min
        cycle = 2.0 * length
        
        
        dist = times * speed + offset
        mod = dist % cycle
        
        
        y_traj = jnp.where(
            mod <= length,
            y_min + mod,
            y_max - (mod - length)
        ) # (T, N)
        
       
        x_traj = jnp.tile(x_fixed, (self.T, 1))
        
       
        # self.dyn_obs_traj[t, i] is the coordinates of the ith obstacle at time t
        self.dyn_obs_traj = jnp.stack([x_traj, y_traj], axis=-1)
        
        
        self.dyn_radii = DYN_OBS_PARAMS[None, :, 4]

        print(f"[STL Init] Pre-calculated dynamic trajectories shape: {self.dyn_obs_traj.shape}")

        # ==========================================
        # 2. Define STL predicates and formulas
        # ==========================================
        
      
        def dyn_safety_func(states):
            """
            states: (T, state_dim)
            """
            pos = states[:, :2] # (T, 2)
            
            # Broadcast calculation distance: 
            # pos expands to (T, 1, 2) - obs expands to (T, N, 2)
            # diff shape: (T, N, 2)
            diff = pos[:, None, :] - self.dyn_obs_traj
            
            # Calculating Euclidean Distance (T, N)
            dist_matrix = jnp.linalg.norm(diff, axis=-1)
            
            
            sdf_matrix = dist_matrix - (self.robot_radius + self.dyn_radii)
            
            # Take min for N obstacles (find the most dangerous one at the same moment) -> (T,)
            return jnp.min(sdf_matrix, axis=-1)

        self.dyn_safe_pred = Predicate("dyn_safety", predicate_function=dyn_safety_func)
        self.dyn_safety_formula = DifferentiableAlways(self.dyn_safe_pred > 0.0)

        # --- B. Static Obstacle Safety (Maintaining Original Logic) ---
        def static_safety_func(states):
            pos = states[:, :2] # (T, 2)
            pos_exp = pos[:, None, :] # (T, 1, 2)
            centers = self.obstacles[None, :, :2] # (1, M, 2)
            half_sizes = self.obstacles[None, :, 2:] / 2.0
            
            d = jnp.abs(pos_exp - centers) - half_sizes
            dist_outside = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
            max_d = jnp.maximum(d[..., 0], d[..., 1])
            dist_inside = jnp.minimum(max_d, 0.0)
            return jnp.min(dist_outside + dist_inside, axis=-1)

        self.static_safe_pred = Predicate("static_safety", predicate_function=static_safety_func)
        self.static_safety_formula = DifferentiableAlways(self.static_safe_pred > self.robot_radius)

        # --- C. Workspace Boundaries ---
        def workspace_func(states):
            pos = states[:, :2]
            d_min = pos - 0.0
            d_max = self.ws_limits - pos
            return jnp.minimum(jnp.min(d_min, axis=-1), jnp.min(d_max, axis=-1))

        self.ws_pred = Predicate("workspace", predicate_function=workspace_func)
        self.ws_formula = DifferentiableAlways(self.ws_pred > self.robot_radius)

        # --- D. Goal Reachability (Eventually) ---
        self.goal_formulas = []
        self.time_windows = []
        
        custom_windows_abs = TGPOConfig.CUSTOM_TIME_WINDOWS
        for i in range(self.num_goals):
            gx, gy, gr, _ = self.subgoals[i]
            
            def goal_func(states, _gx=gx, _gy=gy, _gr=gr):
                dist = jnp.linalg.norm(states[:, :2] - jnp.array([_gx, _gy]), axis=-1)
                return _gr - dist # >0 if reached
            
            pred = Predicate(f"goal_{i}", predicate_function=goal_func)
            
            
            formula = DifferentiableEventually(pred > 0.0)
            self.goal_formulas.append(formula)
            
            
            s, e = custom_windows_abs[i]
            self.time_windows.append((s / self.T, e / self.T))

    def evaluate_single(self, trajectory):
        
        # 1. Static security
        score_static = self.static_safety_formula.robustness(
            trajectory, t_start=0.0, t_end=1.0, scale=10.0
        )
        
        # 2. Dynamic safety (directly using pre-calculated trajectories, very fast)
        score_dyn = self.dyn_safety_formula.robustness(
            trajectory, t_start=0.0, t_end=1.0, scale=10.0
        )
        
        # 3. Workspace safety
        score_ws = self.ws_formula.robustness(
            trajectory, t_start=0.0, t_end=1.0, scale=10.0
        )
        
        # 4. Goal reachability
        goal_scores = []
        for i, formula in enumerate(self.goal_formulas):
            ts, te = self.time_windows[i]
            s = formula.robustness(trajectory, t_start=ts, t_end=te, scale=10.0)
            goal_scores.append(s)
            
        # 5. (min)
        all_scores = jnp.array([score_static, score_dyn, score_ws] + goal_scores)
        #all_scores = jnp.array([score_static, score_ws] + goal_scores)
        return jnp.min(all_scores)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, batch_trajectories):
        return jax.vmap(self.evaluate_single)(batch_trajectories)