import sys
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import random, lax


import animation 


matplotlib.use('Agg')

# ==========================================
# 0. Module import is isolated from the environment
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- Loading RLtrain ---
rltrain_path = os.path.join(current_dir, "RLtrain")
sys.path.insert(0, rltrain_path)
try:
    if 'config' in sys.modules: del sys.modules['config']
    from config import TGPOConfig
    from train import TGPOTrainer
    from planning.uniform import sample_uniform_times
    from planning.mcmc import metropolis_hastings_sampling
    from envs.dynamics import unicycle_step as rl_dynamics_step
    print("✅ The RLtrain module loads successfully。")
except ImportError as e:
    print(f"❌ The RLtrain module failed to load: {e}")
    sys.exit(1)
finally:
    if rltrain_path in sys.path: sys.path.remove(rltrain_path)
    if 'config' in sys.modules: del sys.modules['config']

# --- Loading MPPI ---
mppi_path = os.path.join(current_dir, "MPPI")
sys.path.insert(0, mppi_path)
try:
    import config as mppi_cfg
    from model import jax_dynamics as sim_dynamics_step 
    from cbf_functions_rl import h_x_jax
    print("✅ The MPPI module loads successfully。")
except ImportError as e:
    print(f"❌ The MPPI module failed to load: {e}")
    sys.exit(1)

# ==========================================
# 1. Global configuration
# ==========================================
DT_RL = TGPOConfig.DT       # 0.2s
DT_MPPI = mppi_cfg.dt       # 0.02s
STEPS_PER_RL = int(DT_RL / DT_MPPI) # 10

RL_HORIZON = 20             # 6.0s
MPPI_HORIZON = mppi_cfg.N   # 6.0s

N_SAMPLES = mppi_cfg.n_samples
R_MPPI = jnp.array(mppi_cfg.R)
Q_TRACK = jnp.diag(jnp.array([30.0, 30.0, 10.0, 0.0]))

# Prefer the sampling window in Config
try:
    LOGIC_WINDOWS = jnp.array(TGPOConfig.SAMPLING_WINDOWS)
except AttributeError:
    LOGIC_WINDOWS = jnp.array(TGPOConfig.CUSTOM_TIME_WINDOWS)

print(f"🔒 Logical decision window: \n{LOGIC_WINDOWS}")

# ==========================================
# 2. Minimalist window logic (Fixed Indexing)
# ==========================================
@jax.jit
def window_based_update(aug_state, next_phys_state, subgoals_array, obstacles_array, windows_array, time_vars):
    """
    Minimalist logic updates
    """
    tau = aug_state[4]
    p = aug_state[6].astype(jnp.int32)
    r = aug_state[7].astype(jnp.int32)
    chi = aug_state[8]
    
    # 1. Time passes naturally
    tau_next = tau + 1.0
    
    # 2. Get current goals
    max_p = subgoals_array.shape[0]
    safe_p = jnp.minimum(p, max_p - 1)
    
    curr_goal = subgoals_array[safe_p]
    goal_pos = curr_goal[:2]
    goal_radius = curr_goal[2]
    t_idx = curr_goal[3].astype(jnp.int32)
    
    # 3. Get the Config window [t_min, t_max]
    win_start = windows_array[t_idx, 0]
    win_end   = windows_array[t_idx, 1]
    time_var = time_vars[t_idx]

    # 4. Judgment conditions
    # A. Space arrives
    dist = jnp.linalg.norm(next_phys_state[:2] - goal_pos)
    is_spatial_reached = dist <= goal_radius
    
    # B. The time is in the window
    is_time_valid = (tau_next >= time_var) & (tau_next <= win_end)
    
    # 5. Status jumps
    all_completed = (p >= max_p)
    cond_finish = (r != 2) & is_spatial_reached & is_time_valid & (~all_completed)
    cond_reset = (r == 2)
    
    r_next = jnp.where(cond_reset, 0.0,
                jnp.where(cond_finish, 2.0, r))
    
    # 6. Progress updates
    p_increment = (r_next == 2.0).astype(jnp.float32)
    p_next = jnp.minimum(p + p_increment, max_p)
    p_prev_next = p
    
    # 7. Safety score
    diff = next_phys_state[:2] - obstacles_array[:, :2]
    dists_obs = jnp.linalg.norm(diff, axis=-1) - obstacles_array[:, 2]
    is_safe = (jnp.min(dists_obs) > 0.0).astype(jnp.float32)
    chi_next = chi * is_safe
    
    return jnp.array([tau_next, p_prev_next, p_next, r_next, chi_next])

# ==========================================
# 3. Controller and core algorithms
# ==========================================
class HierarchicalController:
    def __init__(self, model_path):
        self.trainer = TGPOTrainer(scenario_name="reach_avoid_simple")
        self.trainer.load_checkpoint(model_path)
        self.actor_apply = self.trainer.actor.apply
        self.actor_params = self.trainer.actor_state.params
        self.critic_apply = self.trainer.critic.apply
        self.critic_params = self.trainer.critic_state.params
        self.subgoals = self.trainer.scenario.subgoals 
        self.obstacles = self.trainer.scenario.obstacles
        self._compile_functions()

    def _compile_functions(self):
        # Modify the predict_loop to return the action
        def predict_loop(carry, _):
            s, rng, t_vars = carry
            rng, subkey = jax.random.split(rng)
            mean, _ = self.actor_apply(self.actor_params, s, t_vars)
            a = jnp.tanh(mean) 
            curr_p = s[..., :4]
            next_p = jax.vmap(rl_dynamics_step)(curr_p, a)
            s_next = s.at[..., :4].set(next_p) 
            # Return: (carry), (phys_state, action)
            return (s_next, rng, t_vars), (next_p, a)

        def predict_fn(init_state, time_vars, rng):
            # scan Return (traj, actions)
            _, (pred_traj, pred_actions) = jax.lax.scan(
                predict_loop, (init_state, rng, time_vars), None, length=RL_HORIZON
            )
            
            traj_out = jnp.swapaxes(pred_traj, 0, 1)[0]
            act_out = jnp.swapaxes(pred_actions, 0, 1)[0]
            return traj_out, act_out
            
        self.predict_fn = jax.jit(predict_fn)
        
        def score_fn(params, obs, times): return self.critic_apply(params, obs, times)
        self.mcmc_score_fn = score_fn

    def sample_initial_time(self, init_state, rng_key):
        rng_key, k_uni, k_mcmc = jax.random.split(rng_key, 3)
        init_times = sample_uniform_times(k_uni, 1, self.subgoals.shape[0])
        if init_state.ndim == 1: init_state = init_state[None, :]
        optimized_times = metropolis_hastings_sampling(
            k_mcmc, self.mcmc_score_fn, self.critic_params, init_state, init_times
        )
        return optimized_times[0], rng_key 

    
    def get_reference_trajectory_and_actions(self, full_state, time_vars, rng):
        if full_state.ndim == 1: full_state = full_state[None, :]
        if time_vars.ndim == 1: time_vars = time_vars[None, :]
        return self.predict_fn(full_state, time_vars, rng)

    
    def get_reference_trajectory(self, full_state, time_vars, rng):
        traj, _ = self.get_reference_trajectory_and_actions(full_state, time_vars, rng)
        return traj

@jax.jit
def interpolate_to_mppi(current_phys, coarse_traj):
    t_coarse = jnp.linspace(0.0, RL_HORIZON * DT_RL, RL_HORIZON + 1)
    traj_points = jnp.concatenate([current_phys[None, :], coarse_traj], axis=0)
    t_fine = jnp.linspace(DT_MPPI, MPPI_HORIZON * DT_MPPI, MPPI_HORIZON)
    def interp_dim(y_vals): return jnp.interp(t_fine, t_coarse, y_vals)
    return jax.vmap(interp_dim)(traj_points.T).T


@jax.jit
def mppi_optimize_step(current_state, U_prev, rng_key, ref_traj, current_time, time_vars, real_reach_flags,noise_scale=1.0, weight_scale=1.0):
    
   
    Q_TRACK_SCALED = Q_TRACK * weight_scale

    def rollout_scan(carry, inputs):

        cost_acc, state, t_now, sim_flags = carry
        u_nominal, noise, ref_state = inputs
        
   
        u = u_nominal + noise * noise_scale
        
        u = u.at[0].set(jnp.clip(u[0], -mppi_cfg.MAX_OMEGA, mppi_cfg.MAX_OMEGA))
        u = u.at[1].set(jnp.clip(u[1], -mppi_cfg.MAX_ACCEL, mppi_cfg.MAX_ACCEL))
        next_state = sim_dynamics_step(state, u, dt=DT_MPPI)
        t_next = t_now + DT_MPPI
        c_ctrl = jnp.dot(u, jnp.dot(R_MPPI, u))
        

        c_track = 1.0 * jnp.dot(next_state - ref_state, jnp.dot(Q_TRACK_SCALED, next_state - ref_state))
        
  
        h_val, next_sim_flags = h_x_jax(next_state, t_next, time_vars, sim_flags)
        c_cbf = 1e4 * jnp.clip(-h_val, 0.0, jnp.inf)
  
        return (cost_acc + c_ctrl + c_track + c_cbf, next_state, t_next, next_sim_flags), next_state

    def single_rollout(subkey):
     
        noise = jax.random.normal(subkey, (MPPI_HORIZON, 2)) * jnp.array(mppi_cfg.NOISE_SIGMA)

        init_carry = (0.0, current_state, current_time, real_reach_flags)
        scan_inputs = (U_prev, noise, ref_traj)
      
        (final_cost, _, _, _), rollout_states = lax.scan(rollout_scan, init_carry, scan_inputs)
        return final_cost, rollout_states

    def single_rollout_full(subkey):
        noise = jax.random.normal(subkey, (MPPI_HORIZON, 2)) * jnp.array(mppi_cfg.NOISE_SIGMA)
  
        init_carry = (0.0, current_state, current_time, real_reach_flags)
        scan_inputs = (U_prev, noise, ref_traj)
        
        def scan_fn_u(carry, inputs):
  
            cost_acc, state, t_now, sim_flags = carry
            u_nominal, noise, ref_state = inputs
            
   
            u = u_nominal + noise * noise_scale
            
            u = u.at[0].set(jnp.clip(u[0], -mppi_cfg.MAX_OMEGA, mppi_cfg.MAX_OMEGA))
            u = u.at[1].set(jnp.clip(u[1], -mppi_cfg.MAX_ACCEL, mppi_cfg.MAX_ACCEL))
            next_state = sim_dynamics_step(state, u, dt=DT_MPPI)
            t_next = t_now + DT_MPPI
            c_ctrl = jnp.dot(u, jnp.dot(R_MPPI, u))
            c_track = 1.0 * jnp.dot(next_state - ref_state, jnp.dot(Q_TRACK_SCALED, next_state - ref_state))
    
            h_val, next_sim_flags = h_x_jax(next_state, t_next, time_vars, sim_flags)
            c_cbf = 1e4 * jnp.clip(-h_val, 0.0, jnp.inf)
    
            return (cost_acc + c_ctrl + c_track + c_cbf, next_state, t_next, next_sim_flags), (next_state, u)


        (final_cost, _, _, _), (rollout_states, rollout_us) = lax.scan(scan_fn_u, init_carry, scan_inputs)
        return final_cost, rollout_states, rollout_us

    rng_keys = jax.random.split(rng_key, N_SAMPLES)
    costs, trajectories, sequences = jax.vmap(single_rollout_full)(rng_keys)
    
    min_cost = jnp.min(costs)
    weights = jax.nn.softmax(-1.0 * (costs - min_cost) / mppi_cfg.Temperature)
    best_U = jnp.sum(weights[:, None, None] * sequences, axis=0)
    next_U = jnp.roll(best_U, shift=-1, axis=0).at[-1].set(best_U[-2])
    
    return best_U[0], next_U, trajectories

# ==========================================
# 4. Main program
# ==========================================
def run_final_merged_ref_virtual():
  
    save_dir = os.path.join(current_dir, "rviz")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 The visualization results are saved to: {save_dir}")

    SIM_DURATION = 60.0
    TOTAL_STEPS = int(SIM_DURATION / DT_MPPI)
    
    print(f"\n{'='*60}")
    print(f"🚀 Simulation of two-layer hierarchical control (Focus Warmup + Reach Avoid)")
    print(f"{'='*60}")
    
    model_path = os.path.join(current_dir, "RLtrain/tgpo_model_3goals_s_6.msgpack")
    controller = HierarchicalController(model_path)
    
    rng = jax.random.PRNGKey(int(time.time()))
    rng, k_init, k_mcmc = jax.random.split(rng, 3)
    k_x, k_y, k_theta = jax.random.split(k_init, 3)

    # Initial state: Reach Avoid (x, y 在 -6.0 ~ -5.0)
    start_x = jax.random.uniform(k_x, minval=-6.0, maxval=-5.0)
    start_y = jax.random.uniform(k_y, minval=-6.0, maxval=-5.0)
    start_theta = jax.random.uniform(k_theta, minval=-jnp.pi, maxval=jnp.pi)
    curr_phys = jnp.array([start_x, start_y, start_theta, 0.0])
    print(f"   Initial state: x={start_x:.4f}, y={start_y:.4f}, theta={start_theta:.4f} rad")
    curr_logic = jnp.array([0., 0., 0., 0., 1.])
    curr_full = jnp.concatenate([curr_phys, curr_logic])
    
    time_vars, _ = controller.sample_initial_time(curr_full, k_mcmc)
    print(f"🔒 RL Plan time variables: {time_vars}")

    # === [NEW] Initialize 3 target area fuse flag bits in the real world ===
    curr_real_flags = jnp.array([False, False, False])
    
     # === [Step 1] Initialize MPPI (ZOH) === with RL action sequences
    print("\n⚡ [Init] ZOH: Initializing U_mppi with RL action sequences...")
    
    rng, k_plan = jax.random.split(rng)
    # 1. Get RL rough tracks and actions
    current_coarse_ref, coarse_actions = controller.get_reference_trajectory_and_actions(curr_full, time_vars, k_plan)
    
    # 2. [Critical fix] fine_ref (interpolated reference track) must be calculated first! You may have mistakenly deleted this line before
    fine_ref = interpolate_to_mppi(curr_phys, current_coarse_ref)  # <--- Make up for this line

    # 3. [Fix 1] De-normalization
    # Assuming the RL output is [-1, 1], we need to map it to the physical limit
    # Please confirm the specific physical limit values according to the definitions in mppi_cfg
    # The order here must be consistent with your MPPI config definition, usually [Omega, Accel] or [Accel, Omega]
    action_scale = jnp.array([mppi_cfg.MAX_OMEGA, mppi_cfg.MAX_ACCEL]) 
    
  
    coarse_actions_physical = coarse_actions * action_scale
    
 
    k_ratio = int(DT_RL / DT_MPPI) # 10
    
  
    fine_actions_seeded = jnp.repeat(coarse_actions_physical, repeats=k_ratio, axis=0)
    
    if fine_actions_seeded.shape[0] >= MPPI_HORIZON:
        U_mppi = fine_actions_seeded[:MPPI_HORIZON]
    else:
        pad_len = MPPI_HORIZON - fine_actions_seeded.shape[0]
        last_action = fine_actions_seeded[-1:]
        padding = jnp.tile(last_action, (pad_len, 1))
        U_mppi = jnp.concatenate([fine_actions_seeded, padding], axis=0)
        
    print(f"   U_mppi Seed Shape: {U_mppi.shape}")
        

    # Historical data collection
    log_pos, log_cbf = [], []
    all_ref_trajs = []      
    all_mppi_samples = []   
    
    latencies = []
    current_coarse_ref = None
    task_completed_flag = False
    tau_offset = 0.0
    
    # === (Focus Warm-up) ===
    print("\n🔥 [Init] Warm-up:  (Low Noise, High Weight)...")
    
    WARMUP_STEPS = 5
    for w in range(WARMUP_STEPS):
        rng, k_mppi = jax.random.split(rng)
   
        _, U_mppi, _ = mppi_optimize_step(
            curr_phys, U_mppi, k_mppi, fine_ref, 0.0, time_vars, curr_real_flags,
            noise_scale=0.1, weight_scale=10.0
        )
    
    print(f"✅ Preheating is complete. The control sequence is aligned。")
    
    print(f"\n⚡ Start the simulation (60s)...")
    total_sim_start = time.perf_counter()
    for step in range(TOTAL_STEPS):
        loop_start = time.perf_counter()
        t_now = step * DT_MPPI
        rng, k_plan, k_mppi = jax.random.split(rng, 3)
        
        # --- A. RL Reference (5Hz) ---
        if step % STEPS_PER_RL == 0:
            virtual_tau = curr_full[4] #- tau_offset
            virtual_full = curr_full.at[4].set(virtual_tau)
            # No action is needed in the simulation, only a trajectory
            current_coarse_ref, _ = controller.get_reference_trajectory_and_actions(virtual_full, time_vars, k_plan)
            
        if step % STEPS_PER_RL == 0:
            all_ref_trajs.append(np.array(current_coarse_ref[:, :2]))
            
        if current_coarse_ref is None:
             current_coarse_ref, _ = controller.get_reference_trajectory_and_actions(curr_full, time_vars, k_plan)

        fine_ref = interpolate_to_mppi(curr_phys, current_coarse_ref)
        
        # --- B. MPPI Control (50Hz) ---
        # Normal operation: Use default noise_scale=1.0, weight_scale=1.0 (do not pass parameters)
        # [Modified] Normal operation: Added time_vars, curr_real_flags parameters
        action, U_mppi, mppi_trajs_jax = mppi_optimize_step(curr_phys, U_mppi, k_mppi, fine_ref, t_now, time_vars, curr_real_flags)

        # Collect samples
        mppi_trajs_np = np.array(mppi_trajs_jax[:100, :, :2])
        all_mppi_samples.append(mppi_trajs_np)

        # ====================================================
        # === [ADDED] Absolute braking mechanism after mission completion (hard override) ===
        # ====================================================
        if task_completed_flag:
            action = jnp.array([0.0, 0.0])          
            curr_phys = curr_phys.at[3].set(0.0)     
        
        # --- C. Physics Update ---
        next_phys = sim_dynamics_step(curr_phys, action, dt=DT_MPPI)
        # === [NEW] D. Updating real-world latch state ===
        # Use real next_phys calculations to update real-world flags
        _, curr_real_flags = h_x_jax(next_phys, t_now + DT_MPPI, time_vars, curr_real_flags)
        
        # --- D. Logical Update ---
        should_inc_tau = ((step + 1) % STEPS_PER_RL == 0)
        
        if should_inc_tau:
            next_logic = window_based_update(
                curr_full, next_phys, 
                controller.subgoals, controller.obstacles, LOGIC_WINDOWS, time_vars
            )
            
            curr_p = int(curr_full[6])
            next_p = int(next_logic[2])
            
            if next_p != curr_p:
                print(f"✨ [Switch] Goal {curr_p} -> {next_p} at T={t_now:.2f}s")
                plan_tau_finished = time_vars[curr_p] 
                tau_offset = curr_full[4] #- jnp.round(plan_tau_finished)
              
            
            if next_p >= 3 and not task_completed_flag:
                print(f"🎉 The task is completed at T={t_now:.2f}s!")
                task_completed_flag = True
        else:
            next_logic = curr_full[4:]

        curr_full = jnp.concatenate([next_phys, next_logic])
        curr_phys = next_phys
        
        curr_full.block_until_ready()
        latencies.append(time.perf_counter() - loop_start)
        
        log_pos.append(curr_phys)
        # [Modified] Extract the first value from the h_x_jax return tuple for logging
        h_val_for_log, _ = h_x_jax(curr_phys, t_now, time_vars, curr_real_flags)
        log_cbf.append(h_val_for_log)
        
        if step % 50 == 0:
            valid_lats = latencies[1:] if len(latencies) > 1 else latencies
            freq = 1.0 / np.mean(valid_lats) if len(valid_lats) > 0 else 0.0
            status = "DONE" if task_completed_flag else f"G{int(curr_full[6])}"
            print(f"Step {step:04d} | Status={status} | Hz={freq:.1f}")

    # Simulation end statistics
    total_sim_end = time.perf_counter()
    total_wall_time = total_sim_end - total_sim_start
    total_compute_time = np.sum(latencies)
    avg_latency = np.mean(latencies)
    
    print(f"\n{'='*40}")
    print(f"⏱️  Total Time Spent Report")
    print(f"{'='*40}")
    print(f"Total steps                 : {TOTAL_STEPS}")
    print(f"Total wall clock time (Wall Time) : {total_wall_time:.4f} s")
    print(f"Total calculation time (Compute)   : {total_compute_time:.4f} s")
    print(f"Average latency (per step)        : {avg_latency*1000:.2f} ms")
    print(f"Real-Time Factor (RTF)         : {total_wall_time / SIM_DURATION:.2f}x")
    print(f"Average frame rate (FPS)         : {TOTAL_STEPS / total_wall_time:.2f} Hz")
    print(f"{'='*40}\n")



    # ==========================================
    # 5. Save simulation data and call visualizations
    # ==========================================
    print("\n📊 Simulation data is being organized and saved...")
    
  
    history_states = np.array(log_pos)                  
    cbf_values = np.array(log_cbf)                      
    rl_ref_trajs_np = np.array(all_ref_trajs)         
    mppi_samples_np = np.array(all_mppi_samples)       
    time_axis = np.linspace(0, len(cbf_values)*DT_MPPI, len(cbf_values))
    """
    # === [NEW] Save core data packages as .npz files ===
    data_save_path = os.path.join(save_dir, "v3_data.npz")
    np.savez_compressed(
        data_save_path,
        actual_trajectory=history_states,
        hx_values=cbf_values,
        rl_reference=rl_ref_trajs_np,
        mppi_predictions=mppi_samples_np,
        time_axis=time_axis,          
        time_vars=np.array(time_vars)  
    )
    print(f"💾 All track and status data have been safely deposited: {data_save_path}")
    print("\n📊 The visualization module is being called...")"""
    
    history_states = np.array(log_pos)
    cbf_values = np.array(log_cbf)
    time_axis = np.linspace(0, len(cbf_values)*DT_MPPI, len(cbf_values))
    
    # 1. Static trajectory diagram
    animation.plot_static_trajectory(
        history_states=history_states,
        rl_ref_trajs=all_ref_trajs,
        obstacles=controller.obstacles,
        subgoals=controller.subgoals,
        map_limit=TGPOConfig.X_LIMIT,
        save_path=os.path.join(save_dir, "v_t.pdf")
    )
    
    # 2. CBF safety diagram
    animation.plot_cbf_safety(
        time_axis=time_axis,
        cbf_values=cbf_values,
        save_path=os.path.join(save_dir, "v_c.pdf")
    )
    """
    # 3. MPPI Animated video
    anim_path = os.path.join(save_dir, "reach_avoid_v2.mp4")

    animation.generate_mppi_animation(
        history_states=history_states,
        rl_ref_trajs=all_ref_trajs,
        mppi_sampled_trajs=all_mppi_samples,
        obstacles=controller.obstacles,
        subgoals=controller.subgoals,
        map_limit=TGPOConfig.X_LIMIT,
        steps_per_rl=STEPS_PER_RL,
        dt=DT_MPPI,
        save_path=anim_path
    )"""

if __name__ == "__main__":
    run_final_merged_ref_virtual()