import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import config
from model import dynamics

def plot_cbf_values(h_values, save_path):
    
    print(f"  ... is plotting CBF (h(x)) to: {save_path}")


    max_time = 35.0
    max_steps = int(max_time / config.dt)
    

    h_values = np.array(h_values)
    
    if len(h_values) > max_steps:
        print(f"    [Tip] Data length {len(h_values)} over {max_time}s，Truncated display。")
        h_values = h_values[:max_steps]
    
    # 1. Create a timeline
    num_steps = len(h_values)
    time_axis = np.arange(num_steps) * config.dt
    
    fig = plt.figure(figsize=(12, 6))
    
    # 2. Plotting the value of h(x).
    plt.plot(time_axis, h_values, label='CBF Value h(x, t)', linewidth=2)
    
    # 3. Drawing a safety boundary (h=0)
    plt.axhline(0, color='red', linestyle='--', label='Safety Boundary (h=0)', linewidth=2)
    
    # 4. Format the chart
    plt.title(f'Control Barrier Function Value (0-{max_time}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('h(x, t)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Automatically adjusts the Y-axis range
    if len(h_values) > 0:
        h_min = np.min(h_values)
        plt.ylim(bottom=min(h_min, -0.1) - 0.2, top=max(np.max(h_values), 1.0) + 0.2)
    
    # 5. Check for violations (h < 0)
    if np.any(h_values < 0):
        print("    [Warning]: CBF value is negative at some time steps (marked in red on the plot)!, CBF:", np.min(h_values))
        violation_indices = np.where(h_values < 0)[0]
        plt.scatter(time_axis[violation_indices], h_values[violation_indices], 
                    color='red', zorder=5, s=10, label='Violation')
    else:
        print("    [Success]: CBF value is always >= 0. System remains safe.")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def get_predicted_trajecotories(current_state, control_sequence):
    "Forecast line for drawing MPPI"
    predicted_state = [current_state]
    for ctrl in control_sequence:
        next_state = dynamics(predicted_state[-1], ctrl, config.dt)
        predicted_state.append(next_state.copy())
    return np.array(predicted_state)

def plot_simulation_result(states, save_path):
    """
    Plotting Static Trajectory Results (Limit 35s)
    """
    print(f"  ... is plotting static trajectory to: {save_path}")

   
    max_steps = int(35.0 / config.dt)
    if len(states) > max_steps:
        states = states[:max_steps]

    # Extract data
    states = np.array(states)
    x_vals = states[:, 0]
    y_vals = states[:, 1]

    # Calculate the adaptive canvas size
    if len(x_vals) > 0:
        x_len = np.max(x_vals) - np.min(x_vals)
        fig_width = np.clip(15 * (x_len / 50.0), 12, 25)
    else:
        fig_width = 15

    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    # --- 1. Draw the lane lines ---
    if len(x_vals) > 0:
        x_min, x_max = np.min(x_vals) - 5, np.max(x_vals) + 10
    else:
        x_min, x_max = 0, 100
    
    # Pavement background
    ax.fill_between([x_min, x_max], 0, 4, color='#333333', alpha=0.1)
    
    # Boundary line
    plt.axhline(0, color='black', linewidth=3, linestyle='-')
    plt.axhline(4, color='black', linewidth=3, linestyle='-')
    
    # Divider (distinct yellow dotted line)
    plt.axhline(2, color='orange', linewidth=2.5, linestyle='--', dashes=(5, 5))
    
    plt.text(x_min + 2, 1.0, "Lane 1 (Right)", fontsize=12, color='gray', ha='center', va='center')
    plt.text(x_min + 2, 3.0, "Lane 2 (Left)", fontsize=12, color='gray', ha='center', va='center')

    # --- 2. Drawing Obstacles (Initial Position) ---
    obs_x_init = np.array([4.0, 8.0, 12.0, 16.0])
    obs_y      = np.array([1.0, 3.0, 1.0, 3.0])
    # [NEW] Define an array of obstacles velocity
    obs_vx     = np.array([0.4, 0.6, 0.8, 1.0])
    # obs_vx     = np.array([0.5, 0.5, 0.8, 1.0])
    # obs_vx     = np.array([0.5, 0.5, 0.5, 0.5])
    obs_radius = 0.5
    
    for i in range(len(obs_x_init)):
        circle = plt.Circle((obs_x_init[i], obs_y[i]), obs_radius, 
                            color='black', alpha=0.6, label="Obstacle (Init)" if i == 0 else None)
        ax.add_artist(circle)
        # [Modified] Arrow length reflects speed (multiply by 2 to make the arrow more obvious)
        arrow_len = obs_vx[i] * 2.0
        plt.arrow(obs_x_init[i], obs_y[i], arrow_len, 0, head_width=0.2, color='black', alpha=0.6)
        # Label speed
        plt.text(obs_x_init[i], obs_y[i]-0.8, f"v={obs_vx[i]}", fontsize=8, ha='center', color='black')

    # --- 3. Drawing the Main Vehicle Track ---
    plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='Ego Trajectory')
    
    if len(states) > 0:
        start_circle = plt.Circle((x_vals[0], y_vals[0]), 0.5, color='green', alpha=0.8, label='Start')
        ax.add_artist(start_circle)
        end_circle = plt.Circle((x_vals[-1], y_vals[-1]), 0.5, color='red', alpha=0.8, label='End')
        ax.add_artist(end_circle)

    plt.axis('equal') 
    plt.ylim(-1, 5)
    plt.xlim(x_min, x_max)
    plt.title(f'Simulation Result: Highway Overtaking (Max 35s)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def animate_simulation(states, sampled_us=[], optimal_us=None, save_path="simulation.mp4"):
    """
    Generate global view animations (fixed perspective, strip canvas)
    """
    print(f"  ... is rendering global view animation to: {save_path}")

   
    max_steps = int(35.0 / config.dt)
    if len(states) > max_steps:
        print(f"    [Tip] Screenshot the first 35s (first {max_steps} frames)")
        states = states[:max_steps]
        if len(sampled_us) > max_steps:
            sampled_us = sampled_us[:max_steps]
    
    # 2. Calculate the global X-axis range
    states_np = np.array(states)
    all_x = states_np[:, 0]
    
    if len(all_x) > 0:
        x_min_global = min(np.min(all_x), 0) - 2
        x_max_global = max(np.max(all_x), 20) + 5
    else:
        x_min_global, x_max_global = -2, 25
        
    total_length = x_max_global - x_min_global
    
    # 3. Set up a strip canvas
    fig_width = np.clip(total_length / 3.0, 10, 20) 
    fig_height = 4.5 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Initial Obstacles Parameters (Ensure Consistent with main.py)
    obs_x_init = np.array([4.0, 8.0, 12.0, 16.0])
    
    obs_y      = np.array([1.0, 3.0, 1.0, 3.0])
    
    # [Modified] Define an array of obstacles velocity (corresponding to O1, O2, O3, O4)
    obs_vx     = np.array([0.4, 0.6, 0.8, 1.0])
    #obs_vx     = np.array([0.5, 0.5, 0.8, 1.0])
    #obs_vx     = np.array([0.5, 0.5, 0.5, 0.5])
    obs_radius = 0.5
    ego_radius = 0.5

    def update(frame, states):
        ax.cla()
        
        current_time = frame * config.dt
        current_state = states[frame]
        ego_x, ego_y, ego_theta = current_state[0], current_state[1], current_state[2]
        
        # --- 1. Set up a fixed global view ---
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal', adjustable='box') 
        
        # --- 2. Drawing an extra-long road ---
        ax.fill_between([x_min_global, x_max_global], 0, 4, color='#333333', alpha=0.2) 
        
        # 边界线
        ax.plot([x_min_global, x_max_global], [0, 0], 'k-', linewidth=3)
        ax.plot([x_min_global, x_max_global], [4, 4], 'k-', linewidth=3)
        
        # Divider (yellow dotted line)
        ax.plot([x_min_global, x_max_global], [2, 2], color='#FFA500', linestyle='--', linewidth=2.5, dashes=(5, 5)) 
        
        # --- 3. Dynamic Obstacles (Modify Logic Support Arrays) ---
        
        current_obs_x = obs_x_init + obs_vx * current_time
        
        for i in range(len(current_obs_x)):
            obs_circle = plt.Circle((current_obs_x[i], obs_y[i]), obs_radius, 
                                    color='#8B0000', alpha=0.8) 
            ax.add_artist(obs_circle)
            # Displays ID and current speed
            ax.text(current_obs_x[i], obs_y[i], f"O{i+1}\nv={obs_vx[i]}", color='white', 
                    ha='center', va='center', fontsize=7, fontweight='bold')

        # --- 4. Draw MPPI Forecast Lines (Enabled) ---
        # If a sampled_us is provided, it is drawn
        if len(sampled_us) > 0 and frame < len(sampled_us):
            # Take the first 20 to avoid too much impact on performance
            num_samples_to_plot = min(20, len(sampled_us[frame]))
            for i in range(num_samples_to_plot):
                # You need to call get_predicted_trajecotories to calculate the trajectory point
                traj = get_predicted_trajecotories(current_state, sampled_us[frame][i])
                # Use bright green (lime) for low transparency
                ax.plot(traj[:, 0], traj[:, 1], color='lime', alpha=0.15, linewidth=1.0)

        # --- 5. Drawing the main car ---
        ego_circle = plt.Circle((ego_x, ego_y), ego_radius, color='blue', alpha=0.9, label='Ego', zorder=10)
        ax.add_artist(ego_circle)
        
        arrow_len = 0.8
        ax.arrow(ego_x, ego_y, arrow_len * np.cos(ego_theta), arrow_len * np.sin(ego_theta),
                 head_width=0.3, color='cyan', zorder=11)

        # --- 6. Drawing Trail Afterimages - Enhanced ---
        # Draw past trajectories
        traj_x = [s[0] for s in states[:frame+1]]
        traj_y = [s[1] for s in states[:frame+1]]
        # Modification: The line width is set to 4, the color is Cyan, and the alpha is 0.9, which is very eye-catching
        ax.plot(traj_x, traj_y, color='cyan', linewidth=4, alpha=0.9)

        # Title information
        ax.set_title(f"Simulation Time: {current_time:.2f}s | Speed: {current_state[3]:.2f} m/s", fontsize=10)
        ax.set_xlabel("Position X (m)")
        ax.grid(True, linestyle=':', alpha=0.3)

    # Rendering settings
    real_fps = int(1.0 / config.dt)
    frame_interval_ms = int(config.dt * 1000) 
    
    print(f"    [Settings] Canvas range X: {x_min_global:.1f} -> {x_max_global:.1f}")
    print(f"    [Settings] DT={config.dt}s -> FPS={real_fps}, Interval={frame_interval_ms}ms")

    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), 
                         interval=frame_interval_ms, blit=False) 

    try:
        anim.save(save_path, writer='ffmpeg', fps=real_fps, extra_args=['-vcodec', 'libx264'])
        print(f"  The animation is saved to: {save_path}")
    except Exception as e:
        print(f"  [Error] MP4 saving failed: {e}")
        print("  Trying to save as GIF...")
        try:
            anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=real_fps)
            print(f"  GIF has been saved: {save_path.replace('.mp4', '.gif')}")
        except Exception as e2:
             print(f"  [Error] Failed to save animation: {e2}")
    
    plt.close(fig)