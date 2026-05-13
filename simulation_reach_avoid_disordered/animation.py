import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

def plot_static_trajectory(
    history_states, 
    rl_ref_trajs, 
    obstacles, 
    subgoals, 
    map_limit, 
    save_path
):
    """
    Draw and save static track summary maps.
    Features: The final trajectory is bolded, and the start and end points draw solid circles with a radius of 0.2m.
    """
    print(f"📊 Static trajectory mapping is being drawn: {save_path} ...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
   
    ROBOT_RADIUS = 0.2
    
   
    ax.add_patch(patches.Circle((0, 0), map_limit, color='k', fill=False, ls='--', label='Boundary'))
    
   
    for i, obs in enumerate(obstacles):
        label = 'Obstacle' if i == 0 else None
        ax.add_patch(patches.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.3, label=label))
        
   
    for i, g in enumerate(subgoals):
        label = 'Subgoal' if i == 0 else None
        ax.add_patch(patches.Circle((g[0], g[1]), g[2], color='purple', alpha=0.3, label=label))
        ax.text(g[0], g[1], f"G{i}", ha='center', color='black', fontweight='bold')

    #  Draw RL guide trajectory (green thin line cluster)
    if rl_ref_trajs and len(rl_ref_trajs) > 0:
        lc = LineCollection(rl_ref_trajs, colors='limegreen', linewidths=1.0, alpha=0.5)
        ax.add_collection(lc)
        ax.plot([], [], color='limegreen', linewidth=1.5, label='RL Guidance Plan')

    #  Draw the actual execution trajectory (blue, bold!)
    ax.plot(history_states[:, 0], history_states[:, 1], 
            color='blue', linewidth=4.0, alpha=0.9, label='Actual Trajectory') 

    #  Draw Start and End Points (modified to solid circles)
  
    start_pos = history_states[0]
    start_circle = patches.Circle((start_pos[0], start_pos[1]), ROBOT_RADIUS, 
                                  facecolor='lime', edgecolor='k', alpha=0.9, zorder=10, label='Start (r=0.2)')
    ax.add_patch(start_circle)

    end_pos = history_states[-1]
    end_circle = patches.Circle((end_pos[0], end_pos[1]), ROBOT_RADIUS, 
                                facecolor='blue', edgecolor='k', alpha=0.9, zorder=10, label='End (r=0.2)')
    ax.add_patch(end_circle)
    

    ax.set_title("Simulation Summary: Trajectory & Environment", fontsize=14)
    ax.axis('equal')
    ax.set_xlim(-map_limit - 1, map_limit + 1)
    ax.set_ylim(-map_limit - 1, map_limit + 1)
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已保存: {save_path}")


def plot_cbf_safety(time_axis, cbf_values, save_path):
    """
    Plot and save the CBF safety function curve.
    (There is no need to modify the radius display here, because it is a numerical curve)adius display here, as it's a numerical curve)
    """
    print(f"📊 CBF safety mapping is being drawn: {save_path} ...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    cbf_values = np.array(cbf_values)
    
    ax.plot(time_axis, cbf_values, 'k-', linewidth=2.0, label='h(x) Value')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
    
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values < 0), 
                    color='red', alpha=0.3, label='Unsafe Region (h<0)')
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values >= 0), 
                    color='green', alpha=0.1, label='Safe Region (h>=0)')
    
    ax.set_title("Safety Verification (Control Barrier Function)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(x)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ CBF safety mapping saved: {save_path}")


def generate_mppi_animation(
    history_states,       
    rl_ref_trajs,         
    mppi_sampled_trajs,   
    obstacles,            
    subgoals,             
    map_limit,            
    steps_per_rl,         
    dt,                   
    save_path
):
    """
    Generate a dynamic video MP4 with the MPPI sampling process.
    Modified: The robot appears as a solid circle with a radius of 0.2m.
    """
    print(f"🎬 Initializing animation generation: {save_path}")
    
    real_time_fps = int(1.0 / dt)
    print(f"   -> Physical step size dt={dt:.3f}s")
    print(f"   -> Set video frame rate FPS={real_time_fps}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define robot radius
    ROBOT_RADIUS = 0.2

    # --- Initializing the environment ---
    def init_environment():
        ax.set_xlim(-map_limit - 1, map_limit + 1)
        ax.set_ylim(-map_limit - 1, map_limit + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Guided MPPI Simulation (Virtual Clock)")
        
        # borders
        ax.add_patch(patches.Circle((0, 0), map_limit, color='k', fill=False, ls='--'))
        
        # Obstacles
        for obs in obstacles:
            ax.add_patch(patches.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.4))
            
        # Goal
        for i, g in enumerate(subgoals):
            ax.add_patch(patches.Circle((g[0], g[1]), g[2], color='purple', alpha=0.3))
            ax.text(g[0], g[1], f"G{i}", ha='center', color='white', fontweight='bold')
            
        return []

    # --- Predefined Drawing Elements ---
    
    # 1. MPPI sample cluster
    max_samples = 50
    mppi_lines = [
        ax.plot([], [], color='darkorange', linewidth=0.8, alpha=0.6)[0] 
        for _ in range(max_samples)
    ]
    
    # 2. RL reference track
    ref_line, = ax.plot([], [], 'g--', linewidth=2.5, label='RL Guidance', alpha=0.9)
    
    # 3. Actual Track Line
    traj_line, = ax.plot([], [], 'b-', linewidth=4.0, label='Actual Traj', alpha=0.8)
    
    # 4. [Modified] Bot Entity (Circle Patch)
    # Instead of using ax.plot's marker, create a patch
    robot_body = patches.Circle((0, 0), ROBOT_RADIUS, fc='dodgerblue', ec='navy', 
                                alpha=1.0, zorder=20, label='Robot (r=0.2)')
    ax.add_patch(robot_body)
    
    # 5. [NEW] Simple direction indicator line (if the status contains theta, here it is simply represented by a straight line)
    # Assuming history_states is [x, y, theta, ...], this part may not be needed without theta
    # For insurance, we initialize a short line
    robot_dir_line, = ax.plot([], [], color='navy', linewidth=2, zorder=21)

    # --- Build the full legend ---
    legend_elements = [
        plt.Line2D([0], [0], color='r', marker='o', linestyle='None', alpha=0.4, markersize=10, label='Obstacle'),
        plt.Line2D([0], [0], color='purple', marker='o', linestyle='None', alpha=0.3, markersize=10, label='Goal'),
        plt.Line2D([0], [0], color='darkorange', lw=1, label='MPPI Samples'),
        plt.Line2D([0], [0], color='green', lw=2.5, ls='--', label='RL Guidance'),
        # The legend shows what the robot looks like
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', 
                   markeredgecolor='navy', markersize=10, label=f'Robot (r={ROBOT_RADIUS}m)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # --- Updating Functions ---
    total_frames = len(history_states)
    
    def update(frame):
        if frame % 50 == 0:
            print(f"\r   -> Rendering frame {frame}/{total_frames} ({frame/total_frames*100:.1f}%)", end="")
            
        artists = []
        
        # A. Update the actual track line
        curr_traj = history_states[:frame+1]
        traj_line.set_data(curr_traj[:, 0], curr_traj[:, 1])
        artists.append(traj_line)
        
        # B. [Modified] Updated the bot entity location
        curr_state = history_states[frame]
        x, y = curr_state[0], curr_state[1]
        
        # Update the center of the circle
        robot_body.center = (x, y)
        artists.append(robot_body)
        
        # Try updating the orientation line (if you have Theta)
        if len(curr_state) > 2:
            theta = curr_state[2]
            # Draw a line from the center of the circle with a length equal to the radius
            end_x = x + ROBOT_RADIUS * np.cos(theta)
            end_y = y + ROBOT_RADIUS * np.sin(theta)
            robot_dir_line.set_data([x, end_x], [y, end_y])
            artists.append(robot_dir_line)
        
        # C. 更新 RL 参考文献
        rl_idx = frame // steps_per_rl
        if rl_ref_trajs and rl_idx < len(rl_ref_trajs):
            ref = rl_ref_trajs[rl_idx]
            ref_line.set_data(ref[:, 0], ref[:, 1])
            artists.append(ref_line)
            
        # D. Update MPPI sampling
        if mppi_sampled_trajs and frame < len(mppi_sampled_trajs):
            samples = mppi_sampled_trajs[frame]
            if samples is not None:
                for i, line in enumerate(mppi_lines):
                    if i < len(samples) and i < max_samples:
                        line.set_data(samples[i, :, 0], samples[i, :, 1])
                        artists.append(line)
                    else:
                        line.set_data([], [])
            else:
                for line in mppi_lines: line.set_data([], [])
        
        return artists

    # --- Generate ---
    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init_environment, 
                        blit=True, interval=20)
    
    print("\n   -> Encoding is using FFmpeg ...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=real_time_fps, dpi=200, bitrate=3000)
        print(f"\n✅ The animation is generated: {save_path}")
    except Exception as e:
        print(f"\n❌ Error: Failed to generate MP4。")
        print(f"   Cause: {e}")
        
    plt.close(fig)