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
    
    print(f"📊 Static trajectory mapping is being drawn: {save_path} ...")
    
    
    final_x = history_states[-1, 0]
    view_min_x = -2.0
    view_max_x = max(map_limit, final_x) + 5.0
    view_range_x = view_max_x - view_min_x
    
    view_min_y = -1.0
    view_max_y = 5.0
    view_range_y = view_max_y - view_min_y
    
    
    fig_height = 4.0

    fig_width = fig_height * (view_range_x / view_range_y)
    

    fig_width = min(fig_width, 24.0)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
   
    ax.set_aspect('equal')
    
  
    ax.set_xlim(view_min_x, view_max_x)
    ax.set_ylim(view_min_y, view_max_y)
    

    ax.axhline(0, color='black', linewidth=3)
    ax.axhline(4, color='black', linewidth=3)
    ax.axhline(2, color='white', linestyle='--', linewidth=2) 
    ax.set_facecolor('#d9d9d9') 
    

    obs_has_vel = (obstacles.shape[1] >= 4)
    #default_vels = [0.4, 0.6, 0.8, 1.0] 
    default_vels = [0.5, 0.5, 0.8, 1.0] 
    for i, obs in enumerate(obstacles):

        obs_circle = patches.Circle((obs[0], obs[1]), obs[2] if obs.shape[0]>2 else 0.5, 
                                    color='red', alpha=0.5, label='Obstacle (Init)' if i == 0 else None)
        ax.add_patch(obs_circle)
        

        vx = obs[2] if obs_has_vel else default_vels[i % 4]
        ax.arrow(obs[0], obs[1], vx*5.0, 0, head_width=0.2, color='darkred', alpha=0.8)
        ax.text(obs[0], obs[1]-0.8, f"O{i}", ha='center', fontsize=8, color='darkred')

    for i, g in enumerate(subgoals):
        offset_x, goal_y, goal_r, _, ref_idx = g
        ref_idx = int(ref_idx)
        init_goal_x = obstacles[ref_idx, 0] + offset_x
        
        goal_circle = patches.Circle((init_goal_x, goal_y), goal_r, 
                                     color='purple', alpha=0.2, linestyle='--', label='Subgoal' if i == 0 else None)
        ax.add_patch(goal_circle)
        ax.text(init_goal_x, goal_y, f"G{i}", ha='center', va='center', color='purple', fontweight='bold', fontsize=8)


    if rl_ref_trajs and len(rl_ref_trajs) > 0:
        lc = LineCollection(rl_ref_trajs, colors='limegreen', linewidths=1.0, alpha=0.5)
        ax.add_collection(lc)
        ax.plot([], [], color='limegreen', linewidth=1.5, label='RL Guidance')

    ax.plot(history_states[:, 0], history_states[:, 1], 
            color='blue', linewidth=3.0, alpha=0.9, label='Actual Trajectory') 


    ax.scatter(history_states[0,0], history_states[0,1], c='lime', s=100, zorder=10, edgecolors='k', label='Start')
    ax.scatter(history_states[-1,0], history_states[-1,1], c='red', marker='X', s=100, zorder=10, edgecolors='k', label='End')
    
    ax.set_title(f"Simulation Summary: Dynamic Overtaking", fontsize=14)
    ax.grid(True, alpha=0.3)
    

    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ saved: {save_path}")


def plot_cbf_safety(time_axis, cbf_values, save_path):
  
    print(f"📊 CBF safety mapping is being drawn: {save_path} ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    cbf_values = np.array(cbf_values)
    ax.plot(time_axis, cbf_values, 'k-', linewidth=2.0, label='h(x) Value')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values < 0), color='red', alpha=0.3, label='Unsafe')
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values >= 0), color='green', alpha=0.1, label='Safe')
    ax.set_title("Safety Verification (CBF)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(x)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ saved: {save_path}")


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
    
    print(f"🎬 Initializing animation generation: {save_path}")
    
    real_time_fps = int(1.0 / dt)
    
 
    final_x = history_states[-1, 0]
    view_min_x = -2.0
    view_max_x = max(map_limit, final_x) + 10.0
    view_range_x = view_max_x - view_min_x
    
    view_min_y = -1.0
    view_max_y = 5.0
    view_range_y = view_max_y - view_min_y
    
  
    fig_height = 4.0
  
    fig_width = fig_height * (view_range_x / view_range_y)
    
   
    fig_width = max(10.0, min(fig_width, 24.0))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    

    ax.set_aspect('equal')
    
  
    if obstacles.shape[1] >= 4:
        obs_vxs = obstacles[:, 2]
    else:
        obs_vxs = np.array([0.4, 0.6, 0.8, 1.0])
        
    obs_start_xs = obstacles[:, 0]
    obs_ys = obstacles[:, 1]
    
  
    def init_environment():
        ax.set_ylim(view_min_y, view_max_y)
        ax.set_xlim(view_min_x, view_max_x)
        
       
        ax.axhline(0, color='black', linewidth=3)
        ax.axhline(4, color='black', linewidth=3)
        ax.axhline(2, color='white', linestyle='--', linewidth=2)
        ax.set_facecolor('#d9d9d9')
        
        ax.set_title("Guided MPPI Simulation (Fixed Global View)")
        return []

   
    
  
    obs_patches = []
    obs_texts = []
    for i in range(len(obstacles)):
        c = patches.Circle((0, 0), 0.5, color='darkred', alpha=0.6, zorder=5)
        t = ax.text(0, 0, f"O{i}", color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
        ax.add_patch(c)
        obs_patches.append(c)
        obs_texts.append(t)
        
  
    goal_patch = patches.Circle((0, 0), 0.5, color='purple', alpha=0.0, linestyle='--', linewidth=2, zorder=4)
    ax.add_patch(goal_patch)
    goal_text = ax.text(0, 0, "", color='purple', ha='center', fontweight='bold', zorder=4)

   
    max_samples = 30
    mppi_lines = [
        ax.plot([], [], color='darkorange', linewidth=1.0, alpha=0.4)[0] 
        for _ in range(max_samples)
    ]
    
    
    ref_line, = ax.plot([], [], 'g--', linewidth=2.5, label='RL Guidance', alpha=0.9, zorder=8)
    
    
    traj_line, = ax.plot([], [], 'b-', linewidth=3.0, label='Actual Traj', alpha=0.8, zorder=9)
    
    robot_marker = patches.Circle((0, 0), 0.5, color='blue', zorder=10)
    ax.add_patch(robot_marker)
    
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12, fontweight='bold')


    total_frames = len(history_states)
    
    def update(frame):
        t = frame * dt
        ego_state = history_states[frame]
        ego_x = ego_state[0]
        ego_y = ego_state[1]
        
        artists = []
        
      
        curr_obs_xs = obs_start_xs + obs_vxs * t
        for i, (c, txt) in enumerate(zip(obs_patches, obs_texts)):
            c.center = (curr_obs_xs[i], obs_ys[i])
            txt.set_position((curr_obs_xs[i], obs_ys[i]))
            artists.append(c)
            artists.append(txt)
            
      
        active_goal_idx = -1
        min_dist = 1e9
        closest_goal_pos = None
        
        for i, g in enumerate(subgoals):
            off, gy, gr, _, ref_idx = g
            ref_idx = int(ref_idx)
            g_x = curr_obs_xs[ref_idx] + off
            
          
            if g_x > ego_x - 5.0:
                d = abs(g_x - ego_x)
                if d < min_dist:
                    min_dist = d
                    closest_goal_pos = (g_x, gy, gr, i)
        
        if closest_goal_pos: 
            gx, gy, gr, g_idx = closest_goal_pos
            goal_patch.center = (gx, gy)
            goal_patch.set_radius(gr)
            goal_patch.set_alpha(0.3)
            goal_text.set_position((gx, gy + 0.8))
            goal_text.set_text(f"G{g_idx}")
            artists.append(goal_patch)
            artists.append(goal_text)
        else:
            goal_patch.set_alpha(0.0)
            goal_text.set_text("")
            
      
        traj_line.set_data(history_states[:frame+1, 0], history_states[:frame+1, 1])
        artists.append(traj_line)
        
        robot_marker.center = (ego_x, ego_y)
        artists.append(robot_marker)
        
     
        rl_idx = frame // steps_per_rl
        if rl_ref_trajs and rl_idx < len(rl_ref_trajs):
            ref = rl_ref_trajs[rl_idx]
            ref_line.set_data(ref[:, 0], ref[:, 1])
            artists.append(ref_line)
        else:
            ref_line.set_data([], [])


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
        
        time_text.set_text(f"Time: {t:.1f}s")
        artists.append(time_text)
        
        return artists

  
    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init_environment, 
                        blit=True, interval=20)
    
    print("\n   -> Using FFmpeg encoding (fixed global view, equal scale)...")
    try:
        ani.save(save_path, writer='ffmpeg', fps=real_time_fps, dpi=150, bitrate=2500)
        print(f"\n✅ The animation is generated: {save_path}")
    except Exception as e:
        print(f"\n❌ Error: Failed to generate MP4. Cause: {e}")
        
    plt.close(fig)