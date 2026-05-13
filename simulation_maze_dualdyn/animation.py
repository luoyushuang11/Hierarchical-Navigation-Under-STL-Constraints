import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation


INTERNAL_DYN_OBS_PARAMS = np.array([
    [5.8, 0.5, 9.5, 0.4, 0.5, 0.0], # Obs 1
    [7.3, 0.5, 9.5, 0.4, 0.5, 9.0]  # Obs 2
    #[6.0, 0.5, 9.5, 0.4, 0.5, 0.0]
])

def get_dyn_pos_internal(t, param):
    
    x_fixed, y_min, y_max, speed, _, offset = param
    length = y_max - y_min
    cycle = 2.0 * length
    dist = t * speed + offset
    mod = dist % cycle
    if mod <= length:
        y = y_min + mod
    else:
        y = y_max - (mod - length)
    return x_fixed, y

def plot_static_trajectory(
    history_states, 
    rl_ref_trajs, 
    obstacles, 
    subgoals, 
    map_limit, 
    save_path
):
    
    print(f"📊 Static trajectory mapping is being drawn: {save_path} ...")
    

    fig, ax = plt.subplots(figsize=(14, 10))
    
  
    ax.add_patch(patches.Rectangle((0, 0), 12.0, 10.0, color='k', fill=False, ls='--', lw=2, label='Boundary'))
    

    if obstacles is not None:
        obstacles_np = np.array(obstacles)
        for i, obs in enumerate(obstacles_np):
            label = 'Obstacle' if i == 0 else None
            if len(obs) == 3: # Circle
                ax.add_patch(patches.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.3, label=label))
            elif len(obs) == 4: # Rectangle
                cx, cy, w, h = obs
                lx, ly = cx - w / 2.0, cy - h / 2.0
                ax.add_patch(patches.Rectangle((lx, ly), w, h, color='r', alpha=0.3, label=label))
        

    if subgoals is not None:
        for i, g in enumerate(subgoals):
            label = 'Subgoal' if i == 0 else None
            ax.add_patch(patches.Circle((g[0], g[1]), g[2], color='purple', alpha=0.3, label=label))
            ax.text(g[0], g[1], f"G{i}", ha='center', va='center', color='black', fontweight='bold')


    if rl_ref_trajs and len(rl_ref_trajs) > 0:
        try:
            lc = LineCollection(rl_ref_trajs, colors='limegreen', linewidths=1.0, alpha=0.5)
            ax.add_collection(lc)
        except:
            pass
        ax.plot([], [], color='limegreen', linewidth=1.5, label='RL Guidance Plan')


    if len(history_states) > 0:
        history_states = np.array(history_states)
        ax.plot(history_states[:, 0], history_states[:, 1], 
                color='blue', linewidth=4.0, alpha=0.9, label='Actual Trajectory') 

        ax.scatter(history_states[0,0], history_states[0,1], c='lime', s=150, zorder=10, edgecolors='k', label='Start')
        ax.scatter(history_states[-1,0], history_states[-1,1], c='red', marker='X', s=150, zorder=10, edgecolors='k', label='End')
    

    ax.set_title("Simulation Summary: Trajectory & Environment", fontsize=16)
    ax.set_aspect('equal')
    
   
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1.0, 11.0)
    ax.grid(True, alpha=0.3)
    

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=12)
    
    plt.tight_layout() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ saved: {save_path}")


def plot_cbf_safety(time_axis, cbf_values, save_path):
    
    print(f"📊 CBF safety mapping is being drawn: {save_path} ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    cbf_values = np.array(cbf_values)
    ax.plot(time_axis, cbf_values, 'k-', linewidth=2.0, label='h(x) Value')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values < 0), color='red', alpha=0.3, label='Unsafe Region (h<0)')
    ax.fill_between(time_axis, cbf_values, 0, where=(cbf_values >= 0), color='green', alpha=0.1, label='Safe Region (h>=0)')
    ax.set_title("Safety Verification (Control Barrier Function)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(x)")
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    
    real_time_fps = int(1.0 / dt) if dt > 0 else 30
    print(f"   -> Physical step length dt={dt:.3f}s")
    print(f"   -> Set the video frame rate FPS={real_time_fps} ")
    

    fig, ax = plt.subplots(figsize=(14, 10))
    
   
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1.0, 11.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Guided MPPI Simulation (Virtual Clock)", fontsize=16)
    

    ax.add_patch(patches.Rectangle((0, 0), 12.0, 10.0, color='k', fill=False, ls='--', lw=2, label='Boundary'))
    

    if obstacles is not None:
        obstacles_np = np.array(obstacles)
        for obs in obstacles_np:
            if len(obs) == 3:
                ax.add_patch(patches.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.4))
            elif len(obs) == 4:
                cx, cy, w, h = obs
                lx = cx - w / 2.0
                ly = cy - h / 2.0
                ax.add_patch(patches.Rectangle((lx, ly), w, h, color='r', alpha=0.4))
            

    if subgoals is not None:
        subgoals_np = np.array(subgoals)
        for i, g in enumerate(subgoals_np):
            ax.add_patch(patches.Circle((g[0], g[1]), g[2], color='purple', alpha=0.3))
            ax.text(g[0], g[1], f"G{i}", ha='center', va='center', color='black', fontweight='bold')
            
   
    dyn_patches = []
    for param in INTERNAL_DYN_OBS_PARAMS:
        r = param[4]
  
        patch = patches.Circle((0, 0), r, color='darkred', alpha=0.7, zorder=15)
        ax.add_patch(patch)
        dyn_patches.append(patch)


    max_samples = 50
    mppi_lines = [
        ax.plot([], [], color='darkorange', linewidth=0.8, alpha=0.6)[0] 
        for _ in range(max_samples)
    ]
    

    ref_line, = ax.plot([], [], 'g--', linewidth=2.5, label='RL Guidance', alpha=0.9)
    
  
    robot_patch = patches.Circle((0, 0), 0.2, color='gold', zorder=20, label='Robot')
    ax.add_patch(robot_patch)
    

    traj_line, = ax.plot([], [], 'b-', linewidth=4.0, label='Actual Traj', alpha=0.8)
    

    legend_elements = [
        plt.Line2D([0], [0], color='k', ls='--', lw=2, label='Boundary'),
        plt.Line2D([0], [0], color='r', marker='s', linestyle='None', alpha=0.4, markersize=10, label='Static Obs'),
        plt.Line2D([0], [0], color='darkred', marker='o', linestyle='None', alpha=0.7, markersize=10, label='Dyn Obs'),
        plt.Line2D([0], [0], color='purple', marker='o', linestyle='None', alpha=0.3, markersize=10, label='Goal'),
        plt.Line2D([0], [0], color='darkorange', lw=1, label='MPPI Samples'),
        plt.Line2D([0], [0], color='green', lw=2.5, ls='--', label='RL Guidance'),
        plt.Line2D([0], [0], color='blue', lw=4.0, label='Actual Traj'),
        plt.Line2D([0], [0], color='gold', marker='o', linestyle='None', markersize=10, label='Robot'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=12)

    total_frames = len(history_states)
    
    def update(frame):
        if frame % 50 == 0:
            print(f"\r   -> Rendering frame {frame}/{total_frames} ({frame/total_frames*100:.1f}%)", end="")
            
        artists = []
        t_now = frame * dt
        
        
        curr_pos = history_states[frame]
        robot_patch.center = (curr_pos[0], curr_pos[1])
        artists.append(robot_patch)
        
 
        curr_traj = np.array(history_states[:frame+1])
        traj_line.set_data(curr_traj[:, 0], curr_traj[:, 1])
        artists.append(traj_line)
        
       
        for i, param in enumerate(INTERNAL_DYN_OBS_PARAMS):
            dx, dy = get_dyn_pos_internal(t_now, param)
            dyn_patches[i].center = (dx, dy)
            artists.append(dyn_patches[i])
        
  
        rl_idx = frame // steps_per_rl
        if rl_ref_trajs and rl_idx < len(rl_ref_trajs):
            ref = rl_ref_trajs[rl_idx]
            ref_line.set_data(ref[:, 0], ref[:, 1])
            artists.append(ref_line)
            
   
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

    def init(): return []

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, 
                        blit=True, interval=20)
    
    print("\n   -> Encoding in FFmpeg (guaranteed 1:1 live playback)...")
    
    
    plt.tight_layout()
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=real_time_fps, dpi=200, bitrate=3000)
        print(f"\n✅ Animation generated successfully: {save_path}")
    except Exception as e:
        print(f"\n❌ Error: Failed to generate MP4.")
        print(f"   Reason: {e}")
        print("   Hint: Please check if FFmpeg is installed. If it fails, try saving as a GIF.")
    plt.close(fig)
    

'''

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
    
    real_time_fps = int(1.0 / dt) if dt > 0 else 30
    print(f"   -> Physical step size dt={dt:.3f}s")
    print(f"   -> Video frame rate FPS={real_time_fps} (to match real time)")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1.0, 11.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Guided MPPI Simulation (Virtual Clock)", fontsize=16)
    
    ax.add_patch(patches.Rectangle((0, 0), 12.0, 10.0, color='k', fill=False, ls='--', lw=2, label='Boundary'))

    if obstacles is not None:
        obstacles_np = np.array(obstacles)
        for obs in obstacles_np:
            if len(obs) == 3:
                ax.add_patch(patches.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.4))
            elif len(obs) == 4:
                cx, cy, w, h = obs
                lx = cx - w / 2.0
                ly = cy - h / 2.0
                ax.add_patch(patches.Rectangle((lx, ly), w, h, color='r', alpha=0.4))

    if subgoals is not None:
        subgoals_np = np.array(subgoals)
        for i, g in enumerate(subgoals_np):
            ax.add_patch(patches.Circle((g[0], g[1]), g[2], color='purple', alpha=0.3))
            ax.text(g[0], g[1], f"G{i}", ha='center', va='center', color='black', fontweight='bold')
            
   
    # dyn_patches = []
    # for param in INTERNAL_DYN_OBS_PARAMS:
    #     r = param[4]
    #     patch = patches.Circle((0, 0), r, color='darkred', alpha=0.7, zorder=15)
    #     ax.add_patch(patch)
    #     dyn_patches.append(patch)


    max_samples = 50
    mppi_lines = [
        ax.plot([], [], color='darkorange', linewidth=0.8, alpha=0.6)[0] 
        for _ in range(max_samples)
    ]
    
  
    ref_line, = ax.plot([], [], 'g--', linewidth=2.5, label='RL Guidance', alpha=0.9)
    
  
    robot_patch = patches.Circle((0, 0), 0.2, color='gold', zorder=20, label='Robot')
    ax.add_patch(robot_patch)
    

    traj_line, = ax.plot([], [], 'b-', linewidth=4.0, label='Actual Traj', alpha=0.8)
    

    legend_elements = [
        plt.Line2D([0], [0], color='k', ls='--', lw=2, label='Boundary'),
        plt.Line2D([0], [0], color='r', marker='s', linestyle='None', alpha=0.4, markersize=10, label='Static Obs'),
        # plt.Line2D([0], [0], color='darkred', marker='o', linestyle='None', alpha=0.7, markersize=10, label='Dyn Obs'),
        plt.Line2D([0], [0], color='purple', marker='o', linestyle='None', alpha=0.3, markersize=10, label='Goal'),
        plt.Line2D([0], [0], color='darkorange', lw=1, label='MPPI Samples'),
        plt.Line2D([0], [0], color='green', lw=2.5, ls='--', label='RL Guidance'),
        plt.Line2D([0], [0], color='blue', lw=4.0, label='Actual Traj'),
        plt.Line2D([0], [0], color='gold', marker='o', linestyle='None', markersize=10, label='Robot'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=12)

    total_frames = len(history_states)
    
    def update(frame):
        if frame % 50 == 0:
            print(f"\r   -> Rendering frame {frame}/{total_frames} ({frame/total_frames*100:.1f}%)", end="")
            
        artists = []
        t_now = frame * dt 
        
    
        curr_pos = history_states[frame]
        robot_patch.center = (curr_pos[0], curr_pos[1])
        artists.append(robot_patch)
        

        curr_traj = np.array(history_states[:frame+1])
        traj_line.set_data(curr_traj[:, 0], curr_traj[:, 1])
        artists.append(traj_line)
      
        # for i, param in enumerate(INTERNAL_DYN_OBS_PARAMS):
        #     dx, dy = get_dyn_pos_internal(t_now, param)
        #     dyn_patches[i].center = (dx, dy)
        #     artists.append(dyn_patches[i])
        
  
        rl_idx = frame // steps_per_rl
        if rl_ref_trajs and rl_idx < len(rl_ref_trajs):
            ref = rl_ref_trajs[rl_idx]
            ref_line.set_data(ref[:, 0], ref[:, 1])
            artists.append(ref_line)

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

    def init(): return []

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, 
                        blit=True, interval=20)
    
    print("\n   -> Encoding in FFmpeg (guaranteed 1:1 live playback)...")
    
    plt.tight_layout()
    
    try:
        ani.save(save_path, writer='ffmpeg', fps=real_time_fps, dpi=200, bitrate=3000)
        print(f"\n✅ Animation generated successfully: {save_path}")
    except Exception as e:
        print(f"\n❌ Error: Failed to generate MP4.")
        print(f"   Reason: {e}")
        
    plt.close(fig)
'''