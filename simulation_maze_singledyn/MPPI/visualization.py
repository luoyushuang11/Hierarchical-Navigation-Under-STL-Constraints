import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np


try:
    import config
    DT = config.dt
except ImportError:
    DT = 0.02 

try:
    from model import dynamics
except ImportError:
   
    def dynamics(state, action, dt): return state 

# ==========================================
# 0. Environmental data definition
# ==========================================
# Workspace: [0, 12] x [0, 10]
WS_RANGE = [0, 12, 0, 10]

#  [cx, cy, half_w, half_h]
STATIC_OBSTACLES = [
    [2.0, 1.0, 1.0, 1.0], [1.5, 4.25, 1.5, 0.25], 
    [2.0, 7.0, 1.0, 1.0], [4.75, 6.0, 0.25, 4.0],
    [8.25, 4.0, 0.25, 4.0], [11.25, 3.5, 0.75, 0.5], 
    [9.25, 5.5, 0.75, 0.5], [11.5, 9.0, 0.5, 1.0]
]

# 2 dynamic obstacle parameters
DYN_OBS_PARAMS = {
    'x_fixed': [5.8, 7.3],#[6.0]
    'phases': [0.0, 9.0],#[0.0]
    'radius': 0.5,
    'speed': 0.4,
    'y_min': 0.5,
    'span': 9.0, 'cycle': 18.0
}


TARGETS = [
    {'pos': [1.0, 3.0], 'radius': 0.6, 'active_until': 30.0},
    {'pos': [2.25, 9.0], 'radius': 0.6, 'active_until': float('inf')}
]


# ==========================================
# 1. Draw the CBF curve
# ==========================================
def plot_cbf_values(h_values, save_path):
    
    print(f"  ... CBF (h(x)) : {save_path}")

    max_time = 40.0 
    max_steps = int(max_time / DT)
    
    h_values = np.array(h_values)
    
    if len(h_values) > max_steps:
        print(f"    [Tip] Data length {len(h_values)} over {max_time}s，Truncated display。")
        h_values = h_values[:max_steps]
    
    num_steps = len(h_values)
    time_axis = np.arange(num_steps) * DT
    
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(time_axis, h_values, label='CBF Value h(x, t)', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', label='Safety Boundary (h=0)', linewidth=2)
    
    plt.title(f'Control Barrier Function Value (0-{max_time}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('h(x, t)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    if len(h_values) > 0:
        h_min = np.min(h_values)
        plt.ylim(bottom=min(h_min, -0.1) - 0.2, top=max(np.max(h_values), 1.0) + 0.2)
    
    if np.any(h_values < 0):
        print("    [Warning]: The CBF value is less than 0 at some moments (marked in red in the figure)!")
        violation_indices = np.where(h_values < 0)[0]
        idx_to_plot = violation_indices[::max(1, len(violation_indices)//100)]
        plt.scatter(time_axis[idx_to_plot], h_values[idx_to_plot], 
                    color='red', zorder=5, s=10, label='Violation')
    else:
        print("    [Success]: CBF value is always >= 0. The system is safe.")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ==========================================
# 2. Trajectory prediction auxiliary function
# ==========================================
def get_predicted_trajecotories(current_state, control_sequence):
    predicted_state = [current_state]
    for ctrl in control_sequence:
        next_state = dynamics(predicted_state[-1], ctrl, DT)
        predicted_state.append(next_state.copy())
    return np.array(predicted_state)


# ==========================================
# 3. Static trajectory result plot
# ==========================================
def plot_simulation_result(states, save_path):
    """
    Plot the static trajectory result (adapted for 12x10 environment)
    """
    print(f"  ... Static trajectory maps are being drawn to: {save_path}")

    max_steps = int(40.0 / DT)
    if len(states) > max_steps:
        states = states[:max_steps]

    states = np.array(states)
    x_vals = states[:, 0]
    y_vals = states[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    
   
    ws_width = WS_RANGE[1] - WS_RANGE[0]
    ws_height = WS_RANGE[3] - WS_RANGE[2]
    ws_rect = Rectangle((WS_RANGE[0], WS_RANGE[2]), ws_width, ws_height,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2, label="Workspace")
    ax.add_artist(ws_rect)


    for i, obs in enumerate(STATIC_OBSTACLES):
        cx, cy, hw, hh = obs
        rect = Rectangle((cx - hw, cy - hh), hw * 2, hh * 2,
                         color='grey', alpha=0.6, label="Static Obs" if i == 0 else None)
        ax.add_artist(rect)
    
    """
  
    for x_fixed in DYN_OBS_PARAMS['x_fixed']:
        plt.plot([x_fixed, x_fixed], [0.5, 9.5], color='orange', linestyle=':', linewidth=2, label="Dyn Path")"""

  
    for i, goal in enumerate(TARGETS):
        circle = plt.Circle(goal['pos'], goal['radius'], color='purple', alpha=0.3, label="Target" if i==0 else None)
        ax.add_artist(circle)
        plt.text(goal['pos'][0], goal['pos'][1], f"G{i+1}", ha='center', va='center', fontweight='bold')

  
    plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='Ego Trajectory')
    
    if len(states) > 0:
        start_circle = plt.Circle((x_vals[0], y_vals[0]), 0.2, color='green', alpha=0.8, label='Start')
        ax.add_artist(start_circle)
        end_circle = plt.Circle((x_vals[-1], y_vals[-1]), 0.2, color='red', alpha=0.8, label='End')
        ax.add_artist(end_circle)

    plt.axis('equal') 
    plt.xlim(-1, 14)
    plt.ylim(-1, 12)
    plt.title(f'Simulation Result: Static/Dynamic Obstacles')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


#2balls
# ==========================================
# 4. Animation generation (with MPPI prediction line)
# ==========================================

def animate_simulation(states, sampled_us=[], optimal_us=None, save_path="simulation.mp4"):
    print(f"  ... Rendering animation to: {save_path}")

    max_steps = int(40.0 / DT)
    if len(states) > max_steps:
        print(f"    [Tip] Animation truncated to first 40s")
        states = states[:max_steps]
        if len(sampled_us) > max_steps:
            sampled_us = sampled_us[:max_steps]
    
    fig, ax = plt.subplots(figsize=(12, 10))

    def update(frame, states):
        ax.cla() 
        
        current_time = frame * DT
        
        if frame >= len(states): return
        
        current_state = states[frame]
        ego_x, ego_y, ego_theta = current_state[0], current_state[1], current_state[2]
        
        
        ax.set_xlim(-1, 14)
        ax.set_ylim(-1, 12)
        ax.set_aspect('equal', adjustable='box') 
        
       
        ws_width = WS_RANGE[1] - WS_RANGE[0]
        ws_height = WS_RANGE[3] - WS_RANGE[2]
        ws_rect = Rectangle((WS_RANGE[0], WS_RANGE[2]), ws_width, ws_height,
                            edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
        ax.add_artist(ws_rect)

     
        for obs in STATIC_OBSTACLES:
            cx, cy, hw, hh = obs
            rect = Rectangle((cx - hw, cy - hh), hw * 2, hh * 2, color='#404040', alpha=0.6)
            ax.add_artist(rect)
            # 边框
            rect_border = Rectangle((cx - hw, cy - hh), hw * 2, hh * 2, edgecolor='black', facecolor='none', lw=1.0)
            ax.add_artist(rect_border)

     
        for i, phase in enumerate(DYN_OBS_PARAMS['phases']):
            dist_travel = DYN_OBS_PARAMS['speed'] * current_time + phase
            cycle = DYN_OBS_PARAMS['cycle']
            span = DYN_OBS_PARAMS['span']
            
            y_offset = span - abs((dist_travel % cycle) - span)
            y_curr = DYN_OBS_PARAMS['y_min'] + y_offset
            x_curr = DYN_OBS_PARAMS['x_fixed'][i]
            
            obs_circle = plt.Circle((x_curr, y_curr), DYN_OBS_PARAMS['radius'], 
                                    color='#FF8C00', alpha=0.9, ec='black')
            ax.add_artist(obs_circle)

      
        for i, goal in enumerate(TARGETS):
            c = Circle(goal['pos'], goal['radius'], color='purple', alpha=0.3)
            ax.add_artist(c)
           
            ax.text(goal['pos'][0], goal['pos'][1], f"G{i+1}", 
                    ha='center', va='center', fontweight='bold')

        
        if len(sampled_us) > 0 and frame < len(sampled_us):
            current_samples = sampled_us[frame]
            num_samples_to_plot = min(20, len(current_samples))
            for i in range(num_samples_to_plot):
                traj = get_predicted_trajecotories(current_state, current_samples[i])
                ax.plot(traj[:, 0], traj[:, 1], color='lime', alpha=0.15, linewidth=1.0)

     
        ego_circle = plt.Circle((ego_x, ego_y), 0.3, color='dodgerblue', alpha=0.9, label='Ego', zorder=10)
        ax.add_artist(ego_circle)
        ego_border = plt.Circle((ego_x, ego_y), 0.3, facecolor='none', edgecolor='black', lw=1.0, zorder=10)
        ax.add_artist(ego_border)
        
        arrow_len = 0.6
        ax.arrow(ego_x, ego_y, arrow_len * np.cos(ego_theta), arrow_len * np.sin(ego_theta),
                 head_width=0.3, color='white', zorder=11)

       
        if frame > 0:
            traj_x = [s[0] for s in states[:frame+1]]
            traj_y = [s[1] for s in states[:frame+1]]
            ax.plot(traj_x, traj_y, color='cyan', linewidth=3, alpha=0.6)

      
        ax.set_title(f"Time: {current_time:.2f}s | Speed: {current_state[3]:.2f} m/s", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle=':', alpha=0.3)

    real_fps = int(1.0 / DT)
    
    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), 
                         interval=20, blit=False) 

    try:
        anim.save(save_path, writer='ffmpeg', fps=real_fps, extra_args=['-vcodec', 'libx264'])
        print(f"  [Success] The animation has been saved to: {save_path}")
    except Exception as e:
        print(f"  [Error] MP4 saving failed: {e}")
        try:
            anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=real_fps)
            print(f"  [Demotion] The GIF is saved: {save_path.replace('.mp4', '.gif')}")
        except Exception as e2:
             print(f"  [Critical error] Unable to save animation: {e2}")
    
    plt.close(fig)


"""
#no balls
def animate_simulation(states, sampled_us=[], optimal_us=None, save_path="simulation.mp4"):
   
    print(f"  ... Rendering animation to: {save_path}")

    max_steps = int(40.0 / DT)
    if len(states) > max_steps:
        print(f"    [Tip] Animation truncated to first 40s")
        states = states[:max_steps]
        if len(sampled_us) > max_steps:
            sampled_us = sampled_us[:max_steps]
    
    fig, ax = plt.subplots(figsize=(12, 10))

    def update(frame, states):
        ax.cla() 
        
        current_time = frame * DT
        
        if frame >= len(states): return
        
        current_state = states[frame]
        ego_x, ego_y, ego_theta = current_state[0], current_state[1], current_state[2]
        
       
        ax.set_xlim(-1, 14)
        ax.set_ylim(-1, 12)
        ax.set_aspect('equal', adjustable='box') 
        
       
        ws_width = WS_RANGE[1] - WS_RANGE[0]
        ws_height = WS_RANGE[3] - WS_RANGE[2]
        ws_rect = Rectangle((WS_RANGE[0], WS_RANGE[2]), ws_width, ws_height,
                            edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
        ax.add_artist(ws_rect)

       
        for obs in STATIC_OBSTACLES:
            cx, cy, hw, hh = obs
            rect = Rectangle((cx - hw, cy - hh), hw * 2, hh * 2, color='#404040', alpha=0.6)
            ax.add_artist(rect)
            
            rect_border = Rectangle((cx - hw, cy - hh), hw * 2, hh * 2, edgecolor='black', facecolor='none', lw=1.0)
            ax.add_artist(rect_border)

       
       
        # for i, phase in enumerate(DYN_OBS_PARAMS['phases']):
        #     dist_travel = DYN_OBS_PARAMS['speed'] * current_time + phase
        #     cycle = DYN_OBS_PARAMS['cycle']
        #     span = DYN_OBS_PARAMS['span']
        #     
        #     y_offset = span - abs((dist_travel % cycle) - span)
        #     y_curr = DYN_OBS_PARAMS['y_min'] + y_offset
        #     x_curr = DYN_OBS_PARAMS['x_fixed'][i]
        #     
        #     obs_circle = plt.Circle((x_curr, y_curr), DYN_OBS_PARAMS['radius'], 
        #                             color='#FF8C00', alpha=0.9, ec='black')
        #     ax.add_artist(obs_circle)
        # =========================================================

       
        for i, goal in enumerate(TARGETS):
            c = Circle(goal['pos'], goal['radius'], color='purple', alpha=0.3)
            ax.add_artist(c)
            ax.text(goal['pos'][0], goal['pos'][1], f"G{i+1}", 
                    ha='center', va='center', fontweight='bold')

        
        if len(sampled_us) > 0 and frame < len(sampled_us):
            current_samples = sampled_us[frame]
            num_samples_to_plot = min(20, len(current_samples))
            for i in range(num_samples_to_plot):
                traj = get_predicted_trajecotories(current_state, current_samples[i])
                ax.plot(traj[:, 0], traj[:, 1], color='lime', alpha=0.15, linewidth=1.0)

        ego_circle = plt.Circle((ego_x, ego_y), 0.3, color='dodgerblue', alpha=0.9, label='Ego', zorder=10)
        ax.add_artist(ego_circle)
        ego_border = plt.Circle((ego_x, ego_y), 0.3, facecolor='none', edgecolor='black', lw=1.0, zorder=10)
        ax.add_artist(ego_border)
        
        arrow_len = 0.6
        ax.arrow(ego_x, ego_y, arrow_len * np.cos(ego_theta), arrow_len * np.sin(ego_theta),
                 head_width=0.3, color='white', zorder=11)

  
        if frame > 0:
            traj_x = [s[0] for s in states[:frame+1]]
            traj_y = [s[1] for s in states[:frame+1]]
            ax.plot(traj_x, traj_y, color='cyan', linewidth=3, alpha=0.6)

        ax.set_title(f"Time: {current_time:.2f}s | Speed: {current_state[3]:.2f} m/s", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle=':', alpha=0.3)

    real_fps = int(1.0 / DT)
    
    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), 
                         interval=20, blit=False) 

    try:
        anim.save(save_path, writer='ffmpeg', fps=real_fps, extra_args=['-vcodec', 'libx264'])
        print(f"  [Success] Animation saved to: {save_path}")
    except Exception as e:
        print(f"  [Error] Failed to save MP4: {e}")
        try:
            anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=real_fps)
            print(f"  [Demotion] GIF saved: {save_path.replace('.mp4', '.gif')}")
        except Exception as e2:
             print(f"  [Critical error] Unable to save animation: {e2}")
    
    plt.close(fig)"""
