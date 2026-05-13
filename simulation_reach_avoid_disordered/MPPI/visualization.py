# visualization.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import config

from model import dynamics

def plot_cbf_values(h_values, save_path):
    """
    Plot the change in CBF barrier function value (h) over time.
    """
    print(f"  ... is plotting CBF (h(x)) to: {save_path}")
    
    # 1. Create a timeline
    time_axis = np.linspace(0, config.T, num=len(h_values))
    
    fig = plt.figure(figsize=(12, 6))
    
    # 2. Plotting the value of h(x).
    plt.plot(time_axis, h_values, label='CBF Value h(x, t)')
    
    # 3. Drawing a safety boundary (h=0)
    plt.axhline(0, color='red', linestyle='--', label='Safety Boundary (h=0)')
    
    # 4. Format the chartt the chart
    plt.title('CBF Value (h) Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('CBF Value h(x, t)')
    plt.legend()
    plt.grid(True)
    # Prevents the y-axis range from being too small
    if len(h_values) > 0:
        plt.ylim(bottom=min(np.min(h_values), -0.5))
    
    # 5. Check for violations (h < 0)iolations (h < 0)
    h_array = np.array(h_values)
    if np.any(h_array < 0):
        print("    Warning: CBF values are less than 0 at some point (already marked in red in the graph)!")
        violation_times = time_axis[h_array < 0]
        violation_values = h_array[h_array < 0]
        plt.scatter(violation_times, violation_values, color='red', zorder=5, 
                    label='Safety Violation (h<0)')
        plt.legend()
    else:
        print("    Success: CBF values are always >= 0. The system remains safe.")
            
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def get_predicted_trajecotories(current_state, control_sequence):
    predicted_state = [current_state]
    for ctrl in control_sequence:
        next_state = dynamics(predicted_state[-1], ctrl)
        predicted_state.append(next_state.copy())
    return np.array(predicted_state)

def plot_simulation_result(states, save_path):
    """
    Plot the static trajectory.
    """
    print(f"  ... is plotting the trajectory to: {save_path}")
    
    # Extract trajectory data
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    workspace_radius = 10.0
    
    # --- 1. Drawing a workspace ---
    workspace_circle = plt.Circle((0, 0), workspace_radius, color='black', fill=False,
                                  linestyle='--', linewidth=2, label="Workspace")
    ax.add_artist(workspace_circle)

    # --- 2. Define and draw obstacles ---
    """
    # Disorganized
    obstacles = [
        [-2.5, 0.0, 1.5], [ 0.0, -5.0, 1.0], [ 2.5, 0.0, 1.0],
        [ 0.0, 5.0, 0.75], [ -2.5, -3.0, 0.75], [ -1.0, 2.5, 0.75],
        [ 5.0, 5.0, 0.75], [ 6.0, 0.0, 1.0], [ -5.0, 2.5, 0.75],
    ]"""
    
    """
    # Orderly 9 obstacles
    obstacles = [
        [-2.5, 0.0, 1.0], [ 6.0, -2.5, 1.0], [ 2.5, 0.0, 1.0],
        [ 2.5, 5.0, 1.0], [ -2.5, -2.5, 1.0], [ 6.0, 2.5, 1.0],
        [ 6.0, 5.0, 1.0], [ 6.0, 0.0, 1.0], [ -2.5, 2.5, 1.0],
    ]
    

    """
    # Orderly 13 obstacles
    obstacles = [
        [-2.5, 0.0, 0.75], [ 6.0, -2.5, 0.75], [ 2.5, 0.0, 0.75],
        [ 2.5, 5.0, 0.75], [ -2.5, -2.5, 0.75], [ 6.0, 2.5, 0.75],
        [ 6.0, 5.0, 0.75], [ 6.0, 0.0, 0.75], [ -2.5, 2.5, 0.75],
        [ 0.0, -2.5, 0.75], [ 0.0, 0.0, 0.75], [ 0.0, 2.5, 0.75], [ 0.0, 5.0, 0.75],
    ]

    for i, obs in enumerate(obstacles):
        label = "Obstacle" if i == 0 else None
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='grey', fill=True,
                            alpha=0.5, label=label)
        ax.add_artist(circle)

    # --- 3. Define and map the target area ---
    """
    # Disorganized
    subgoals = [
        [2.5,  -2.5, 1.0, 0], [2.0,  3.0, 1.0, 1], [-2.5, 5.0, 1.0, 2]
    ]"""
    
    # Orderly
    subgoals = [
        [2.5,  -2.5, 1.0, 0], [2.5,  2.5, 1.0, 1], [-2.5, 5.0, 1.0, 2]
    ]


    for i, goal in enumerate(subgoals):
        gx, gy, gr, idx = goal
        label = "Target Region" if i == 0 else None
        circle = plt.Circle((gx, gy), gr, color='purple', fill=True,
                            linestyle='--', alpha=0.3, label=label)
        ax.add_artist(circle)
        plt.scatter(gx, gy, s=50, color="purple", alpha=0.6)
        plt.text(gx, gy, f"G{int(idx)}", ha='center', va='center', 
                 fontweight='bold', color='black', fontsize=12)

    # --- 4. Drawing a trajectory ---
    plt.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4, alpha=0.5, color='tab:blue')
    
    step = max(1, int(len(states)/30))
    for i in range(0, len(states), step):
        x, y, theta, _ = states[i]
        arrow_len = 0.4
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red')

    # --- 5. Draw Start/End ---
    if len(states) > 0:
        plt.scatter(states[0][0], states[0][1], s=200, color="green", edgecolors='black', 
                    alpha=0.9, label="Init. Position", zorder=10)
        plt.scatter(states[-1][0], states[-1][1], s=200, color="red", edgecolors='black', marker='X',
                    alpha=0.9, label="End Position", zorder=10)

    # --- 6. Formatting Charts ---
    plt.title('Simulation Result: Trajectory & Environment')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    margin = 1.0
    plt.xlim([-workspace_radius - margin, workspace_radius + margin])
    plt.ylim([-workspace_radius - margin, workspace_radius + margin])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def animate_simulation(states, sampled_us=[], optimal_us=None, save_path="simulation.mp4"):
    """
    Generate MP4 animations for the simulation process.
    Fixes:
    1. Use the correct obstacle list specified by the user.
    2. The bot appears as a solid with a radius of 0.2m.
    3. Adjust to real-time playback speed (0.02 FPS) based on dt=50s.
    """
    print(f"  ... is rendering animation frames to: {save_path}")
    
    workspace_radius = 10.0
    robot_radius = 0.2 
    
    # Try to read dt, default is 0.02
    try:
        dt = config.DT
    except AttributeError:
        dt = 0.02
        
    realtime_fps = int(1 / dt) # 50 FPS
    frame_interval_ms = int(dt * 1000) # 20 ms

    # --- Here is an update for the correct obstacle data [x, y, radius] ---
    """
    # Disorganized
    obstacles = [
        [-2.5, 0.0, 1.5],
        [ 0.0, -5.0, 1.0],
        [ 2.5, 0.0, 1.0],
        [ 0.0, 5.0, 0.75], 
        [ -2.5, -3.0, 0.75],
        [ -1.0, 2.5, 0.75],
        [ 5.0, 5.0, 0.75],
        [ 6.0, 0.0, 1.0],
        [ -5.0, 2.5, 0.75],
    ]

    subgoals = [
        [2.5,  -2.5, 1.0, 0], 
        [2.0,  3.0, 1.0, 1], 
        [-2.5, 5.0, 1.0, 2]
    ]"""

    

    """
    # Orderly
    obstacles = [
        [-2.5, 0.0, 1.0], [ 6.0, -2.5, 1.0], [ 2.5, 0.0, 1.0],
        [ 2.5, 5.0, 1.0], [ -2.5, -2.5, 1.0], [ 6.0, 2.5, 1.0],
        [ 6.0, 5.0, 1.0], [ 6.0, 0.0, 1.0], [ -2.5, 2.5, 1.0],
    ] 
    
     
    """
    obstacles = [
        [-2.5, 0.0, 0.75], [ 6.0, -2.5, 0.75], [ 2.5, 0.0, 0.75],
        [ 2.5, 5.0, 0.75], [ -2.5, -2.5, 0.75], [ 6.0, 2.5, 0.75],
        [ 6.0, 5.0, 0.75], [ 6.0, 0.0, 0.75], [ -2.5, 2.5, 0.75],
        [ 0.0, -2.5, 0.75], [ 0.0, 0.0, 0.75], [ 0.0, 2.5, 0.75], [ 0.0, 5.0, 0.75],
    ] 

    
    subgoals = [
        [2.5,  -2.5, 1.0, 0], 
        [2.5,  2.5, 1.0, 1], 
        [-2.5, 5.0, 1.0, 2]
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    def update(frame, states):
        ax.cla()  # Clear the current axis
        
        # --- 1. Set the axis ---
        ax.set_xlim([-workspace_radius - 1, workspace_radius + 1])
        ax.set_ylim([-workspace_radius - 1, workspace_radius + 1])
        ax.set_aspect('equal', adjustable='box')

        # --- 2. Drawing Environment ---
        workspace_circle = plt.Circle((0, 0), workspace_radius, color='black', fill=False,
                                      linestyle='--', linewidth=2, label="Workspace")
        ax.add_artist(workspace_circle)
        
        # Draw the right obstacles
        for i, obs in enumerate(obstacles):
            # obs: [x, y, radius]
            ox, oy, r = obs[0], obs[1], obs[2]
            label = "Obstacle" if i == 0 else None
            circle = plt.Circle((ox, oy), r, color='grey', fill=True, alpha=0.5, label=label)
            ax.add_artist(circle)

        # Map the right target area
        for i, goal in enumerate(subgoals):
            gx, gy, gr, idx = goal
            label = 'Target Region' if i == 0 else None
            circle = plt.Circle((gx, gy), gr, color='purple', fill=True, linestyle='--', alpha=0.3, label=label)
            ax.add_artist(circle)
            # Mark the target number
            ax.text(gx, gy, f"G{int(idx)}", ha='center', va='center', color='purple', fontsize=8, fontweight='bold')
        
        # --- 3. Plot the MPPI Forecast Track ---
        if len(sampled_us) > 0 and frame < len(sampled_us):
            num_trajs_plotted = np.minimum(20, sampled_us[frame].shape[0])
            for idx in range(num_trajs_plotted):
                pred_traj = get_predicted_trajecotories(states[frame], sampled_us[frame][idx])
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], color="k", alpha=0.05)
                
        if optimal_us is not None and frame < len(optimal_us):
            opt_pred_traj = get_predicted_trajecotories(states[frame], optimal_us[frame])
            plt.plot(opt_pred_traj[:, 0], opt_pred_traj[:, 1], color="orange", alpha=0.8, linewidth=2, label="Predicted Traj.")

        # --- 4. Drawing Vehicles and Actual Trajectories ---
        # Draw a historical trajectory
        current_x_vals = [state[0] for state in states[:frame+1]]
        current_y_vals = [state[1] for state in states[:frame+1]]
        plt.plot(current_x_vals, current_y_vals, '-o', markersize=2, linewidth=1, alpha=0.5, label='Actual Trajectory')
        
        # Get the current status
        x, y, theta, _ = states[frame]
        
        # [Modified] Draw Robot Solids (0.2m radius)
        robot_circle = plt.Circle((x, y), robot_radius, color='tab:blue', fill=True, 
                                  alpha=0.9, zorder=10, label='Robot')
        ax.add_artist(robot_circle)
        
        # [Modified] Draw the robot towards the arrow
        arrow_len = robot_radius * 1.5 
        adx = arrow_len * np.cos(theta)
        ady = arrow_len * np.sin(theta)
        plt.arrow(x, y, adx, ady, head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=11)

        # --- 5. Draw the start and end points (consistent with plot_simulation_result---
        if len(states) > 0:
            # Starting point
            plt.scatter(states[0][0], states[0][1], s=150, color="green", edgecolors='black', 
                       alpha=0.8, label="Init. Position", zorder=9)
            # Finish line
            plt.scatter(states[-1][0], states[-1][1], s=150, color="red", edgecolors='black', marker='X',
                       alpha=0.8, label="End Position", zorder=9)

        # --- 6. Formatting ---
        current_time = frame * dt
        plt.title(f'Simulation Animation (Time: {current_time:.2f}s)')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize='x-small')

    # --- Create an animation ---
    anim = FuncAnimation(fig, update, frames=len(states), fargs=(states,), 
                         interval=frame_interval_ms, blit=False)

    print("Encoding and saving animations... (This may take some time)")
    
    # --- Save as MP4 ---
    try:
        anim.save(save_path, writer='ffmpeg', fps=realtime_fps, dpi=150)
        print(f"The animation is saved to: {save_path}")
    except Exception as e:
        print(f"Saving MP4 failed: {e}")
        backup_path = save_path.replace(".mp4", ".gif")
        anim.save(backup_path, writer='pillow', fps=realtime_fps)

    plt.close(fig)