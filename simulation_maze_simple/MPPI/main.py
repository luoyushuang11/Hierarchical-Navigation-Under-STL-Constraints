# main.py
import jax
import numpy as np
import os  # <--- 1. 导入 os


import config
from simulation import run_simulation
from visualization import plot_simulation_result, animate_simulation, plot_cbf_values


def main():
   
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All visualizations are saved to: {output_dir}/")

    path_trajectory_plot = os.path.join(output_dir, "maze_trajectory_v3.pdf")
    path_cbf_plot = os.path.join(output_dir, "maze_cbf_values_v3.pdf")
    path_animation = os.path.join(output_dir, "maze_environment_v3.mp4")

    
    print("--- The JAX device is being checked ---")
    try:
        devices_str = str(jax.devices())
        print(f"JAX device: {devices_str}")
        
        
        if 'Gpu' not in devices_str and 'Cuda' not in devices_str:
            print("Warning: The GPU is not detected by JAX. will fall back to CPU run。")
        else:
            print("Success: JAX detected the GPU （CudaDevice）。")
            
    except Exception as e:
        print(f"Error checking the JAX device: {e}")
    print("-------------------------")

    # Initialize state and control
    rng_key = jax.random.PRNGKey(0)
    
    initial_state = np.array([np.random.uniform(9.0, 11.5), np.random.uniform(0.5, 1.5), np.random.uniform(-np.pi, np.pi), 0.0])  # [x, y, theta, v]
    initial_U = np.zeros((config.N, 2))  # Initial control sequence for MPPI
    
    print(f"Initial state: {initial_state}")

    
    states, h_values, executed_controls, mppi_sampled_us, global_Us, computation_times = run_simulation(
        initial_state, initial_U, rng_key
    )

   
    print("\n--- Performance Analysis (Control Loop) ---")
    if len(computation_times) > 1:
        stable_times = np.array(computation_times[1:]) 
        mean_time = np.mean(stable_times)
        max_time = np.max(stable_times)
        min_time = np.min(stable_times)
        mean_freq = 1.0 / mean_time
        
        print(f"JIT compile time (first call)): {computation_times[0]:.4f} sec")
        print(f"Stable calculation time (average): {mean_time:.4f} sec")
        print(f"Stable calculation time (fastest): {min_time:.4f} sec")
        print(f"Stable calculation time (slowest): {max_time:.4f} sec")
        print(f"Average Control Frequency (Stable): {mean_freq:.2f} Hz")
        
        target_dt = config.dt
        target_freq = 1.0 / target_dt
        print("-------------------------")
        print(f"Target Control Frequency (config.dt): {target_freq:.2f} Hz (i.e., {target_dt:.4f} sec)")
        
        if mean_time > target_dt:
            print(f"Warning: Average calculation time ({mean_time:.4f}s) > Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【not】 real-time.")
        else:
            print(f"Success: Average calculation time ({mean_time:.4f}s) <= Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【real-time feasible】.")
    else:
        print("Simulation run time is too short to analyze performance.")
    print("-------------------------")
    import time
    npz_filename = os.path.join(output_dir, f"mppi_v1_data.npz")
    print(f"\n⏳ When preparing to integrate the data matrix, there may be a brief lag...")
    

    time_axis = np.arange(len(states)) * config.dt
    
    print("⏳ When converting the state matrix...")
    traj_array = np.asarray(states, dtype=np.float32) 
    
    print("⏳ When converting the CBF matrix...")
    hx_array = np.asarray(h_values, dtype=np.float32)  
    
    print("⏳ When converting the MPPI sampled matrix (executing aggressive truncation and compression)...")
    # [core volume reduction]
    # 1. List comprehension: Keep only the first 100 trajectories at each time step, discard the rest
    mppi_subset = [u[:100, :, :] for u in mppi_sampled_us]
    # 2. Type conversion: Convert to float32 single precision, reduce volume by half
    mppi_array = np.asarray(mppi_subset, dtype=np.float32)

    print(f"💾 When saving data to disk (using high compression rate): {npz_filename} ...")
   
    np.savez_compressed(
        npz_filename,
        actual_trajectory=traj_array,
        hx_values=hx_array,
        mppi_predictions=mppi_array,
        time_axis=time_axis.astype(np.float32)
    )
    print("✅ The data is perfectly completed! The volume has been greatly reduced！")
    # ===================================

    

    print("\nCBF change plots are being saved...")
    #plot_cbf_values(h_values, save_path=path_cbf_plot)

    
    print("\nCBF change plots are being saved...")
    #plot_simulation_result(states, save_path=path_trajectory_plot)

    print("\nCBF change plots are being saved...")
    #plot_cbf_values(h_values, save_path=path_cbf_plot)
    
    
    # print("\nCBF change plots are being saved...")
    #animate_simulation(states, mppi_sampled_us, global_Us, save_path=path_animation)
    # print("The animation is saved。")
    

    
    

if __name__ == "__main__":
    main()