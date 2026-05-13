# main.py
import jax
import numpy as np
import os 


import config
from simulation import run_simulation
from visualization import plot_simulation_result, animate_simulation, plot_cbf_values


def main():
    # --- 2. Create an output directory and file path ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All visualizations are saved to: {output_dir}/")

    path_trajectory_plot = os.path.join(output_dir, "tt_v3.pdf")
    path_cbf_plot = os.path.join(output_dir, "tc_v3.pdf")
    path_animation = os.path.join(output_dir, "reach_avoid_v3.mp4")

    # Check if JAX can see the GPU
    print("--- Checking JAX Devices ---")
    try:
        devices_str = str(jax.devices())
        print(f"JAX Devices: {devices_str}")
        
     
        if 'Gpu' not in devices_str and 'Cuda' not in devices_str:
            print("Warning: The GPU is not detected by JAX. will fall back to CPU run。")
        else:
            print("Success: JAX has detected the GPU (CudaDevice)。")
            
    except Exception as e:
        print(f"Error occurred while checking JAX devices: {e}")
    print("-------------------------")

    # Initialize state and control
    rng_key = jax.random.PRNGKey(0)
  
    initial_state = np.array([np.random.uniform(-6.0, -5.0), np.random.uniform(-6.0, -5.0), np.random.uniform(-np.pi, np.pi), 0.0])  # [x, y, theta, v]
    initial_U = np.zeros((config.N, 2))  # Initial control sequence for MPPI
    
    print(f"Initial state: {initial_state}")

    # Run simulation
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
        
        print(f"JIT Compilation Time (First Call): {computation_times[0]:.4f} seconds")
        print(f"Stable Computation Time (Average): {mean_time:.4f} seconds")
        print(f"Stable Computation Time (Fastest): {min_time:.4f} seconds")
        print(f"Stable Computation Time (Slowest): {max_time:.4f} seconds")
        print(f"Average Control Frequency (Stable): {mean_freq:.2f} Hz")
        
        target_dt = config.dt
        target_freq = 1.0 / target_dt
        print("-------------------------")
        print(f"Target Control Frequency (config.dt): {target_freq:.2f} Hz (i.e., {target_dt:.4f} seconds)")
        
        if mean_time > target_dt:
            print(f"Warning: Average computation time ({mean_time:.4f}s) > Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【not】 real-time capable.")
        else:
            print(f"Success: Average computation time ({mean_time:.4f}s) <= Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【real-time feasible】.")
            
    else:
        print("Simulation runtime is too short to analyze performance.")
    print("-------------------------")

    import time
    npz_filename = os.path.join(output_dir, f"mppi_v1_data.npz")
    print(f"\n⏳ When preparing to integrate the data matrix, there may be a brief lag...")
    
    # Generate a physical timeline
    time_axis = np.arange(len(states)) * config.dt
    
    print("⏳ Transitioning state matrix...")
    traj_array = np.asarray(states, dtype=np.float32) 
    
    print("⏳ Transitioning CBF matrix...")
    hx_array = np.asarray(h_values, dtype=np.float32)
    
    print("⏳ Transitioning MPPI sampling matrix (executing aggressive truncation and瘦身)...")
  

    mppi_subset = [u[:100, :, :] for u in mppi_sampled_us] 
  
    mppi_array = np.asarray(mppi_subset, dtype=np.float32)
    
    print(f"💾 Writing data to hard disk (saving with high compression): {npz_filename} ...")
    
    np.savez_compressed(
        npz_filename,
        actual_trajectory=traj_array,
        hx_values=hx_array,
        mppi_predictions=mppi_array,
        time_axis=time_axis.astype(np.float32)
    )
    print("✅ Data writing completed successfully! Volume has been significantly reduced!")
    # ===================================



    print("\nSaving final trajectory plot...")
    plot_simulation_result(states, save_path=path_trajectory_plot)

    print("\nSaving CBF change plot...")
    plot_cbf_values(h_values, save_path=path_cbf_plot)
    
  
    print("\nSaving animation (this may take a long time)...")
    #animate_simulation(states, mppi_sampled_us, global_Us, save_path=path_animation)
    print("Animation saved.")
    


if __name__ == "__main__":
    main()
