# main.py
import jax
import numpy as np
import os 


import config
from simulation import run_simulation
from visualization import plot_simulation_result, animate_simulation, plot_cbf_values


def main():
   
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All visualizations are saved to: {output_dir}/")

    path_trajectory_plot = os.path.join(output_dir, "tt_v3.pdf")
    path_cbf_plot = os.path.join(output_dir, "tc_v3.pdf")
    path_animation = os.path.join(output_dir, "reach_avoid_v3.mp4")


    print("--- The JAX device is being checked ---")
    try:
        devices_str = str(jax.devices())
        print(f"JAX Equipment: {devices_str}")
        
       
        if 'Gpu' not in devices_str and 'Cuda' not in devices_str:
            print("Warning: The GPU is not detected by JAX. will fall back to CPU run.")
        else:
            print("Success: JAX detected the GPU detected the GPU (CudaDevice).")
            
    except Exception as e:
        print(f"Error checking the JAX device: {e}")
    print("-------------------------")

    # Initialize state and control
    rng_key = jax.random.PRNGKey(0)
    
    initial_state = np.array([np.random.uniform(-6.0, -5.0), np.random.uniform(-6.0, -5.0), np.random.uniform(-np.pi, np.pi), 0.0])  # [x, y, theta, v]
    initial_U = np.zeros((config.N, 2))  # Initial control sequence for MPPI
    
    print(f"Initial state: {initial_state}")

   
    states, h_values, executed_controls, mppi_sampled_us, global_Us, computation_times = run_simulation(
        initial_state, initial_U, rng_key
    )

    
    print("\n--- Performance analysis (control cycle) ---")
    if len(computation_times) > 1:
        stable_times = np.array(computation_times[1:]) 
        mean_time = np.mean(stable_times)
        max_time = np.max(stable_times)
        min_time = np.min(stable_times)
        mean_freq = 1.0 / mean_time
        
        print(f"JIT Compile Time (First Call): {computation_times[0]:.4f} s")
        print(f"Stable calculation time (average): {mean_time:.4f} s")
        print(f"Stable calculation time (fastest): {min_time:.4f} s")
        print(f"Stable calculation time (slowest): {max_time:.4f} s")
        print(f"Average Control Frequency (Stable): {mean_freq:.2f} Hz")
        
        target_dt = config.dt
        target_freq = 1.0 / target_dt
        print("-------------------------")
        print(f"Target control frequency (config.dt): {target_freq:.2f} Hz (i.e., {target_dt:.4f} s)")
        
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
    
    print("⏳ When preparing to integrate the data matrix, there may be a brief lag...")
    traj_array = np.asarray(states, dtype=np.float32)  
    
    print("⏳ When preparing to integrate the CBF matrix, there may be a brief lag...")
    hx_array = np.asarray(h_values, dtype=np.float32) 
    
    print("⏳ When preparing to integrate the MPPI sampled matrix, there may be a brief lag...")
   
   
    mppi_subset = [u[:100, :, :] for u in mppi_sampled_us] 


    mppi_array = np.asarray(mppi_subset, dtype=np.float32)
    
    print(f"💾 When preparing to save the data to disk (using high compression): {npz_filename} ...")


    np.savez_compressed(
        npz_filename,
        actual_trajectory=traj_array,
        hx_values=hx_array,
        mppi_predictions=mppi_array,
        time_axis=time_axis.astype(np.float32)
    )
    print("✅ The data is perfectly completed! The volume has been greatly reduced!")
    # ===================================


    
    print("\nThe final trajectory map is being saved...")
    plot_simulation_result(states, save_path=path_trajectory_plot)

    print("\nThe CBF values are being saved...")
    plot_cbf_values(h_values, save_path=path_cbf_plot)
    
  
    print("\nThe animation is being saved (this may take a very long time)...")
    #animate_simulation(states, mppi_sampled_us, global_Us, save_path=path_animation)
    print("The animation has been saved.")
    


if __name__ == "__main__":
    main()
