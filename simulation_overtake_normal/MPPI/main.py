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

    path_trajectory_plot = os.path.join(output_dir, "tt.pdf")
    path_cbf_plot = os.path.join(output_dir, "tc_cbf_values.pdf")
  
    path_animation = os.path.join(output_dir, "fast_overtake.mp4")

  
    print("--- The JAX device is being checked ---")
    try:
        devices_str = str(jax.devices())
        print(f"JAX Equipment: {devices_str}")
        if 'Gpu' not in devices_str and 'Cuda' not in devices_str:
            print("Warning: The GPU is not detected by JAX. will fall back to CPU run。")
        else:
            print("Success： JAX detected GPU （CudaDevice）。")
    except Exception as e:
        print(f"Error checking the JAX device: {e}")
    print("-------------------------")

    # Initialize state and control
    rng_key = jax.random.PRNGKey(0)
    
    
    start_x = np.random.uniform(0.0, 1.0)
    start_y = np.random.uniform(0.6, 1.4)
    start_theta = np.random.uniform(0, np.pi / 4)
    start_v = 1.0
    
  
    initial_state = np.array([start_x, start_y, start_theta, start_v])
    

    initial_U = np.zeros((config.N, 2))  
    
    print(f"Initial state: {initial_state}")

    
    states, h_values, executed_controls, mppi_sampled_us, global_Us, computation_times = run_simulation(
        initial_state, initial_U, rng_key
    )

    
    print("\n--- Performance analysis (control cycle) ---")
    if len(computation_times) > 1:
        stable_times = np.array(computation_times[1:]) 
        mean_time = np.mean(stable_times)
        min_time = np.min(stable_times)
        max_time = np.max(stable_times)
        
        print(f"JIT compilation time (first call): {computation_times[0]:.4f} seconds")
        print(f"Stable computation time (average): {mean_time:.4f} seconds")
        print(f"Stable computation time (fastest): {min_time:.4f} seconds")
        print(f"Stable computation time (slowest): {max_time:.4f} seconds")
        
        target_dt = config.dt
        if mean_time > target_dt:
            print(f"Warning: Average computation time ({mean_time:.4f}s) > Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【not】 real-time.")
        else:
            print(f"Success: Average computation time ({mean_time:.4f}s) <= Target step size ({target_dt:.4f}s)")
            print("  ==> This means your controller is 【real-time feasible】.")
    else:
        print("Simulation run time is too short to analyze performance.")
    print("-------------------------")


    import time
    npz_filename = os.path.join(output_dir, f"mppi_slow_data.npz")
    print(f"\n⏳ When preparing to integrate the data matrix, there may be a brief lag...")

    time_axis = np.arange(len(states)) * config.dt
    
    print("⏳ Transitioning state matrix...")
    traj_array = np.asarray(states)
    
    print("⏳ The CBF matrix is being converted...")
    hx_array = np.asarray(h_values)
    
    print("⏳ Converting the MPPI sampling matrix (this step is the most time-consuming)...")
    mppi_array = np.asarray(mppi_sampled_us)
    
    print(f"💾 Writing data to hard disk (using uncompressed ultra-fast saving): {npz_filename} ...")
    np.savez(
        npz_filename,
        actual_trajectory=traj_array,
        hx_values=hx_array,
        mppi_predictions=mppi_array,
        time_axis=time_axis
    )
    print("✅ The data is perfectly completed!")
    # ===================================


   
    print("\nThe final trajectory map is being saved...")
  
    plot_simulation_result(states, save_path=path_trajectory_plot)

    print("\nCBF change plots are being saved...")
  
    plot_cbf_values(h_values, save_path=path_cbf_plot)
    
  
    print("\nAnimations are being generated and saved (MP4)...")
    #animate_simulation(states, sampled_us=mppi_sampled_us, optimal_us=global_Us, save_path=path_animation)
    print(f"The animation is saved: {path_animation}")
    

if __name__ == "__main__":
    main()