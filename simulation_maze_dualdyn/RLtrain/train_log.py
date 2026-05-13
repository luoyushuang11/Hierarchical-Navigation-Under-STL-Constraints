import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches 


from config import TGPOConfig
from envs.scenarios import get_scenario
from models.networks import Actor, Critic
from planning.rollout import rollout_episode
from planning.uniform import sample_uniform_times
from planning.mcmc import metropolis_hastings_sampling
from planning.elite import EliteBuffer
from utils.stljax_evaluator import STLEvaluator

from algo.ppo import calculate_gae, compute_log_prob, update_step

# ==========================================
# 1. Trajectory Visualizer (for rectangular boundaries and obstacles)
# ==========================================
def debug_plot(phys_trajs, scores, scenario, iter_idx, filename="final_best_traj.png"):
    best_idx = jnp.argmax(scores)
    best_score = scores[best_idx]
    traj = np.array(phys_trajs[best_idx])  # (T, 4)
    
    plt.figure(figsize=(12, 10)) 
    ax = plt.gca()
    
    #  ( [0, 12] x [0, 10])
    ws_w, ws_h = 12.0, 10.0
    workspace_rect = patches.Rectangle((0, 0), ws_w, ws_h, color='black', fill=False, 
                                     linestyle='--', linewidth=2.0, label='Workspace')
    ax.add_patch(workspace_rect)

    # (Start Zone)
    if scenario.name == "local_minima_trap":
        start_rect = patches.Rectangle((9.0, 0.5), 2.5, 1.0, 
                                     linewidth=2, edgecolor='blue', facecolor='none', 
                                     linestyle=':', label='Start Zone')
    elif scenario.name == "dynamic_overtaking":
        start_rect = patches.Rectangle((0.0, 0.6), 1.0, 0.8, 
                                     linewidth=2, edgecolor='blue', facecolor='none', 
                                     linestyle=':', label='Start Zone')
    else:
        # Default
        start_rect = patches.Rectangle((-6.0, -6.0), 1.0, 1.0, 
                                     linewidth=2, edgecolor='blue', facecolor='none', 
                                     linestyle=':', label='Start Zone')
    ax.add_patch(start_rect)
    
    # Draw obstacles
    if scenario.obstacles is not None:
        obs_data = np.array(scenario.obstacles)
        if obs_data.shape[1] == 4: 
            for obs in obs_data:
                cx, cy, w, h = obs
                lx = cx - w / 2.0
                ly = cy - h / 2.0
                rect = patches.Rectangle((lx, ly), w, h, color='red', alpha=0.5, label='Obstacle')
                ax.add_patch(rect)
        else: 
            for obs in obs_data:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5, label='Obstacle')
                ax.add_patch(circle)
        
    # Draw sub-goals
    colors = ['green', 'blue', 'purple']
    for i, goal in enumerate(scenario.subgoals):
        circle = plt.Circle((goal[0], goal[1]), goal[2], color=colors[i % 3], alpha=0.3, label=f'Goal {i}')
        ax.add_patch(circle)
        plt.text(goal[0], goal[1], f"G{i}", ha='center', va='center', fontweight='bold')

    # Draw trajectory
    T = traj.shape[0]
    for t in range(T - 1):
        plt.plot(traj[t:t+2, 0], traj[t:t+2, 1], color=plt.cm.cool(t/T), linewidth=2)
    
    plt.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'kx', markersize=8, label='End')
    
    plt.title(f"Scenario: {scenario.name}\nIter: {iter_idx} | Best Score: {best_score:.2f}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    
    plt.xlim(-1.0, ws_w + 1.0)
    plt.ylim(-1.0, ws_h + 1.0)
    plt.grid(True)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.savefig(filename)
    plt.close()
    print(f"\n[Visual] The best trajectory map has been saved to: {filename}")

# ==========================================
# 2. Trainer class
# ==========================================
class TGPOTrainer:
    def __init__(self, seed=0, scenario_name="reach_avoid_simple"): 
        self.rng = jax.random.PRNGKey(seed)
        
        # 1. Environment and scene
        self.scenario = get_scenario(scenario_name)
        print(f"[Init] Scenario: {self.scenario.name}")
        
        # 2. Network initialization
        dummy_aug = jnp.zeros((1, TGPOConfig.AUG_STATE_DIM))
        dummy_time = jnp.zeros((1, self.scenario.subgoals.shape[0])) 
        
        self.actor = Actor(action_dim=2)
        self.critic = Critic()
        
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        actor_params = self.actor.init(k1, dummy_aug, dummy_time)
        critic_params = self.critic.init(k2, dummy_aug, dummy_time)
        
        # 3. Optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(TGPOConfig.GRAD_NORM_CLIP),
            optax.adamw(learning_rate=TGPOConfig.LR, weight_decay=TGPOConfig.WEIGHT_DECAY)
        )
        
        self.actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=optimizer)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=critic_params, tx=optimizer)
            
        # 4. Plan components
        self.elite_buffer = EliteBuffer(n_vars=self.scenario.subgoals.shape[0])
        self.elite_times = self.elite_buffer.times
        self.elite_scores = self.elite_buffer.scores
        self.stl_evaluator = STLEvaluator(self.scenario.obstacles, self.scenario.subgoals)
        
        # 5. Initialize the log
        self.log_history = {
            "iter": [], "reward": [], "success": [], "entropy": [], "critic_loss": [], "actor_loss": [] 
        }
        
        print("[Init] Ready.")

    def save_checkpoint(self, filename="local_minima_trap_model_noballs_s_2.msgpack"):
        ckpt = {'actor': self.actor_state, 'critic': self.critic_state}
        with open(filename, 'wb') as f:
            f.write(serialization.to_bytes(ckpt))
        print(f"[Checkpoint] ✅ Model saved to: {filename}")

    def load_checkpoint(self, filename="local_minima_trap_model_noballs_s_1.msgpack"):
        if not os.path.exists(filename):
            print(f"[Checkpoint] ⚠️ If no checkpoint {filename} is found, a new training will begin。")
            return
        
        print(f"[Checkpoint] 🔄 Loading checkpoints: {filename} ...")
        with open(filename, 'rb') as f:
            bytes_data = f.read()
        
        target = {'actor': self.actor_state, 'critic': self.critic_state}
        restored = serialization.from_bytes(target, bytes_data)
        
        self.actor_state = restored['actor']
        self.critic_state = restored['critic']
        print(f"[Checkpoint] ✅ Model loading successfully! Continue training...")

    # --- [NEW] Saving Training Log Data ---
    def save_logs(self, filename="trap_training_logs.npz"):
        """Save all curve data in log_history to a local file"""
        np.savez(filename, **self.log_history)
        print(f"[Log] ✅ The maze scene training log has been fully saved to: {filename}")

    def plot_training_curves(self, filename="training_curves.png"):
        iters = self.log_history["iter"]
        if len(iters) == 0: return

        plt.figure(figsize=(15, 12))
        plt.suptitle(f'Training Metrics: {self.scenario.name}', fontsize=16)
        
        plt.subplot(3, 2, 1); plt.plot(iters, self.log_history["reward"], 'b-'); plt.title("Avg Reward"); plt.grid(True)
        plt.subplot(3, 2, 2); plt.plot(iters, self.log_history["success"], 'g-'); plt.title("Success Rate (%)"); plt.ylim(-5, 105); plt.grid(True)
        plt.subplot(3, 2, 3); plt.plot(iters, self.log_history["actor_loss"], 'm-'); plt.title("Actor Loss"); plt.grid(True)
        plt.subplot(3, 2, 4); plt.plot(iters, self.log_history["critic_loss"], 'k-'); plt.title("Critic Loss"); plt.grid(True)
        plt.subplot(3, 2, 5); plt.plot(iters, self.log_history["entropy"], 'r-'); plt.title("Entropy"); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[Visual] The training graph is saved to: {filename}")

    # --- [Modified] Added log_filename parameters ---
    def train(self, num_iterations, log_filename="trap_logs.npz"): 
        self.load_checkpoint()
        print(f"Start Training for {num_iterations} iterations on [{self.scenario.name}]...")
        
        def reset_states(key, batch_size):
            k1, k2, k3 = jax.random.split(key, 3)
            
            if self.scenario.name == "local_minima_trap":
                pos_x = jax.random.uniform(k1, shape=(batch_size, 1), minval=9.0, maxval=11.5)
                pos_y = jax.random.uniform(k2, shape=(batch_size, 1), minval=0.5, maxval=1.5)
                theta = jax.random.uniform(k3, shape=(batch_size, 1), minval=-jnp.pi, maxval=jnp.pi)
                v = jnp.zeros((batch_size, 1))
                
            elif self.scenario.name == "dynamic_overtaking":
                pos_x = jax.random.uniform(k1, shape=(batch_size, 1), minval=0.0, maxval=1.0)
                pos_y = jax.random.uniform(k2, shape=(batch_size, 1), minval=0.6, maxval=1.4)
                theta = jax.random.uniform(k3, shape=(batch_size, 1), minval=0.0, maxval=jnp.pi/4.0)
                v = jnp.ones((batch_size, 1)) 
                
            else:
                pos_x = jax.random.uniform(k1, shape=(batch_size, 1), minval=-6.0, maxval=-5.0)
                pos_y = jax.random.uniform(k2, shape=(batch_size, 1), minval=-6.0, maxval=-5.0)
                theta = jax.random.uniform(k3, shape=(batch_size, 1), minval=-jnp.pi, maxval=jnp.pi)
                v = jnp.zeros((batch_size, 1))         
            
            pos = jnp.concatenate([pos_x, pos_y], axis=-1)
            phys = jnp.concatenate([pos, theta, v], axis=-1)
            logic = jnp.tile(jnp.array([0., 0., 0., 0., 1.]), (batch_size, 1))
            return jnp.concatenate([phys, logic], axis=-1)

        last_phys_trajs = None
        last_robustness_scores = None

        # --- Main cycle ---
        for it in range(num_iterations):
            self.rng, key_iter = jax.random.split(self.rng)
            
            # 1. Mixed sampling
            n_batch = TGPOConfig.BATCH_SIZE
            n_uni = int(TGPOConfig.RATIO_UNI * n_batch)
            n_mcmc = int(TGPOConfig.RATIO_MCMC * n_batch)
            n_elite = n_batch - n_uni - n_mcmc
            n_vars = self.scenario.subgoals.shape[0]
            
            key_uni, key_mcmc, key_elite, key_rollout = jax.random.split(key_iter, 4)
            
            t_uni = sample_uniform_times(key_uni, n_uni, n_vars)
            
            init_s0_mcmc = reset_states(key_mcmc, n_mcmc)
            init_t0_mcmc = sample_uniform_times(key_mcmc, n_mcmc, n_vars)
            t_mcmc = metropolis_hastings_sampling(
                key_mcmc, self.critic.apply, self.critic_state.params, init_s0_mcmc, init_t0_mcmc
            )
            
            t_elite_samples = self.elite_buffer.sample(self.elite_times, n_elite, key_elite)
            time_batch = jnp.concatenate([t_uni, t_mcmc, t_elite_samples], axis=0)
            
            # 2. Rollout
            init_states = reset_states(key_rollout, n_batch)
            phys_trajs, transitions = rollout_episode(
                key_rollout, self.actor.apply, self.actor_state.params, init_states, time_batch, self.scenario
            )
            last_phys_trajs = phys_trajs
            
            # 3. Evaluation
            robustness_scores = self.stl_evaluator.evaluate(phys_trajs)
            last_robustness_scores = robustness_scores 
            
            self.elite_times, self.elite_scores = self.elite_buffer.update(
                self.elite_times, self.elite_scores, time_batch, robustness_scores
            )
            
            # 4. Data Preprocessing & GAE
            obs = transitions['obs']
            actions = transitions['action']
            raw_rewards = transitions['reward'] 
            rewards = raw_rewards / 100.0       
            next_obs = transitions['next_obs']
            
            time_seq = jnp.tile(jnp.expand_dims(time_batch, 1), (1, int(TGPOConfig.TIME_SCALE), 1))
            flat_obs = obs.reshape(-1, TGPOConfig.AUG_STATE_DIM)
            flat_next_obs = next_obs.reshape(-1, TGPOConfig.AUG_STATE_DIM)
            flat_time = time_seq.reshape(-1, n_vars)
            
            values = self.critic.apply(self.critic_state.params, flat_obs, flat_time)
            next_values = self.critic.apply(self.critic_state.params, flat_next_obs, flat_time)
            values = values.reshape(n_batch, -1)
            next_values = next_values.reshape(n_batch, -1)
            
            dones = jnp.zeros_like(rewards)
            dones = dones.at[:, -1].set(1.0)
            
            advantages, targets = calculate_gae(
                rewards, values, next_values, dones,
                gamma=TGPOConfig.GAMMA, gae_lambda=TGPOConfig.GAE_LAMBDA
            )
            
            curr_mean, curr_log_std = self.actor.apply(self.actor_state.params, flat_obs, flat_time)
            flat_actions = actions.reshape(-1, 2)
            old_log_probs = compute_log_prob(curr_mean, curr_log_std, flat_actions)
            
            batch_data = {
                'aug_state': flat_obs, 
                'time_vars': flat_time,
                'action': flat_actions, 
                'log_prob': old_log_probs
            }
            flat_adv = advantages.reshape(-1)
            flat_target = targets.reshape(-1)
            
            # 5. PPO updates
            for _ in range(TGPOConfig.PPO_EPOCHS):
                self.actor_state, self.critic_state, info = update_step(
                    self.actor_state, self.critic_state, batch_data, flat_adv, flat_target,
                    self.actor.apply, self.critic.apply
                )
            
            # 6. Data logging and log output
            # [Key Modification]: Removed the if restriction to ensure that data is recorded in each round
            best_idx = jnp.argmax(robustness_scores)
            loss_a = info['loss_actor'].item()
            loss_c = info['loss_critic'].item()
            entropy = info['entropy'].item()
            mean_raw_reward = jnp.mean(jnp.sum(raw_rewards, axis=1)).item()
            succ_rate = jnp.mean(robustness_scores > 0).item() * 100.0 

            self.log_history["iter"].append(it)
            self.log_history["reward"].append(mean_raw_reward)
            self.log_history["success"].append(succ_rate)
            self.log_history["entropy"].append(entropy)
            self.log_history["critic_loss"].append(loss_c)
            self.log_history["actor_loss"].append(loss_a)

            # Terminal printing is still every 10 rounds
            if it % 10 == 0:
                best_t = time_batch[best_idx].tolist()
                if isinstance(best_t, float): best_t = [best_t]
                t_str = "[" + ", ".join([f"{t:.1f}" for t in best_t]) + "]"
                
                print(f"It {it:4d} | R:{mean_raw_reward:5.1f} | S:{succ_rate:.0f}% | "
                      f"Rob:{robustness_scores[best_idx]:5.2f} | AL:{loss_a:6.3f} | "
                      f"CL:{loss_c:6.3f} | Ent:{entropy:5.2f} | Tm:{t_str}")

        print("\n=== Training Finished ===")
        self.save_checkpoint() 
        
        # --- [NEW] Automatic Saving of Lossless Data After Training ---
        self.save_logs(filename=log_filename)
        
        self.plot_training_curves(f"{self.scenario.name}_curves.png")
        debug_plot(
            last_phys_trajs, 
            last_robustness_scores, 
            self.scenario, 
            num_iterations, 
            filename=f"{self.scenario.name}_result.png"
        )

# ==========================================
# 入口函数
# ==========================================
if __name__ == "__main__":
    trainer = TGPOTrainer(scenario_name="local_minima_trap")
    # You can specify file names saved at different stages by log_filename
    trainer.train(num_iterations=400, log_filename="trap_v2_s_2.npz")