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

# 导入配置和模块
from config import TGPOConfig
from envs.scenarios import get_scenario
from models.networks import Actor, Critic
from planning.rollout import rollout_episode
from planning.uniform import sample_uniform_times
from planning.mcmc import metropolis_hastings_sampling
from planning.elite import EliteBuffer
from utils.stljax_evaluator import STLEvaluator
from algo.ppo import calculate_gae, compute_log_prob, update_step


def plot_dynamic_overtaking(phys_trajs, scores, scenario, iter_idx, filename="dynamic_result.png"):
    
    best_idx = jnp.argmax(scores)
    best_score = scores[best_idx]
    traj = np.array(phys_trajs[best_idx])  # (T, 4) [x, y, theta, v]
    
    T = traj.shape[0]
    dt = TGPOConfig.DT
    
    
    plt.figure(figsize=(18, 6))
    ax = plt.gca()
    
   
    plt.axhline(y=0.0, color='black', linewidth=2)
    plt.axhline(y=4.0, color='black', linewidth=2)
    plt.axhline(y=2.0, color='white', linestyle='--', linewidth=2) # 车道中线
    
   
    plt.fill_between([-5, 70], 0, 4, color='gray', alpha=0.2, label='Road')

    
    obs_params = scenario.dynamic_obs_params 
    colors = ['red', 'orange', 'brown', 'purple']
    
    for i in range(len(obs_params)):
        ox, oy, vx, r = obs_params[i]
        
       
        end_x = ox + vx * (T * dt)
        plt.plot([ox, end_x], [oy, oy], color=colors[i], linestyle=':', linewidth=1.5, alpha=0.6)
        plt.text(ox, oy - 0.4, f"Obs{i}", color=colors[i], fontsize=9, fontweight='bold')
        
     
        snapshot_steps = [0, 25, 50, 75, 100]
        for step in snapshot_steps:
            curr_x = ox + vx * (step * dt)
         
            if step < T:
                circle = plt.Circle((curr_x, oy), r, color=colors[i], alpha=0.3)
                ax.add_patch(circle)

    
    sc = plt.scatter(traj[:, 0], traj[:, 1], c=np.arange(T), cmap='viridis', s=15, zorder=10, label='Agent Path')
    plt.colorbar(sc, label='Time Step (0-100)')
    
  
    goal_times = [25, 50, 75, 100]
    
    ref_obs_indices = [0, 1, 2, 3] 
    goal_ys = [3.0, 1.0, 3.0, 1.0] 
    
    for k, t_g in enumerate(goal_times):
       
        ref_idx = ref_obs_indices[k]
        ox_start, _, vx, _ = obs_params[ref_idx]
        
     
        obs_x_at_deadline = ox_start + vx * (t_g * dt)
        
      
        target_x = obs_x_at_deadline + 2.0
        target_y = goal_ys[k]
        
       
        goal_circle = plt.Circle((target_x, target_y), 0.5, 
                               edgecolor='blue', facecolor='none', linewidth=2, linestyle='--')
        ax.add_patch(goal_circle)
        
      
        if t_g < T:
            agent_pos = traj[t_g]
            plt.plot(agent_pos[0], agent_pos[1], 'bx', markersize=10, markeredgewidth=2)
            plt.text(target_x, target_y + 0.6, f"G{k}\nT={t_g}", color='blue', ha='center', fontsize=8, fontweight='bold')

    plt.title(f"Dynamic Overtaking (Iter {iter_idx}) | Best Score: {best_score:.3f}", fontsize=14)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    

    max_x = max(60, traj[-1, 0] + 5)
    plt.xlim(-2, max_x)
    plt.ylim(-1, 5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close()
    print(f"[Visual] The dynamic overtaking trajectory map has been saved to: {filename}")


class TGPOTrainer:
    def __init__(self, seed=0, scenario_name="dynamic_overtaking"): 
        self.rng = jax.random.PRNGKey(seed)
        
        
        self.scenario = get_scenario(scenario_name)
        print(f"[Init] Scenario: {self.scenario.name}")
        print(f"[Init] Map Limit: {TGPOConfig.X_LIMIT}")
        
      
        dummy_aug = jnp.zeros((1, TGPOConfig.AUG_STATE_DIM))
        dummy_time = jnp.zeros((1, self.scenario.subgoals.shape[0])) 
        
        self.actor = Actor(action_dim=2)
        self.critic = Critic()
        
        self.rng, k1, k2 = jax.random.split(self.rng, 3)
        actor_params = self.actor.init(k1, dummy_aug, dummy_time)
        critic_params = self.critic.init(k2, dummy_aug, dummy_time)
        
     
        optimizer = optax.chain(
            optax.clip_by_global_norm(TGPOConfig.GRAD_NORM_CLIP),
            optax.adamw(learning_rate=TGPOConfig.LR, weight_decay=TGPOConfig.WEIGHT_DECAY)
        )
        
        self.actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=optimizer)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=critic_params, tx=optimizer)
            
     
        self.elite_buffer = EliteBuffer(n_vars=self.scenario.subgoals.shape[0])
        self.elite_times = self.elite_buffer.times
        self.elite_scores = self.elite_buffer.scores
        
        
        if hasattr(self.scenario, 'dynamic_obs_params') and self.scenario.dynamic_obs_params is not None:
            self.stl_evaluator = STLEvaluator(self.scenario.dynamic_obs_params, self.scenario.subgoals)
            print("[Init] Using Dynamic STL Evaluator")
        else:
            
            self.stl_evaluator = STLEvaluator(self.scenario.obstacles, self.scenario.subgoals)
            print("[Init] Using Static STL Evaluator")
        
   
        self.log_history = {
            "iter": [], "reward": [], "success": [], "entropy": [], "critic_loss": [], "actor_loss": [] 
        }
        
        print("[Init] Ready.")

    def save_checkpoint(self, filename="dover_model_slow.msgpack"):
      
        ckpt = {'actor': self.actor_state, 'critic': self.critic_state}
        with open(filename, 'wb') as f:
            f.write(serialization.to_bytes(ckpt))
        print(f"[Checkpoint] ✅ Model saved to: {filename}")

    def load_checkpoint(self, filename="dover_model_slow.msgpack"):
  
        if not os.path.exists(filename):
            print(f"[Checkpoint] ⚠️ If no checkpoint {filename} is found, a new training will begin。")
            return
        
        print(f"[Checkpoint] 🔄 Loading checkpoint: {filename} ...")
        with open(filename, 'rb') as f:
            bytes_data = f.read()
        
        target = {'actor': self.actor_state, 'critic': self.critic_state}
        restored = serialization.from_bytes(target, bytes_data)
        
        self.actor_state = restored['actor']
        self.critic_state = restored['critic']
        print(f"[Checkpoint] ✅ Model loading successfully! Keep training...")

    def plot_training_curves(self, filename="training_curves.png"):
        iters = self.log_history["iter"]
        if len(iters) == 0: return

        plt.figure(figsize=(15, 10))
        plt.suptitle('Training Metrics', fontsize=16)
        
        plt.subplot(2, 2, 1); plt.plot(iters, self.log_history["reward"], 'b-'); plt.title("Avg Reward"); plt.grid(True)
        plt.subplot(2, 2, 2); plt.plot(iters, self.log_history["success"], 'g-'); plt.title("Success Rate (%)"); plt.grid(True)
        plt.subplot(2, 2, 3); plt.plot(iters, self.log_history["actor_loss"], 'r-'); plt.title("Actor Loss"); plt.grid(True)
        plt.subplot(2, 2, 4); plt.plot(iters, self.log_history["critic_loss"], 'k-'); plt.title("Critic Loss"); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[Visual] The training graph is saved to: {filename}")

    def train(self, num_iterations): 
       
        self.load_checkpoint()
        
        print(f"Start Training for {num_iterations} iterations...")
        
       
        def reset_states(key, batch_size):
            k1, k2, k3 = jax.random.split(key, 3)
            
           
            pos_x = jax.random.uniform(k1, shape=(batch_size, 1), minval=0.0, maxval=1.0)
            pos_y = jax.random.uniform(k2, shape=(batch_size, 1), minval=0.6, maxval=1.4)
            pos = jnp.concatenate([pos_x, pos_y], axis=-1)

            
            theta = jax.random.uniform(k3, shape=(batch_size, 1), minval=0.0, maxval=jnp.pi/4.0)
            
           
            v = jnp.ones((batch_size, 1)) * 1.0

            phys = jnp.concatenate([pos, theta, v], axis=-1)
        
            logic = jnp.tile(jnp.array([0., 0., 0., 0., 1.]), (batch_size, 1))
            return jnp.concatenate([phys, logic], axis=-1)

        last_phys_trajs = None
        last_robustness_scores = None

      
        for it in range(num_iterations):
            self.rng, key_iter = jax.random.split(self.rng)
            
         
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
            
            
            init_states = reset_states(key_rollout, n_batch)
            
           
            phys_trajs, transitions = rollout_episode(
                key_rollout, self.actor.apply, self.actor_state.params, init_states, time_batch, self.scenario
            )
            last_phys_trajs = phys_trajs
            
            
            robustness_scores = self.stl_evaluator.evaluate(phys_trajs)
            last_robustness_scores = robustness_scores 
            
            self.elite_times, self.elite_scores = self.elite_buffer.update(
                self.elite_times, self.elite_scores, time_batch, robustness_scores
            )
            
         
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
            
           
            for _ in range(TGPOConfig.PPO_EPOCHS):
                self.actor_state, self.critic_state, info = update_step(
                    self.actor_state, self.critic_state, batch_data, flat_adv, flat_target,
                    self.actor.apply, self.critic.apply
                )
            
          
            if it % 10 == 0:
                best_idx = jnp.argmax(robustness_scores)
                best_t = time_batch[best_idx].tolist()
           
                t_str = "[" + ", ".join([f"{t:.1f}" for t in best_t]) + "]"
                
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

                print(f"It {it:4d} | R:{mean_raw_reward:5.1f} | S:{succ_rate:3.0f}% | "
                      f"Rob:{robustness_scores[best_idx]:5.2f} | "
                      f"Ent:{entropy:5.2f} | "
                      f"L_A:{loss_a:5.4f} | L_C:{loss_c:5.4f} | T_best:{t_str}")

        print("\n=== Training Finished ===")
        
        self.save_checkpoint()
        self.plot_training_curves()
        
      
        plot_dynamic_overtaking(
            last_phys_trajs, 
            last_robustness_scores, 
            self.scenario, 
            num_iterations, 
            filename="dynamic_final_result.png"
        )


if __name__ == "__main__":
   
    trainer = TGPOTrainer(scenario_name="dynamic_overtaking")
   
    trainer.train(num_iterations=100)