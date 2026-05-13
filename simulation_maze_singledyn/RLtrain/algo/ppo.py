import jax
import jax.numpy as jnp
import optax
from functools import partial
from config import TGPOConfig

# ==========================================
# 1. Advantage Function Calculation (GAE)
# ==========================================
@partial(jax.jit, static_argnames=['gamma', 'gae_lambda'])
def calculate_gae(
    rewards, 
    values, 
    next_values, 
    dones, 
    gamma=TGPOConfig.GAMMA,          
    gae_lambda=TGPOConfig.GAE_LAMBDA 
):
  
  
    values = jnp.squeeze(values)
    next_values = jnp.squeeze(next_values)
    
    
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    
   
    def scan_fn(carry, inputs):
        delta, done = inputs
        advantage_next = carry
        advantage = delta + gamma * gae_lambda * (1.0 - done) * advantage_next
        return advantage, advantage 
    
    
    deltas_T = deltas.T
    dones_T = dones.T
    
    init_advantage = jnp.zeros(deltas.shape[0])
    
    _, advantages_T = jax.lax.scan(
        scan_fn,
        init_advantage,
        (deltas_T, dones_T),
        reverse=True 
    )
    
    advantages = advantages_T.T
    targets = advantages + values
    
    return advantages, targets

# ==========================================
# 2. Probability distribution calculation
# ==========================================
def compute_log_prob(mean, log_std, action):
  
    log_std = jnp.clip(log_std, -20, 2)
    
    std = jnp.exp(log_std)
    var = jnp.square(std)
    log_prob = -0.5 * ((action - mean) ** 2 / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(log_prob, axis=-1)

# ==========================================
# 3.  (Loss Functions)
# ==========================================
def actor_loss_fn(actor_params, actor_apply_fn, batch, advantage):
   
    mean, log_std = actor_apply_fn(actor_params, batch['aug_state'], batch['time_vars'])
    
    
    log_prob = compute_log_prob(mean, log_std, batch['action'])
    old_log_prob = batch['log_prob']
    
    # Ratio
    ratio = jnp.exp(log_prob - old_log_prob)
    
    # PPO Clip
    clip_eps = TGPOConfig.CLIP_EPS
    surr1 = ratio * advantage
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss_actor = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Entropy 
    log_std_clamped = jnp.clip(log_std, -20, 2)
    entropy = jnp.sum(log_std_clamped + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
    loss_ent = -jnp.mean(entropy)
    
    ent_coef = TGPOConfig.ENT_COEF
    total_loss = loss_actor + ent_coef * loss_ent
    
    return total_loss, (loss_actor, loss_ent, jnp.mean(entropy))

def critic_loss_fn(critic_params, critic_apply_fn, batch, target):
    values = critic_apply_fn(critic_params, batch['aug_state'], batch['time_vars'])
    
   
    values = jnp.squeeze(values)
    target = jnp.squeeze(target)
    
    loss_critic = jnp.mean(jnp.square(values - target))
    return loss_critic

# ==========================================
# 4. (Update Step)
# ==========================================
@partial(jax.jit, static_argnames=['actor_apply_fn', 'critic_apply_fn'])
def update_step(
    actor_state,
    critic_state,
    batch,
    advantages,
    targets,
    actor_apply_fn,
    critic_apply_fn
):
    # Advantage Normalization
    
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    # --- Update Actor ---
    grad_actor_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (a_loss, (loss_clip, loss_ent, mean_ent)), a_grads = grad_actor_fn(
        actor_state.params, 
        actor_apply_fn, 
        batch, 
        advantages
    )
    a_updates, new_actor_opt_state = actor_state.tx.update(a_grads, actor_state.opt_state, actor_state.params)
    new_actor_params = optax.apply_updates(actor_state.params, a_updates)
    
    # --- Update Critic ---
    grad_critic_fn = jax.value_and_grad(critic_loss_fn)
    c_loss, c_grads = grad_critic_fn(
        critic_state.params,
        critic_apply_fn,
        batch,
        targets
    )
    c_updates, new_critic_opt_state = critic_state.tx.update(c_grads, critic_state.opt_state, critic_state.params)
    new_critic_params = optax.apply_updates(critic_state.params, c_updates)
    
    # Update States
    new_actor_state = actor_state.replace(params=new_actor_params, opt_state=new_actor_opt_state)
    new_critic_state = critic_state.replace(params=new_critic_params, opt_state=new_critic_opt_state)
    
    info = {
        "loss_actor": a_loss,
        "loss_critic": c_loss,
        "entropy": mean_ent
    }
    
    return new_actor_state, new_critic_state, info