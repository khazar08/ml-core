import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import seaborn as sns
from collections import deque
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return torch.softmax(self.network(state), dim=-1)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, epochs=10, batch_size=64, entropy_coef=0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
        
        self.actor = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.policy_ratios = []
        self.kl_divs = []
    
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor(state_tensor)
        
        if evaluate:
            return probs.argmax().item()
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self):
        advantages = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                gae = 0
            
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def finish_episode(self):
        advantages, returns = self.compute_gae()
        
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                
                self.policy_ratios.append(ratio.mean().item())
                
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                    self.kl_divs.append(kl)
        
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(self.rewards))
        self.actor_losses.append(total_actor_loss / (len(self.states) * self.epochs))
        self.critic_losses.append(total_critic_loss / (len(self.states) * self.epochs))
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        return episode_reward
    
    def train_episode(self, env, max_steps=500, render=False):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_reward(reward, done)
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
            
            if done:
                break
        
        episode_reward = self.finish_episode()
        return episode_reward
    
    def evaluate(self, env, num_episodes=10, render=False):
        episode_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                
                if render:
                    env.render()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)

class PPOVisualizer:
    @staticmethod
    def plot_training_metrics(agent):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(agent.episode_rewards, alpha=0.6)
        if len(agent.episode_rewards) > 10:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=2)
        axes[0, 0].axhline(y=475, color='g', linestyle='--')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(agent.episode_lengths)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(agent.actor_losses, label='Actor')
        axes[0, 2].plot(agent.critic_losses, label='Critic')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Losses')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        if len(agent.policy_ratios) > 0:
            axes[1, 0].plot(agent.policy_ratios)
            axes[1, 0].axhline(y=1.0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Policy Ratio')
            axes[1, 0].set_title('Policy Ratio (π_new/π_old)')
            axes[1, 0].grid(True, alpha=0.3)
        
        if len(agent.kl_divs) > 0:
            axes[1, 1].plot(agent.kl_divs)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('KL Divergence')
            axes[1, 1].set_title('KL Divergence')
            axes[1, 1].grid(True, alpha=0.3)
        
        if len(agent.episode_rewards) > 100:
            first_100 = np.mean(agent.episode_rewards[:100])
            last_100 = np.mean(agent.episode_rewards[-100:])
            axes[1, 2].bar(['First 100', 'Last 100'], [first_100, last_100], color=['blue', 'green'])
            axes[1, 2].set_ylabel('Average Reward')
            axes[1, 2].set_title('Learning Progress')
            axes[1, 2].text(0, first_100 + 10, f'{first_100:.1f}', ha='center')
            axes[1, 2].text(1, last_100 + 10, f'{last_100:.1f}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_policy(agent, env, num_episodes=5):
        fig, axes = plt.subplots(1, num_episodes, figsize=(15, 3))
        if num_episodes == 1:
            axes = [axes]
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            states = [state]
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states.append(next_state)
                
                if done:
                    break
                state = next_state
            
            states = np.array(states)
            axes[episode].plot(states[:, 0], label='Position')
            axes[episode].plot(states[:, 1], label='Velocity')
            axes[episode].plot(states[:, 2], label='Angle')
            axes[episode].plot(states[:, 3], label='Angular Vel')
            axes[episode].set_xlabel('Step')
            axes[episode].set_title(f'Episode {episode+1}')
            axes[episode].legend()
            axes[episode].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    num_episodes = 300
    print(f"Training PPO agent for {num_episodes} episodes...")
    
    episode_rewards = []
    pbar = tqdm(range(num_episodes))
    
    for episode in pbar:
        reward = agent.train_episode(env)
        episode_rewards.append(reward)
        
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        pbar.set_description(f"Ep {episode+1} | Reward: {reward:.1f} | Avg50: {avg_reward:.1f}")
        
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"\nEnvironment solved in {episode+1} episodes!")
            break
    
    env.close()
    
    eval_env = gym.make('CartPole-v1', render_mode='human')
    mean_reward, std_reward = agent.evaluate(eval_env, num_episodes=10, render=True)
    eval_env.close()
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    if mean_reward >= 475:
        print("✓ Environment SOLVED!")
    else:
        print("✗ Environment NOT solved yet.")
    
    visualizer = PPOVisualizer()
    visualizer.plot_training_metrics(agent)
    
    viz_env = gym.make('CartPole-v1')
    visualizer.visualize_policy(agent, viz_env, num_episodes=4)
    viz_env.close()
    
    final_100 = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
    print(f"\nFinal 100 episode average: {np.mean(final_100):.2f} ± {np.std(final_100):.2f}")
    
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'rewards': agent.episode_rewards,
    }, 'ppo_cartpole.pth')
    print("\nModel saved as 'ppo_cartpole.pth'")

if __name__ == "__main__":
    main()
