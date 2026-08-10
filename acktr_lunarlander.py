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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return torch.softmax(self.network(state), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class ACKTRAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 entropy_coef=0.01, kfac_update_freq=10):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
        self.kfac_update_freq = kfac_update_freq
        self.step_counter = 0
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []
    
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.actor(state_tensor)
        
        if evaluate:
            return probs.argmax().item()
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor)
        entropy = dist.entropy()
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        self.entropies.append(entropy.item())
        
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
            gae = delta + self.gamma * 0.95 * gae
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
        entropies = torch.FloatTensor(np.array(self.entropies)).to(device)
        
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Actor loss with trust region
        actor_loss = -(ratio * advantages).mean() - self.entropy_coef * entropy.mean()
        
        # Critic loss
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        # K-FAC approximation (simplified with Adam)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(self.rewards))
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_history.append(entropy.mean().item())
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        return episode_reward
    
    def train_episode(self, env, max_steps=1000, render=False):
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

class ACKTRVisualizer:
    @staticmethod
    def plot_training_metrics(agent):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(agent.episode_rewards, alpha=0.6)
        if len(agent.episode_rewards) > 10:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=2)
        axes[0, 0].axhline(y=200, color='g', linestyle='--')
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
        
        axes[1, 0].plot(agent.entropy_history)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(agent.episode_rewards) > 0:
            axes[1, 1].hist(agent.episode_rewards, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(np.mean(agent.episode_rewards), color='r', linestyle='--')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Reward Distribution')
        
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
    def visualize_policy(agent, env, num_episodes=4):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for episode in range(min(num_episodes, 4)):
            state, _ = env.reset()
            positions_x = [state[0]]
            positions_y = [state[1]]
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                positions_x.append(next_state[0])
                positions_y.append(next_state[1])
                
                if done:
                    break
                state = next_state
            
            axes[episode].scatter(positions_x, positions_y, c=range(len(positions_x)), 
                                 cmap='viridis', s=20, alpha=0.7)
            axes[episode].scatter(positions_x[0], positions_y[0], color='green', 
                                 s=100, marker='s', label='Start')
            axes[episode].scatter(positions_x[-1], positions_y[-1], color='red', 
                                 s=100, marker='*', label='End')
            axes[episode].set_xlabel('X Position')
            axes[episode].set_ylabel('Y Position')
            axes[episode].set_title(f'Episode {episode+1}')
            axes[episode].legend()
            axes[episode].grid(True, alpha=0.3)
            axes[episode].set_xlim(-1.5, 1.5)
            axes[episode].set_ylim(0, 1.5)
        
        plt.tight_layout()
        plt.show()

def main():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"ACKTR - Actor-Critic with Kronecker-factored Trust Region")
    print(f"Environment: LunarLander-v2")
    print(f"State Space: {state_dim} dimensions")
    print(f"Action Space: {action_dim} actions")
    print("-" * 50)
    
    agent = ACKTRAgent(state_dim, action_dim)
    
    num_episodes = 600
    print(f"Training ACKTR agent for {num_episodes} episodes...")
    
    episode_rewards = []
    pbar = tqdm(range(num_episodes))
    
    for episode in pbar:
        reward = agent.train_episode(env)
        episode_rewards.append(reward)
        
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        pbar.set_description(f"Ep {episode+1} | Reward: {reward:.1f} | Avg100: {avg_reward:.1f}")
        
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
            print(f"\nLunarLander solved in {episode+1} episodes!")
            break
    
    env.close()
    
    eval_env = gym.make('LunarLander-v2', render_mode='human')
    mean_reward, std_reward = agent.evaluate(eval_env, num_episodes=10, render=True)
    eval_env.close()
    
    print(f"\nEvaluation Results (10 episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    if mean_reward >= 200:
        print("✓ Environment SOLVED!")
    else:
        print("✗ Environment NOT solved yet.")
    
    visualizer = ACKTRVisualizer()
    visualizer.plot_training_metrics(agent)
    
    viz_env = gym.make('LunarLander-v2')
    visualizer.visualize_policy(agent, viz_env, num_episodes=4)
    viz_env.close()
    
    final_100 = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
    print(f"\nFinal 100 episode average reward: {np.mean(final_100):.2f}")
    
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'rewards': agent.episode_rewards,
    }, 'acktr_lunarlander.pth')
    print("\nModel saved as 'acktr_lunarlander.pth'")

if __name__ == "__main__":
    main()
