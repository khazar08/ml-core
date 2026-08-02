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
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.002, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.memory = ReplayBuffer()
        self.batch_size = 64
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        
        self.noise_scale = 0.1
    
    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * ~dones
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.q_values.append(current_q.mean().item())
        
        return actor_loss.item(), critic_loss.item()
    
    def train_episode(self, env, max_steps=500, render=False):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss_a = 0
        episode_loss_c = 0
        steps = 0
        
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_transition(state, action, reward, next_state, done)
            
            loss_a, loss_c = self.learn()
            episode_loss_a += loss_a
            episode_loss_c += loss_c
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.actor_losses.append(episode_loss_a / steps if steps > 0 else 0)
        self.critic_losses.append(episode_loss_c / steps if steps > 0 else 0)
        
        return episode_reward, steps
    
    def evaluate(self, env, num_episodes=10, render=False):
        episode_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action = self.select_action(state, add_noise=False)
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

class DDPGVisualizer:
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
        
        axes[1, 0].plot(agent.q_values)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Q-value')
        axes[1, 0].set_title('Average Q-Values')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(agent.episode_rewards) > 0:
            axes[1, 1].hist(agent.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
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
    def visualize_policy(agent, env, num_episodes=5):
        fig, axes = plt.subplots(1, num_episodes, figsize=(15, 3))
        if num_episodes == 1:
            axes = [axes]
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            states = [state]
            
            while True:
                action = agent.select_action(state, add_noise=False)
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
    
    agent = DDPGAgent(state_dim, action_dim)
    
    num_episodes = 500
    print(f"Training DDPG agent for {num_episodes} episodes...")
    
    episode_rewards = []
    pbar = tqdm(range(num_episodes))
    
    for episode in pbar:
        reward, length = agent.train_episode(env)
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
    
    visualizer = DDPGVisualizer()
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
    }, 'ddpg_cartpole.pth')
    print("\nModel saved as 'ddpg_cartpole.pth'")

if __name__ == "__main__":
    main()
