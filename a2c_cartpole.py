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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
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

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
        
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.values = []
    
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        
        if evaluate:
            return probs.argmax().item(), value.item()
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.values.append(value.item())
        
        return action.item(), value.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def finish_episode(self):
        returns = []
        advantages = []
        R = 0
        
        # Calculate returns and advantages
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(device)
        
        # Calculate advantages
        for i in range(len(self.rewards)):
            advantage = returns[i] - self.values[i]
            advantages.append(advantage)
        
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        actor_loss = 0
        critic_loss = 0
        entropy = 0
        
        for state, action, adv, ret in zip(self.states, self.actions, advantages, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = self.model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action).to(device))
            
            # Actor loss
            actor_loss += -log_prob * adv
            
            # Critic loss
            critic_loss += nn.MSELoss()(value, ret.unsqueeze(0))
            
            # Entropy bonus
            entropy += dist.entropy()
        
        actor_loss = actor_loss / len(self.states)
        critic_loss = critic_loss / len(self.states)
        entropy = entropy / len(self.states)
        
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(self.rewards))
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropies.append(entropy.item())
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return episode_reward
    
    def train_episode(self, env, max_steps=500, render=False):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action, value = self.select_action(state)
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
                action, _ = self.select_action(state, evaluate=True)
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

class A2CVisualizer:
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
        
        axes[1, 0].plot(agent.entropies)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Policy Entropy')
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
                action, _ = agent.select_action(state, evaluate=True)
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
    
    agent = A2CAgent(state_dim, action_dim)
    
    num_episodes = 500
    print(f"Training A2C agent for {num_episodes} episodes...")
    
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
    
    visualizer = A2CVisualizer()
    visualizer.plot_training_metrics(agent)
    
    viz_env = gym.make('CartPole-v1')
    visualizer.visualize_policy(agent, viz_env, num_episodes=4)
    viz_env.close()
    
    final_100 = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
    print(f"\nFinal 100 episode average: {np.mean(final_100):.2f} ± {np.std(final_100):.2f}")
    
    torch.save({
        'model': agent.model.state_dict(),
        'rewards': agent.episode_rewards,
    }, 'a2c_cartpole.pth')
    print("\nModel saved as 'a2c_cartpole.pth'")

if __name__ == "__main__":
    main()
