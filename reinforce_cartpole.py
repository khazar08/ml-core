import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import seaborn as sns
from collections import deque
from scipy import stats

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=0.001, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.entropy_history = []
        self.policy_entropy = []
        
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.policy_net(state_tensor)
        
        if evaluate:
            return action_probs.argmax().item()
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob)
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def finish_episode(self):
        discounted_rewards = []
        running_reward = 0
        
        for reward in reversed(self.rewards):
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        policy_loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss -= log_prob * reward
        
        # add entropy bonus for exploration
        entropy = 0
        for state in self.states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_net(state_tensor)
            entropy += -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        
        total_loss = policy_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(self.rewards))
        self.losses.append(total_loss.item())
        self.entropy_history.append(entropy.item() / len(self.states))
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        return episode_reward
    
    def train_episode(self, env, max_steps=500, render=False):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_reward(reward)
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

class REINFORCEVisualizer:
    @staticmethod
    def plot_training_metrics(agent, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(agent.episode_rewards) > 10:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(agent.episode_rewards)), moving_avg, 
                          'r-', linewidth=2, label='Moving Avg (10)')
        axes[0, 0].axhline(y=475, color='g', linestyle='--', label='Solved Threshold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(agent.episode_lengths)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(agent.losses)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Policy Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(agent.entropy_history)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Policy Entropy (Exploration)')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(agent.episode_rewards) > 0:
            axes[1, 1].hist(agent.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(np.mean(agent.episode_rewards), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(agent.episode_rewards):.2f}')
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].legend()
        
        if len(agent.episode_lengths) > 0:
            axes[1, 2].plot(agent.episode_lengths, label='Steps per episode', alpha=0.6)
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Steps')
            axes[1, 2].set_title('Steps per Episode')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_policy(agent, env, num_episodes=5):
        fig, axes = plt.subplots(1, num_episodes, figsize=(15, 3))
        if num_episodes == 1:
            axes = [axes]
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            states = [state]
            actions = []
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states.append(next_state)
                actions.append(action)
                
                if done:
                    break
                state = next_state
            
            states = np.array(states)
            axes[episode].plot(states[:, 0], label='Cart Position')
            axes[episode].plot(states[:, 1], label='Cart Velocity')
            axes[episode].plot(states[:, 2], label='Pole Angle')
            axes[episode].plot(states[:, 3], label='Pole Angular Velocity')
            axes[episode].set_xlabel('Step')
            axes[episode].set_ylabel('State Value')
            axes[episode].set_title(f'Episode {episode+1}')
            axes[episode].legend()
            axes[episode].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_action_probabilities(agent, env, num_samples=100):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        states = []
        for _ in range(num_samples):
            state, _ = env.reset()
            states.append(state)
        
        states = np.array(states)
        
        action_probs = []
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = agent.policy_net(state_tensor).cpu().numpy()[0]
            action_probs.append(probs)
        
        action_probs = np.array(action_probs)   
        state_dims = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
        
        for i, ax in enumerate(axes.flat):
            if i < len(state_dims):
                scatter = ax.scatter(states[:, i], action_probs[:, 0], 
                                   c=states[:, 0], cmap='viridis', alpha=0.6)
                ax.set_xlabel(state_dims[i])
                ax.set_ylabel('P(action=0)')
                ax.set_title(f'Action Probability vs {state_dims[i]}')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.show()

def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim)
    
    num_episodes = 1000
    print(f"Training REINFORCE agent for {num_episodes} episodes...")
    
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
        print("Environment solved")
    else:
        print("Environment not solved yet")
    
    visualizer = REINFORCEVisualizer()
    visualizer.plot_training_metrics(agent)
    
    viz_env = gym.make('CartPole-v1')
    visualizer.visualize_policy(agent, viz_env, num_episodes=4)
    viz_env.close()
    
    visualizer.visualize_action_probabilities(agent, env)
    
    final_100_rewards = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
    print(f"\nFinal 100 episode average reward: {np.mean(final_100_rewards):.2f} ± {np.std(final_100_rewards):.2f}")
    
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode_rewards': agent.episode_rewards,
    }, 'reinforce_cartpole.pth')
    print("\nModel saved as 'reinforce_cartpole.pth'")

if __name__ == "__main__":
    main()
