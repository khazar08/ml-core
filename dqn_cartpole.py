import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque
import random
from tqdm import tqdm
import seaborn as sns
from scipy import stats

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

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

class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_history = []
        self.moving_avg_rewards = []
        
        self.q_values_history = []
    
    def select_action(self, state, evaluate=False):
        if evaluate:
            # Greedy action selection for evaluation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if len(self.memory) % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train_episode(self, env, max_steps=500, render=False):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        q_values_episode = []
        
        for step in range(max_steps):
            action = self.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.store_transition(state, action, reward, next_state, done)
            
          
            loss = self.learn()
            if loss > 0:
                episode_loss += loss
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor).cpu().numpy()
                q_values_episode.append(q_values[0])
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.losses.append(episode_loss / steps if steps > 0 else 0)
        self.epsilon_history.append(self.epsilon)
        self.q_values_history.append(np.mean(q_values_episode, axis=0))
        
        return episode_reward, steps
    
    def evaluate(self, env, num_episodes=10, render=False):
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if render:
                    env.render()  
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)

class DQNVisualizer:
    
    @staticmethod
    def plot_training_metrics(agent, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(agent.episode_rewards) > 10:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(agent.episode_rewards)), moving_avg, 
                          'r-', linewidth=2, label='Moving Avg (10)')
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
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(agent.epsilon_history)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        if agent.q_values_history:
            q_values = np.array(agent.q_values_history)
            axes[1, 1].plot(q_values[:, 0], label='Q-value Action 0', alpha=0.7)
            axes[1, 1].plot(q_values[:, 1], label='Q-value Action 1', alpha=0.7)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Q-value')
            axes[1, 1].set_title('Average Q-values')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # reward distribution
        if len(agent.episode_rewards) > 0:
            axes[1, 2].hist(agent.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 2].axvline(np.mean(agent.episode_rewards), color='r', linestyle='--', 
                              label=f'Mean: {np.mean(agent.episode_rewards):.2f}')
            axes[1, 2].set_xlabel('Reward')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Reward Distribution')
            axes[1, 2].legend()
        
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
            rewards = []
            
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                
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
            axes[episode].set_title(f'Episode {episode+1}, Reward: {sum(rewards):.0f}')
            axes[episode].legend()
            axes[episode].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_q_value_heatmap(agent, state_dim=4, action_dim=2):
        # Create grid of states
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Test different state combinations
        x_range = np.linspace(-2.4, 2.4, 20)
        y_range = np.linspace(-3, 3, 20)
        
        for i, (dim1, dim2) in enumerate([(0, 1), (0, 2), (1, 3), (2, 3)]):
            q_values_0 = np.zeros((len(x_range), len(y_range)))
            q_values_1 = np.zeros((len(x_range), len(y_range)))
            
            for xi, x_val in enumerate(x_range):
                for yi, y_val in enumerate(y_range):
                    # Create state
                    state = np.zeros(state_dim)
                    state[dim1] = x_val
                    state[dim2] = y_val
                    
                    # Get Q-values
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_vals = agent.policy_net(state_tensor).cpu().numpy()[0]
                    
                    q_values_0[xi, yi] = q_vals[0]
                    q_values_1[xi, yi] = q_vals[1]
            
            ax = axes[i // 2, i % 2]
            diff = q_values_1 - q_values_0
            im = ax.imshow(diff, extent=[y_range.min(), y_range.max(), x_range.min(), x_range.max()],
                          origin='lower', cmap='RdBu', aspect='auto')
            ax.set_xlabel(f'Dimension {dim2}')
            ax.set_ylabel(f'Dimension {dim1}')
            ax.set_title(f'Q-value diff (Action 1 - Action 0)\nDims {dim1} and {dim2}')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()

def main():
    
    
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    print(f"Agent created with {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    
    num_episodes = 500
    print(f"\nTraining for {num_episodes} episodes...")
    
    episode_rewards = []
    
    pbar = tqdm(range(num_episodes))
    for episode in pbar:
        reward, length = agent.train_episode(env)
        episode_rewards.append(reward)
        
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        pbar.set_description(f"Ep {episode+1},  Reward: {reward:.1f},  Avg50: {avg_reward:.1f}, Eps: {agent.epsilon:.3f}")
        
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"\nEnvironment solved in {episode+1} episodes!")
            break
    
    env.close()
    
    eval_env = gym.make('CartPole-v1', render_mode='human')
    mean_reward, std_reward = agent.evaluate(eval_env, num_episodes=10, render=True)
    eval_env.close()
    
    print(f"Evaluation Results (10 episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    if mean_reward >= 475:
        print("Environment solved (Mean reward >= 475)")
    else:
        print("Environment not solved yet. Keep training")
    
    
    visualizer = DQNVisualizer()
    visualizer.plot_training_metrics(agent)
    
    viz_env = gym.make('CartPole-v1')
    visualizer.visualize_policy(agent, viz_env, num_episodes=4)
    viz_env.close()
    
    visualizer.plot_q_value_heatmap(agent, state_dim, action_dim)
    
    final_100_rewards = agent.episode_rewards[-100:] if len(agent.episode_rewards) >= 100 else agent.episode_rewards
    
    print(f"Total episodes trained: {len(agent.episode_rewards)}")
    print(f"Final 100 episode average reward: {np.mean(final_100_rewards):.2f} ± {np.std(final_100_rewards):.2f}")
    print(f"Max reward achieved: {np.max(agent.episode_rewards):.0f}")
    print(f"Min reward achieved: {np.min(agent.episode_rewards):.0f}")
    
    # convergence analysis
    if len(agent.episode_rewards) > 100:
        # Check if rewards are converging
        first_100 = np.mean(agent.episode_rewards[:100])
        last_100 = np.mean(agent.episode_rewards[-100:])
        improvement = (last_100 - first_100) / first_100 * 100
        
        print(f"First 100 episodes avg: {first_100:.2f}")
        print(f"Last 100 episodes avg: {last_100:.2f}")
        print(f"Improvement: {improvement:.1f}%")
    
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode_rewards': agent.episode_rewards,
        'epsilon_history': agent.epsilon_history,
    }, 'dqn_cartpole_model.pth')
    
    return agent

if __name__ == "__main__":
    agent = main()
    
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    test_agent = DQNAgent()
    checkpoint = torch.load('dqn_cartpole_model.pth')
    test_agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    
    mean_reward, std_reward = test_agent.evaluate(test_env, num_episodes=5, render=True)
    print(f"Test Results: {mean_reward:.2f} ± {std_reward:.2f}")
    test_env.close()
