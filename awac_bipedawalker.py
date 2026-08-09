import argparse
import random
from collections import deque
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class Config:
    seed: int = 123
    episodes: int = 250
    replay_capacity: int = 200_000
    warmup_steps: int = 5_000
    batch_size: int = 256
    updates_per_episode: int = 80

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    awac_temperature: float = 1.0
    max_weight: float = 20.0

    hidden: int = 256
    eval_episodes: int = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)



class GaussianActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.body = mlp(state_dim, hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

        nn.init.uniform_(self.log_std.bias, -1.0, -0.5)

    def forward(self, state):
        h = self.body(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 1.0)
        return mean, log_std

    def distribution(self, state):
        mean, log_std = self(state)
        return Normal(mean, log_std.exp())

    def sample(self, state, deterministic=False):
        dist = self.distribution(state)

        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()

        action = torch.tanh(z)

        # Log-probability correction for tanh squashing.
        log_prob = dist.log_prob(z) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, hidden, 1)
        self.q2 = mlp(state_dim + action_dim, hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden):
        super().__init__()
        self.v = mlp(state_dim, hidden, 1)

    def forward(self, state):
        return self.v(state)



class AWAC:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.device = device
        self.cfg = cfg

        self.actor = GaussianActor(
            state_dim, action_dim, cfg.hidden
        ).to(device)

        self.critic = QNetwork(
            state_dim, action_dim, cfg.hidden
        ).to(device)

        self.target_critic = QNetwork(
            state_dim, action_dim, cfg.hidden
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.value = ValueNetwork(state_dim, cfg.hidden).to(device)
        self.target_value = ValueNetwork(state_dim, cfg.hidden).to(device)
        self.target_value.load_state_dict(self.value.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
        )

        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=cfg.critic_lr,
        )

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action, _ = self.actor.sample(
            state_t,
            deterministic=deterministic,
        )

        return action.squeeze(0).cpu().numpy()

    def soft_update(self, source, target):
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_param.data.mul_(1.0 - self.cfg.tau)
            target_param.data.add_(
                self.cfg.tau * source_param.data
            )

    def update(self, replay):
        states, actions, rewards, next_states, dones = replay.sample(
            self.cfg.batch_size,
            self.device,
        )

        with torch.no_grad():
            next_values = self.target_value(next_states)
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * next_values

        q1, q2 = self.critic(states, actions)

        critic_loss = (
            F.mse_loss(q1, target_q)
            + F.mse_loss(q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()



        with torch.no_grad():
            policy_action, _ = self.actor.sample(states)
            q1_pi, q2_pi = self.target_critic(states, policy_action)
            q_pi = torch.minimum(q1_pi, q2_pi)

        value = self.value(states)
        value_loss = F.mse_loss(value, q_pi)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 5.0)
        self.value_optimizer.step()


        with torch.no_grad():
            q1_data, q2_data = self.target_critic(states, actions)
            q_data = torch.minimum(q1_data, q2_data)
            baseline = self.target_value(states)

            advantage = q_data - baseline

            weights = torch.exp(
                advantage / self.cfg.awac_temperature
            ).clamp(
                max=self.cfg.max_weight
            )

            # Normalize weights within the batch for smoother learning.
            weights = weights / (weights.mean() + 1e-8)

        dist = self.actor.distribution(states)
        mean, log_std = self.actor(states)

        # Because the replay actions are already inside [-1, 1], invert tanh
        # approximately to obtain the corresponding Gaussian-space action.
        clipped_actions = actions.clamp(-0.999, 0.999)
        pre_tanh = torch.atanh(clipped_actions)

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)

        actor_loss = -(weights * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()


        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.value, self.target_value)

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_advantage": advantage.mean().item(),
            "max_weight": weights.max().item(),
        }


def evaluate(agent, episodes, render=False):
    env = gym.make(
        "BipedalWalker-v3",
        render_mode="human" if render else None,
    )

    scores = []

    for episode in range(episodes):
        state, _ = env.reset(seed=10_000 + episode)
        done = False
        total = 0.0

        while not done:
            action = agent.act(state, deterministic=True)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        scores.append(total)
        print(
            f"Evaluation {episode + 1:2d}/{episodes}: "
            f"return={total:8.2f}"
        )

    env.close()

    print(
        f"Evaluation mean: {np.mean(scores):.2f} "
        f"+/- {np.std(scores):.2f}"
    )


def train(cfg):
    seed_everything(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = gym.make("BipedalWalker-v3")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = AWAC(
        state_dim,
        action_dim,
        cfg,
        device,
    )

    replay = ReplayBuffer(cfg.replay_capacity)

    global_step = 0
    best_return = -float("inf")

  

    print(
        f"Device: {device}\n"
        f"State dimension: {state_dim}\n"
        f"Action dimension: {action_dim}\n"
        f"Collecting {cfg.warmup_steps} warm-up transitions..."
    )

    state, _ = env.reset(seed=cfg.seed)

    for _ in range(cfg.warmup_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

        global_step += 1

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    print("Warm-up complete.\n")
  

    for episode in range(1, cfg.episodes + 1):
        state, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        info = {}

        for _ in range(env.spec.max_episode_steps):
            # Add mild exploration through the stochastic policy.
            action = agent.act(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            episode_return += reward
            episode_length += 1
            global_step += 1

            if done:
                break

        # Several gradient updates from the accumulated replay data.
        for _ in range(cfg.updates_per_episode):
            if len(replay) >= cfg.batch_size:
                info = agent.update(replay)

        if episode_return > best_return:
            best_return = episode_return
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "value": agent.value.state_dict(),
                    "algorithm": "AWAC",
                    "environment": "BipedalWalker-v3",
                },
                "awac_bipedalwalker_best.pt",
            )

        print(
            f"Episode {episode:3d}/{cfg.episodes} | "
            f"return={episode_return:8.2f} | "
            f"best={best_return:8.2f} | "
            f"len={episode_length:3d} | "
            f"critic={info.get('critic_loss', 0):7.3f} | "
            f"actor={info.get('actor_loss', 0):7.3f} | "
            f"adv={info.get('mean_advantage', 0):7.3f}"
        )

    env.close()

    # Restore the best actor.
    checkpoint = torch.load(
        "awac_bipedalwalker_best.pt",
        map_location=device,
        weights_only=False,
    )
    agent.actor.load_state_dict(checkpoint["actor"])

    return agent


def main():
    parser = argparse.ArgumentParser(
        description="One-file AWAC project for BipedalWalker-v3"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=250,
        help="training episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render evaluation episodes",
    )
    args = parser.parse_args()

    cfg = Config(episodes=args.episodes)

    agent = train(cfg)

    print("\nBest-policy evaluation:")
    evaluate(
        agent,
        episodes=cfg.eval_episodes,
        render=args.render,
    )

if __name__ == "__main__":
    main()
