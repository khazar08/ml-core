"""
Multi-Armed Bandits: epsilon-greedy, UCB, Thompson Sampling.
Run: python multi_armed_bandits.py
Outputs: results/bandit_comparison.png, results/bandit_summary.csv
"""
import csv
import os
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BanditEnv(Protocol):
    k: int
    optimal_arm: int
    def step(self, arm: int) -> float: ...


@dataclass
class GaussianBandit:
    means: np.ndarray
    sigma: float = 1.0
    seed: int = 0

    def __post_init__(self):
        self.k = len(self.means)
        self.optimal_arm = int(np.argmax(self.means))
        self.rng = np.random.default_rng(self.seed)

    def step(self, arm: int) -> float:
        return float(self.rng.normal(self.means[arm], self.sigma))


@dataclass
class BernoulliBandit:
    probs: np.ndarray
    seed: int = 0

    def __post_init__(self):
        self.k = len(self.probs)
        self.optimal_arm = int(np.argmax(self.probs))
        self.rng = np.random.default_rng(self.seed)

    def step(self, arm: int) -> float:
        return float(self.rng.random() < self.probs[arm])


class BanditAgent(Protocol):
    name: str
    def select_arm(self) -> int: ...
    def update(self, arm: int, reward: float) -> None: ...


class EpsilonGreedy:
    def __init__(self, k, epsilon=0.1, seed=0):
        self.name = f"eps-greedy(eps={epsilon})"
        self.k = k
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k, int)
        self.values = np.zeros(k)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        best = np.flatnonzero(self.values == self.values.max())
        return int(self.rng.choice(best))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class UCB1:
    def __init__(self, k, c=2.0, seed=0):
        self.name = f"UCB1(c={c})"
        self.k = k
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k, int)
        self.values = np.zeros(k)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        if np.any(self.counts == 0):
            return int(np.flatnonzero(self.counts == 0)[0])
        bonus = self.c * np.sqrt(2 * np.log(self.t) / self.counts)
        ucb = self.values + bonus
        best = np.flatnonzero(ucb == ucb.max())
        return int(self.rng.choice(best))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonGaussian:
    def __init__(self, k, sigma=1.0, sigma0=5.0, seed=0):
        self.name = "Thompson(Gaussian)"
        self.k = k
        self.sigma = sigma
        self.sigma0 = sigma0
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k, int)
        self.sums = np.zeros(k)

    def select_arm(self) -> int:
        prec = 1 / self.sigma0**2 + self.counts / self.sigma**2
        mean = (self.sums / self.sigma**2) / prec
        return int(np.argmax(self.rng.normal(mean, np.sqrt(1 / prec))))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward


class ThompsonBernoulli:
    def __init__(self, k, seed=0):
        self.name = "Thompson(Bernoulli)"
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_arm(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm, reward):
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


# ------------------------------------------------------------
# Experiment runner: shared by both environment types.
# Records cumulative regret and optimal-action fraction per step.
# ------------------------------------------------------------
def run_experiment(env_factory, agent_factory, n_steps=2000, n_runs=200):
    k = env_factory().k
    regret = np.zeros((n_runs, n_steps))
    optimal = np.zeros((n_runs, n_steps), bool)
    for r in range(n_runs):
        env = env_factory()
        agent = agent_factory(k, seed=r)
        optimal_mean = env.means[env.optimal_arm] if hasattr(env, "means") else env.probs[env.optimal_arm]
        cumulative = 0.0
        for t in range(n_steps):
            arm = agent.select_arm()
            reward = env.step(arm)
            optimal[r, t] = arm == env.optimal_arm
            cumulative += optimal_mean - reward
            regret[r, t] = cumulative
            agent.update(arm, reward)
    return regret.mean(axis=0), optimal.mean(axis=0)


def compare_on_env(env_name, env_factory, agents, n_steps=2000, n_runs=200):
    results = {}
    for agent_cls in agents:
        regret, opt = run_experiment(env_factory, agent_cls, n_steps, n_runs)
        results[agent_cls(k=len(env_factory().means) if hasattr(env_factory(), "means")
                          else len(env_factory().probs)).name] = (regret, opt)
    return results


def main():
    os.makedirs("results", exist_ok=True)
    gauss_means = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    bern_probs = np.array([0.05, 0.10, 0.15, 0.30, 0.45])

    gauss_agents = [EpsilonGreedy, UCB1, ThompsonGaussian]
    bern_agents = [EpsilonGreedy, UCB1, ThompsonBernoulli]

    gauss_res = compare_on_env(
        "Gaussian", lambda: GaussianBandit(gauss_means), gauss_agents
    )
    bern_res = compare_on_env(
        "Bernoulli", lambda: BernoulliBandit(bern_probs), bern_agents
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (name, (reg, opt)), ax_row in zip(gauss_res.items(), axes[0]):
        pass
    for ax, (name, (reg, opt)) in zip(axes[0], gauss_res.items()):
        ax.plot(reg, label=name)
        ax.set_title(f"Gaussian — {name}")
        ax.set_xlabel("steps"); ax.set_ylabel("cumulative regret")
        ax.legend()
    for ax, (name, (reg, opt)) in zip(axes[1], bern_res.items()):
        ax.plot(opt, label=name)
        ax.set_title(f"Bernoulli — {name}")
        ax.set_xlabel("steps"); ax.set_ylabel("optimal-action fraction")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/bandit_comparison.png", dpi=120)
    plt.close()

    with open("results/bandit_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "agent", "final_regret", "final_optimal_frac"])
        for env, res in [("Gaussian", gauss_res), ("Bernoulli", bern_res)]:
            for name, (reg, opt) in res.items():
                w.writerow([env, name, f"{reg[-1]:.2f}", f"{opt[-1]:.3f}"])

    print(f"{'env':<10}{'agent':<25}{'regret':>10}{'opt%':>8}")
    for env, res in [("Gaussian", gauss_res), ("Bernoulli", bern_res)]:
        for name, (reg, opt) in res.items():
            print(f"{env:<10}{name:<25}{reg[-1]:>10.2f}{opt[-1]*100:>7.1f}%")


if __name__ == "__main__":
    main()
