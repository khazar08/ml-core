from __future__ import annotations
import argparse
import heapq
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

SHORTCUT_MAZE = [
    "........G",
    ".........",
    ".........",
    ".#######D",
    ".........",
    "...S.....",
]
BEFORE_CHANGE = SHORTCUT_MAZE  

ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}
ACTION_GLYPH = {0: "^", 1: "v", 2: "<", 3: ">"}
N_ACTIONS = len(ACTIONS)


class ShortcutMaze:

    def __init__(self, layout=BEFORE_CHANGE, change_at: int = 3000):
        self.grid = [list(row) for row in layout]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.change_at = change_at
        self.door_open = False
        self.global_step = 0

        self.start = self._find("S")
        self.goal = self._find("G")
        self.door = self._find("D")
        self.pos = self.start

    # helpers
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == ch:
                    return (r, c)
        raise ValueError(f"cell {ch!r} not present in layout")

    def _walkable(self, r, c):
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return False
        cell = self.grid[r][c]
        if cell == "#":
            return False
        if cell == "D":
            return self.door_open
        return True

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    def to_index(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def to_pos(self, idx):
        return divmod(idx, self.n_cols)


    def reset(self):
        self.pos = self.start
        return self.to_index(self.pos)

    def step(self, action):
        self.global_step += 1
        if not self.door_open and self.global_step >= self.change_at:
            self.door_open = True  # the world silently changes

        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if self._walkable(nr, nc):
            self.pos = (nr, nc)  # bumping a wall = stay put

        done = self.pos == self.goal
        reward = 50.0 if done else -1.0
        return self.to_index(self.pos), reward, done

    def render(self, agent_pos=None, path=None):
        path = set(path or [])
        out = []
        for r, row in enumerate(self.grid):
            line = []
            for c, cell in enumerate(row):
                p = (r, c)
                if agent_pos is not None and p == agent_pos:
                    line.append("A")
                elif cell == "D":
                    line.append("." if self.door_open else "#")
                elif p in path and cell not in "SG":
                    line.append("o")
                else:
                    line.append(cell)
            out.append(" ".join(line))
        return "\n".join(out)


@dataclass
class DynaAgent:
    

    n_states: int
    n_actions: int = N_ACTIONS
    alpha: float = 0.5
    gamma: float = 0.95
    epsilon: float = 0.1
    planning_steps: int = 20
    kappa: float = 0.0
    prioritized: bool = False
    theta: float = 0.05          # priority threshold for sweeping
    optimistic_unseen: bool = True
    seed: int = 0

    Q: np.ndarray = field(init=False)
    model: dict = field(init=False)
    last_tried: dict = field(init=False)
    predecessors: dict = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.model = {}                       # (s, a) -> (r, s_next)
        self.last_tried = {}                  # (s, a) -> global step of last real try
        self.predecessors = defaultdict(set)  # s_next -> {(s, a)}
        self.queue = []                       # heap of (-priority, s, a)
        self.in_queue = set()
        self.t = 0

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def td_error(self, s, a, r, s_next):
        return r + self.gamma * self.Q[s_next].max() - self.Q[s, a]

    def _bonus(self, s, a):
        if self.kappa <= 0:
            return 0.0
        tau = self.t - self.last_tried.get((s, a), 0)
        return self.kappa * float(np.sqrt(tau))

    def observe(self, s, a, r, s_next, done):
        self.t += 1

        # Dyna-Q+ trick: when we first touch a state, register every action as "known"
        # so untried actions can also accumulate an exploration bonus over time.
        if self.kappa > 0 and self.optimistic_unseen:
            for a_ in range(self.n_actions):
                self.model.setdefault((s, a_), (0.0, s))
                self.last_tried.setdefault((s, a_), 0)

        self.model[(s, a)] = (r, s_next)
        self.last_tried[(s, a)] = self.t
        self.predecessors[s_next].add((s, a))

        self.Q[s, a] += self.alpha * self.td_error(s, a, r, s_next)
        if self.prioritized:
            self._push(s, a, abs(self.td_error(s, a, r, s_next)))
        self.plan()

    def _push(self, s, a, priority):
        if priority > self.theta and (s, a) not in self.in_queue:
            heapq.heappush(self.queue, (-priority, s, a))
            self.in_queue.add((s, a))

    def plan(self):
        """Run `planning_steps` updates on the learned model ('dreaming')."""
        if self.planning_steps <= 0 or not self.model:
            return

        if self.prioritized:
            for _ in range(self.planning_steps):
                if not self.queue:
                    break
                _, s, a = heapq.heappop(self.queue)
                self.in_queue.discard((s, a))
                r, s_next = self.model[(s, a)]
                self.Q[s, a] += self.alpha * self.td_error(s, a, r + self._bonus(s, a), s_next)
                # propagate backwards to whoever can reach s
                for (sp, ap) in self.predecessors[s]:
                    rp, _ = self.model[(sp, ap)]
                    self._push(sp, ap, abs(self.td_error(sp, ap, rp + self._bonus(sp, ap), s)))
            return

        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            s, a = self.py_rng.choice(keys)
            r, s_next = self.model[(s, a)]
            # curiosity about stale knowledge -> re-checks parts of the world it ignored
            self.Q[s, a] += self.alpha * self.td_error(s, a, r + self._bonus(s, a), s_next)

    # utils
    def greedy_path(self, env, max_len=200):
        s = env.reset()
        path = [env.to_pos(s)]
        for _ in range(max_len):
            a = int(np.argmax(self.Q[s]))
            s, _, done = env.step(a)
            path.append(env.to_pos(s))
            if done:
                break
        return path




def run(agent_kwargs, episodes=500, max_steps=400, change_at=2000, seed=0, layout=BEFORE_CHANGE):
    env = ShortcutMaze(layout, change_at=change_at)
    agent = DynaAgent(n_states=env.n_states, seed=seed, **agent_kwargs)

    steps_per_ep, returns, cumulative, cum = [], [], [], 0.0
    for _ in range(episodes):
        s = env.reset()
        total, n = 0.0, 0
        for _ in range(max_steps):
            a = agent.act(s)
            s_next, r, done = env.step(a)
            agent.observe(s, a, r, s_next, done)
            s, total, n = s_next, total + r, n + 1
            if done:
                break
        steps_per_ep.append(n)
        returns.append(total)
        cum += total
        cumulative.append(cum)
    return agent, env, dict(steps=steps_per_ep, returns=returns, cumulative=cumulative)


def average_runs(agent_kwargs, n_seeds=5, **kw):
    curves = []
    last = None
    for seed in range(n_seeds):
        agent, env, hist = run(agent_kwargs, seed=seed, **kw)
        curves.append(hist["steps"])
        last = (agent, env, hist)
    return np.mean(np.array(curves, dtype=float), axis=0), last


def smooth(x, k=9):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def value_and_policy_grids(agent, env):
    V = np.full((env.n_rows, env.n_cols), np.nan)
    P = np.full((env.n_rows, env.n_cols), " ", dtype=object)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.grid[r][c] == "#":
                continue
            if env.grid[r][c] == "D" and not env.door_open:
                continue
            idx = env.to_index((r, c))
            V[r, c] = agent.Q[idx].max()
            P[r, c] = ACTION_GLYPH[int(np.argmax(agent.Q[idx]))]
    P[env.goal] = "G"
    return V, P


def print_policy(agent, env, title):
    _, P = value_and_policy_grids(agent, env)
    print(f"\n  greedy policy -- {title}")
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            cell = env.grid[r][c]
            blocked = cell == "#" or (cell == "D" and not env.door_open)
            row.append("#" if blocked else P[r, c])
        print("   " + " ".join(row))


def make_plots(results, best_agent, env, path, best_label="best", change_at=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    # 1. learning curves
    ax = fig.add_subplot(gs[0, :2])
    for label, curve in results.items():
        ax.plot(smooth(curve), lw=2, label=label)
    ax.axhline(len(best_agent.greedy_path(env)) - 1, ls=":", c="0.4", lw=1.2,
               label="optimal (post-change)")
    ref = next(iter(results.values()))
    change_ep = int(np.searchsorted(np.cumsum(ref), change_at))
    ax.axvline(change_ep, c="crimson", ls="--", lw=1.2, alpha=0.7)
    ax.annotate("shortcut opens", xy=(change_ep, ax.get_ylim()[1]), xytext=(6, -12),
                textcoords="offset points", color="crimson", fontsize=9, va="top")
    ax.set_title("Steps to goal per episode (lower is better, smoothed)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 2. state-value heatmap
    ax = fig.add_subplot(gs[0, 2])
    V, P = value_and_policy_grids(best_agent, env)
    im = ax.imshow(np.ma.masked_invalid(V), cmap="viridis")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if P[r, c] != " ":
                ax.text(c, r, P[r, c], ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
    ax.set_title(f"Learned V(s) + greedy policy\n{best_label}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 3. cumulative regret-ish view
    ax = fig.add_subplot(gs[1, :2])
    for label, curve in results.items():
        ax.plot(np.cumsum(curve), lw=2, label=label)
    ax.set_title("Cumulative environment steps consumed")
    ax.set_xlabel("episode")
    ax.set_ylabel("total steps")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 4. maze + discovered route
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    route = best_agent.greedy_path(env)
    ax.text(0, 1, env.render(path=route), family="monospace", fontsize=11,
            va="top", ha="left", transform=ax.transAxes)
    ax.set_title(f"Greedy route after training ({len(route)-1} steps)", fontsize=10)

    fig.suptitle("Dyna-Q+ with Prioritized Sweeping on a Non-Stationary Shortcut Maze",
                 fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure -> {path}")


# main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--planning", type=int, default=20)
    ap.add_argument("--kappa", type=float, default=5e-2)
    ap.add_argument("--change-at", type=int, default=2000,
                    help="global step at which the shortcut opens")
    ap.add_argument("--out", default="dyna_q_plus_results.png")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--demo", action="store_true", help="ASCII rollout of trained agent")
    args = ap.parse_args()

    common = dict(episodes=args.episodes, change_at=args.change_at)

    configs = {
        "no planning (n=0)":
            dict(planning_steps=0, kappa=0.0),
        "Dyna-Q (n=%d)" % args.planning:
            dict(planning_steps=args.planning, kappa=0.0),
        "Dyna-Q+ (n=%d, kappa=%g)" % (args.planning, args.kappa):
            dict(planning_steps=args.planning, kappa=args.kappa),
        "Dyna-Q+ prioritized sweeping":
            dict(planning_steps=args.planning, kappa=args.kappa, prioritized=True),
    }

    print("=" * 78)
    print("Dyna-Q+ / Prioritized Sweeping -- non-stationary maze")
    print(f"episodes={args.episodes}  seeds={args.seeds}  shortcut opens at step {args.change_at}")
    print("=" * 78)

    results, agents = {}, {}
    for label, kwargs in configs.items():
        curve, (agent, env, _) = average_runs(kwargs, n_seeds=args.seeds, **common)
        results[label] = curve
        agents[label] = (agent, env)
        first, last = curve[: max(1, args.episodes // 10)], curve[-20:]
        print(f"\n{label}")
        print(f"   mean steps, first 10% of episodes : {first.mean():8.1f}")
        print(f"   mean steps, final 20 episodes     : {last.mean():8.1f}")
        print(f"   total steps used                  : {curve.sum():8.0f}")
        print(f"   greedy path length after training : {len(agent.greedy_path(env)) - 1:8d}")

    best_label = min(results, key=lambda k: results[k][-20:].mean())
    best_agent, best_env = agents[best_label]
    print(f"\nBest configuration: {best_label}")
    print_policy(best_agent, best_env, best_label)

    if args.demo:
        print("\nRollout of the greedy policy:")
        s = best_env.reset()
        for _ in range(60):
            print("\n" + best_env.render(agent_pos=best_env.to_pos(s)))
            a = int(np.argmax(best_agent.Q[s]))
            s, _, done = best_env.step(a)
            if done:
                print("\n" + best_env.render(agent_pos=best_env.to_pos(s)))
                print("reached the goal")
                break

    if not args.no_plot:
        try:
            make_plots(results, best_agent, best_env, args.out,
                   best_label=best_label, change_at=args.change_at)
        except ImportError:
            print("matplotlib not installed -- skipping plots")


if __name__ == "__main__":
    main()
