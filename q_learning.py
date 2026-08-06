import random

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SYMBOLS = ["^", "v", "<", ">"]

class Warehouse:
    def __init__(self, size=6):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(2, 2), (2, 3), (4, 1)}
        self.danger = {(1, 3), (3, 1), (5, 2)}
        self.charging = {(0, 5), (5, 0)}
        self.energy_max = 20
        self.max_steps = 100

    def reset(self):
        self.pos = self.start
        self.energy = self.energy_max
        self.steps = 0
        return self.state()

    def state(self):
        return (self.pos[0] * self.size + self.pos[1]) * (self.energy_max + 1) + self.energy

    def num_states(self):
        return self.size * self.size * (self.energy_max + 1)

    def valid(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size and p not in self.obstacles

    def step(self, action):
        self.steps += 1
        self.energy -= 1

        if self.energy <= 0:
            self.energy = 0
            return self.state(), -10.0, True

        r = -0.1
        done = False
        old_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])

        if random.random() < 0.1:
            action = random.randrange(4)

        nr = self.pos[0] + ACTIONS[action][0]
        nc = self.pos[1] + ACTIONS[action][1]
        nxt = (nr, nc)

        if self.valid(nxt):
            self.pos = nxt

        new_dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        r += 0.2 * (old_dist - new_dist)

        if self.pos in self.charging:
            self.energy = min(self.energy_max, self.energy + 5)
            r += 0.5

        if self.pos in self.danger:
            self.energy = max(0, self.energy - 3)
            r -= 1.0
            if self.energy == 0:
                return self.state(), r - 5.0, True

        if self.pos == self.goal:
            r += 20.0 + self.energy * 0.2
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self.state(), r, done

    def states_at(self, p):
        base = (p[0] * self.size + p[1]) * (self.energy_max + 1)
        return [base + e for e in range(self.energy_max + 1)]

    def render_policy(self, q):
        rows = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                p = (i, j)
                if p == self.goal:
                    row.append("G")
                elif p in self.obstacles:
                    row.append("#")
                elif p in self.danger:
                    row.append("!")
                elif p in self.charging:
                    row.append("+")
                else:
                    best = max(range(4), key=lambda a: max(q[s][a] for s in self.states_at(p)))
                    row.append(SYMBOLS[best])
            rows.append(" ".join(row))
        return "\n".join(rows)

def epsilon_greedy(q, state, eps):
    if random.random() < eps:
        return random.randrange(4)

    vals = q[state]
    m = max(vals)
    return random.choice([a for a, v in enumerate(vals) if v == m])

def train(episodes=2500, alpha=0.1, gamma=0.95):
    random.seed(42)

    env = Warehouse()
    q = [[0.0 for _ in range(4)] for _ in range(env.num_states())]
    eps = 1.0
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        done = False

        while not done:
            a = epsilon_greedy(q, s, eps)
            ns, r, done = env.step(a)

            target = r if done else r + gamma * max(q[ns])
            q[s][a] += alpha * (target - q[s][a])

            s = ns
            total += r

        eps = max(0.02, eps * 0.997)
        rewards.append(total)

        if (ep + 1) % 100 == 0:
            window = rewards[-100:]
            print(f"Episode {ep + 1:5d} | Avg100 {sum(window) / len(window):7.2f} | Last {total:7.2f} | Eps {eps:.3f}")

    print(env.render_policy(q))
    return q

if __name__ == "__main__":
    train()
