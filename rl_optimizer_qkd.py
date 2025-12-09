# rl_optimizer_qkd.py
"""
DQN agent that interacts with BB84/Cirq environment (qkd_cirq.bb84_cirq).
Uses GAN to produce adversarial vector that determines channel flip & noise mapping.
Actions: discrete QBER acceptance thresholds.
State: last_qber, last_sift_frac
"""

import numpy as np
import random
from collections import deque, namedtuple
import torch, torch.nn as nn, torch.optim as optim
from qkd_cirq import bb84_cirq
from gan_attack import Generator, generate_noise
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=10000): self.buf=deque(maxlen=capacity)
    def push(self,*args): self.buf.append(Transition(*args))
    def sample(self,b): import random; s=random.sample(self.buf,b); return Transition(*zip(*s))
    def __len__(self): return len(self.buf)

class QNet(nn.Module):
    def __init__(self, sdim, adim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim,hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, adim))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_list, lr=1e-3):
        self.actions = action_list
        self.q = QNet(state_dim, len(action_list)).to(DEVICE)
        self.target = QNet(state_dim, len(action_list)).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.replay=ReplayBuffer(20000)
        self.gamma=0.99; self.tau=1e-3

    def select(self, state, eps):
        if random.random() < eps: return random.randrange(len(self.actions))
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad(): qv = self.q(s)
        return int(torch.argmax(qv).item())

    def push(self,*args): self.replay.push(*args)
    def update(self, batch_size=64):
        if len(self.replay) < batch_size: return 0.0
        trans = self.replay.sample(batch_size)
        s = torch.tensor(np.array(trans.state), dtype=torch.float32, device=DEVICE)
        a = torch.tensor(trans.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
        r = torch.tensor(trans.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(np.array(trans.next_state), dtype=torch.float32, device=DEVICE)
        done = torch.tensor(trans.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        qvals = self.q(s).gather(1,a)
        with torch.no_grad():
            next_q = self.target(ns).max(1)[0].unsqueeze(1)
            target = r + (1.0 - done) * self.gamma * next_q
        loss = nn.functional.mse_loss(qvals, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for tparam, param in zip(self.target.parameters(), self.q.parameters()):
            tparam.data.copy_(self.tau*param.data + (1-self.tau)*tparam.data)
        return loss.item()

# Environment using bb84_cirq
class QKDEnv:
    def __init__(self, gan_generator=None, n_bits=1024, seed=SEED):
        self.gan = gan_generator
        self.n_bits = n_bits
        self.seed = seed

    def reset(self):
        a,b,meta = bb84_cirq(n_bits=self.n_bits, noise_prob=0.0, eve_prob=0.0, seed=self.seed)
        state = np.array([meta["qber"], meta["sifted_len"]/max(1,self.n_bits)], dtype=np.float32)
        return state

    def step(self, action_idx):
        # sample adversarial vector from GAN
        if self.gan is not None:
            vec = generate_noise(self.gan, n_samples=1)[0]
        else:
            vec = np.random.randn(128)
        # map vector mean to channel flip prob
        flip = float(min(0.15, max(0.0, 0.06 + 0.04 * np.tanh(np.mean(vec)))))
        # run Cirq BB84 with this flip as noise_prob
        a,b,meta = bb84_cirq(n_bits=self.n_bits, noise_prob=flip, eve_prob=0.0, seed=random.randint(0,1<<30))
        qber = meta["qber"]; sift_frac = meta["sifted_len"]/max(1,self.n_bits)
        state = np.array([qber, sift_frac], dtype=np.float32)
        done = False
        info = {"meta":meta, "flip":flip}
        return state, None, done, info  # reward computed by trainer

# ---------- Evaluation helpers ----------

from math import log2

def _h2(p: float) -> float:
    """Binary entropy (safe-clamped)."""
    p = max(min(float(p), 1 - 1e-12), 1e-12)
    return -(p * log2(p) + (1 - p) * log2(1 - p))

def estimate_key_rate(sift_frac: float, qber: float, leak_ec: float = 0.1) -> float:
    """
    Very simple secret-key rate proxy for BB84:
      R ≈ sift_frac * (1 - h2(QBER) - leak_ec)
    Clamp at 0 for readability.
    """
    return max(0.0, sift_frac * (1.0 - _h2(qber) - leak_ec))

def greedy_action(agent, state: np.ndarray) -> int:
    """Epsilon=0 greedy pick using the learned Q-net."""
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        qvals = agent.q(s)
        a = int(torch.argmax(qvals).item())
    return a

def evaluate_agent(agent,
                   gan_generator=None,
                   episodes: int = 50,
                   n_bits: int = 512,
                   seed: int | None = 2025):
    """
    Roll out the trained agent with greedy policy and summarize performance.
    Returns a dict of metrics.
    """
    rng = np.random.default_rng(seed)
    env = QKDEnv(gan_generator=gan_generator, n_bits=n_bits, seed=SEED)

    # collections
    rewards, qbers, sifts, flips, accepts, thresh_used, key_rates = [], [], [], [], [], [], []

    for _ in range(episodes):
        state = env.reset()
        a_idx = greedy_action(agent, state)
        next_state, _, done, info = env.step(a_idx)

        qber = float(next_state[0])
        sift_frac = float(next_state[1])
        flip = float(info["flip"])
        threshold = float(agent.actions[a_idx])

        # use the same reward rule as training for apples-to-apples
        if qber <= threshold:
            reward = info['meta']['sifted_len'] * (1.0 - qber)
            accepted = 1
        else:
            reward = - (info['meta']['sifted_len'] + 10000 * (qber - threshold))
            accepted = 0

        # key rate proxy
        kr = estimate_key_rate(sift_frac, qber, leak_ec=0.1)

        # log
        rewards.append(reward)
        qbers.append(qber)
        sifts.append(sift_frac)
        flips.append(flip)
        accepts.append(accepted)
        thresh_used.append(threshold)
        key_rates.append(kr)

    if len(rewards) == 0:
        return None

    metrics = {
        "episodes": int(episodes),
        "avg_reward": float(np.mean(rewards)),
        "avg_qber": float(np.mean(qbers)),
        "avg_sift_frac": float(np.mean(sifts)),
        "avg_flip": float(np.mean(flips)),
        "accept_rate": float(np.mean(accepts)),
        "avg_key_rate": float(np.mean(key_rates)),
        "threshold_mean": float(np.mean(thresh_used)),
        "threshold_std": float(np.std(thresh_used)),
    }
    return metrics


def train_agent(gan_generator, episodes=120, n_bits=1024, out="dqn_qkd_cirq.pt"):
    actions = list(np.linspace(0.005, 0.15, 16))  # thresholds
    env = QKDEnv(gan_generator, n_bits=n_bits)
    agent = DQNAgent(state_dim=2, action_list=actions)
    agent.action_list = actions
    eps=1.0; eps_min=0.05; decay=0.98
    best = -1e9
    for ep in range(1, episodes+1):
        state = env.reset()
        total = 0.0
        act_idx = agent.select(state, eps)
        ns, _, done, info = env.step(act_idx)
        qber = ns[0]; sift_frac = ns[1]
        threshold = actions[act_idx]
        if qber <= threshold:
            reward = info['meta']['sifted_len'] * (1.0 - qber)
        else:
            reward = - (info['meta']['sifted_len'] + 10000*(qber - threshold))
        agent.push(state, act_idx, reward, ns, False)
        loss = agent.update(batch_size=64)
        total += reward
        eps = max(eps_min, eps*decay)
        print(f"Ep {ep:03d} reward {total:.2f} eps {eps:.3f} qber {qber:.4f} sift_frac {sift_frac:.4f} flip {info['flip']:.4f}")
        if total > best:
            best = total
            torch.save({"qnet":agent.q.state_dict(), "actions":actions, "episode":ep}, out)
    print("Done training. Best reward:", best, "model saved to", out)
    return agent

if __name__ == "__main__":
    gen = Generator()
    print("Training RL agent demo (Cirq, GAN untrained)...")
    agent = train_agent(gen, episodes=40, n_bits=512)
