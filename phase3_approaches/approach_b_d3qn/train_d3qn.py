"""
train_d3qn.py — Dueling D3QN + GRU + PER trainer for OBELIX (Phase 3)

Key features:
  1) Dueling architecture: V(s) + A(s,a) - mean(A)
  2) Double DQN target selection
  3) GRU for partial observability (POMDP memory)
  4) Prioritized Experience Replay (PER)
  5) n-step returns for faster credit assignment
  6) Curriculum learning + dense reward shaping
  7) Multi-difficulty training schedule

Usage:
  cd phase3_approaches/approach_b_d3qn
  python train_d3qn.py --episodes 8000 --device auto
"""

import argparse, os, sys, time, random, math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM, ACTION_DIM, STACK_SIZE = 18, 5, 4
INPUT_DIM = OBS_DIM * STACK_SIZE + ACTION_DIM
HIDDEN_DIM = 128


                                       
class DuelingDRQN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.val_stream = nn.Linear(hidden_dim, 1)
        self.adv_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.gru(x, hidden)
        val = self.val_stream(x)
        adv = self.adv_stream(x)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        return q, hidden


                                                  
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=500000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.max_priority = 1.0
        self.eps = 1e-5
        self._step = 0

    def push(self, transition):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        self._step += 1
        self.beta = min(self.beta_end,
                        self.beta_start + (self.beta_end - self.beta_start) * self._step / self.beta_steps)

        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, pri, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total())
                idx, pri, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(pri)

        probs = np.array(priorities, dtype=np.float64) / (self.tree.total() + 1e-10)
        weights = (self.tree.n_entries * probs + 1e-10) ** (-self.beta)
        weights /= weights.max()
        return batch, indices, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


                                                
def build_stacked(frame_stack):
    frames = list(frame_stack)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, dtype=np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:], axis=0).astype(np.float32)


def build_seq(history, seq_len, dim):
    h = list(history)
    if len(h) >= seq_len:
        return np.array(h[-seq_len:], dtype=np.float32)
    pad = [np.zeros(dim, dtype=np.float32)] * (seq_len - len(h))
    return np.array(pad + h, dtype=np.float32)


def curriculum_reset(env, progress):
    obs = env.reset()
    if progress > 0.6:
        return obs

    box_x, box_y = env.box_center_x, env.box_center_y
    margin = env.bot_radius + 15

    if progress < 0.15:
        dist = random.randint(20, 40)
    elif progress < 0.30:
        dist = random.randint(40, 90)
    elif progress < 0.45:
        dist = random.randint(80, 160)
    else:
        dist = random.randint(100, 250)

    angle = random.uniform(0, 360)
    bx = int(box_x + dist * math.cos(math.radians(angle)))
    by = int(box_y + dist * math.sin(math.radians(angle)))
    bx = max(margin, min(env.frame_size[1] - margin, bx))
    by = max(margin, min(env.frame_size[0] - margin, by))
    env.bot_center_x, env.bot_center_y = bx, by
    env.facing_angle = int(math.degrees(math.atan2(box_y - by, box_x - bx)) + random.uniform(-25, 25))
    env._update_frames(show=False)
    env.get_feedback()
    env.update_reward()
    return env.sensor_feedback.copy()


def shape_reward(obs, prev_obs, action_idx, env_reward, action_hist):
    shaped = env_reward
    s = float(np.sum(obs[:17]))
    ps = float(np.sum(prev_obs[:17]))
    shaped += 2.5 * (s - ps)
    if obs[16] == 1 and action_idx == 2:
        shaped += 8.0
    if s > 0 and action_idx == 2:
        shaped += 3.0
    if s == 0:
        shaped += 12.0
        if action_idx == 2:
            shaped += 1.5
    if len(action_hist) >= 6 and all(a != 2 for a in list(action_hist)[-6:]):
        shaped -= 4.0
    if obs[17] == 1:
        shaped -= 30.0
    return shaped


def get_difficulty(progress):
    if progress < 0.20:
        return 0, False
    elif progress < 0.35:
        return 0, True
    elif progress < 0.50:
        return 2, random.random() < 0.5
    elif progress < 0.65:
        return 3, False
    else:
        return random.choice([0, 2, 3, 3, 3]), random.random() < 0.5


                                            
class D3QNAgent:
    def __init__(self, lr, gamma, seq_len, batch_size, buffer_cap,
                 eps_start, eps_end, eps_decay, target_update, n_step, device):
        self.gamma = gamma
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_step = n_step
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps = 0
        self.device = device

        self.online = DuelingDRQN().to(device)
        self.target = DuelingDRQN().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PERBuffer(buffer_cap)
        self.n_step_buf = deque(maxlen=n_step)

    def select_action(self, obs_seq):
        self.eps = self.eps_end + (1.0 - self.eps_end) * math.exp(-self.steps / self.eps_decay)
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(ACTION_DIM)
        s = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q, _ = self.online(s)
        return int(q[:, -1, :].argmax(dim=1).item())

    def _calc_n_step(self):
        """Collapse n-step buffer into single (s, a, R, s', done) transition."""
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buf):
            R += (self.gamma ** i) * r
            if d:
                break
        s0 = self.n_step_buf[0][0]
        a0 = self.n_step_buf[0][1]
        sn = self.n_step_buf[-1][3]
        dn = self.n_step_buf[-1][4]
        return (s0.copy(), a0, R, sn.copy(), dn)

    def step(self, obs_seq, action, reward, next_obs_seq, done):
        self.n_step_buf.append((obs_seq, action, reward, next_obs_seq, done))
        if len(self.n_step_buf) >= self.n_step or done:
            transition = self._calc_n_step()
            self.buffer.push(transition)

        if len(self.buffer) >= self.batch_size:
            self.learn()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def learn(self):
        batch, indices, is_weights = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        S = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        A = torch.as_tensor(a, dtype=torch.long, device=self.device)
        R = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        NS = torch.as_tensor(np.array(ns), dtype=torch.float32, device=self.device)
        D = torch.as_tensor(d, dtype=torch.float32, device=self.device)
        W = is_weights.to(self.device)

        q_all, _ = self.online(S)
        q_sa = q_all[:, -1, :].gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online, _ = self.online(NS)
            next_acts = next_online[:, -1, :].argmax(dim=1, keepdim=True)
            next_target, _ = self.target(NS)
            next_q = next_target[:, -1, :].gather(1, next_acts).squeeze(1)
            target = R + (self.gamma ** self.n_step) * next_q * (1 - D)

        td_errors = (q_sa - target).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        loss = (W * F.smooth_l1_loss(q_sa, target, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()


                                               
def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps)

    agent = D3QNAgent(
        lr=args.lr, gamma=args.gamma, seq_len=args.seq_len, batch_size=args.batch,
        buffer_cap=args.buffer, eps_start=args.eps_start, eps_end=args.eps_end,
        eps_decay=args.eps_decay, target_update=args.target_update, n_step=args.n_step,
        device=device,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or '.', exist_ok=True)
    best_reward_path = os.path.join(
        os.path.dirname(os.path.abspath(args.save)) or ".",
        "weights_best_reward.pth",
    )
    best_avg = -float('inf')
    best_raw = -float('inf')
    all_rewards = []
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  D3QN + GRU + PER | episodes={args.episodes} | n_step={args.n_step}")
    print(f"{'='*70}")

    for ep in range(1, args.episodes + 1):
        progress = ep / args.episodes
        diff, walls = get_difficulty(progress)
        env.difficulty = diff
        env.box_blink_enabled = diff >= 2
        env.box_move_enabled = diff >= 3
        env.wall_obstacles = walls

        obs = curriculum_reset(env, progress)
        frame_stack = deque(maxlen=STACK_SIZE)
        frame_stack.append(obs.astype(np.float32))
        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        action_hist = deque(maxlen=20)

        history = deque(maxlen=args.seq_len)
        stacked = build_stacked(frame_stack)
        aug = np.concatenate([stacked, prev_action])
        history.append(aug)

        agent.n_step_buf.clear()
        done = False
        ep_raw = 0.0

        while not done:
            obs_seq = build_seq(history, args.seq_len, INPUT_DIM)
            action_idx = agent.select_action(obs_seq)
            action_str = ACTIONS[action_idx]

            next_obs, env_reward, done = env.step(action_str, render=False)
            next_obs = next_obs.astype(np.float32)
            shaped = shape_reward(next_obs, obs, action_idx, env_reward, action_hist)

            frame_stack_next = deque(frame_stack, maxlen=STACK_SIZE)
            frame_stack_next.append(next_obs)
            stacked_next = build_stacked(frame_stack_next)
            prev_action_new = np.zeros(ACTION_DIM, dtype=np.float32)
            prev_action_new[action_idx] = 1.0
            aug_next = np.concatenate([stacked_next, prev_action_new])

            history_next = deque(history, maxlen=args.seq_len)
            history_next.append(aug_next)
            next_obs_seq = build_seq(history_next, args.seq_len, INPUT_DIM)

            agent.step(obs_seq, action_idx, shaped, next_obs_seq, done)

            obs = next_obs
            frame_stack = frame_stack_next
            history = history_next
            prev_action = prev_action_new
            action_hist.append(action_idx)
            ep_raw += env_reward

        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-50:]))

        if avg50 > best_avg:
            best_avg = avg50
            torch.save({"online": agent.online.state_dict()}, args.save)
            tag = " <-- BEST"
        else:
            tag = ""

        raw_tag = ""
        if ep_raw > best_raw:
            best_raw = ep_raw
            torch.save({"online": agent.online.state_dict()}, best_reward_path)
            raw_tag = " <-- BEST_RAW"

        if ep % args.checkpoint_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"d3qn_ep{ep}.pth")
            torch.save({"online": agent.online.state_dict()}, ckpt)

        if ep <= 200 or ep % 100 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} avg50={avg50:+8.1f} best={best_avg:+8.1f} | "
                f"eps={agent.eps:.3f} d={diff} w={int(walls)} | {elapsed:.1f}m eta={eta:.1f}m{tag}{raw_tag}")

    print(f"\nDone. Best avg50={best_avg:+.1f}  Saved: {args.save}")
    print(f"Best raw episode reward={best_raw:+.1f}  Saved: {best_reward_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=75000)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=300000)
    p.add_argument("--target_update", type=int, default=1000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_d3qn")
    p.add_argument("--checkpoint_every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
