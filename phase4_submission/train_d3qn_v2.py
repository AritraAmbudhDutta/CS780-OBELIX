"""
train_d3qn_v2.py — Fixed D3QN + GRU trainer (NO CURRICULUM, minimal shaping)

Same architecture as Phase 3's D3QN but fixed training:
  - NO curriculum (random spawns)
  - Minimal reward shaping
  - Always wall_obstacles=True
  - Heuristic exploration for search phase
  - Mixed difficulty

Usage:
  python launch.py --approach d3qn_v2 --episodes 12000
  OR: cd phase4_submission && python train_d3qn_v2.py --episodes 12000
"""

import argparse, os, sys, time, random, math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
        self.val = nn.Linear(hidden_dim, 1)
        self.adv = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.gru(x, hidden)
        v = self.val(x)
        a = self.adv(x)
        return v + a - a.mean(dim=-1, keepdim=True), hidden


class SeqReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((np.array(s, np.float32), int(a), float(r),
                         np.array(ns, np.float32), bool(d)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a, np.int64), np.array(r, np.float32),
                np.array(ns), np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)


def build_stacked(fs):
    frames = list(fs)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:]).astype(np.float32)


def build_seq(hist, seq_len, dim):
    h = list(hist)
    if len(h) >= seq_len:
        return np.array(h[-seq_len:], np.float32)
    pad = [np.zeros(dim, np.float32)] * (seq_len - len(h))
    return np.array(pad + h, np.float32)


def heuristic_action(obs, fw_count, turn_dir):
    if obs[17] == 1:
        return (random.choice([0, 4]), 0, -turn_dir)
    seg = 15
    if fw_count < seg:
        return (2, fw_count + 1, turn_dir)
    elif fw_count < seg + 2:
        return (4 if turn_dir > 0 else 0, fw_count + 1, turn_dir)
    else:
        return (2, 0, -turn_dir)


def shape_reward(obs, prev_obs, action_idx, env_reward, action_hist):
    shaped = env_reward
    if len(action_hist) >= 6 and all(a != 2 for a in list(action_hist)[-6:]):
        shaped -= 3.0
    if obs[17] == 1:
        shaped -= 10.0
    return shaped


def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    online = DuelingDRQN().to(device)
    target = DuelingDRQN().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr)
    buffer = SeqReplayBuffer(args.buffer)

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps,
                 wall_obstacles=True, difficulty=0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_avg = -float('inf')
    best_raw = -float('inf')
    all_rewards = []
    total_steps = 0
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  D3QN v2 + GRU | episodes={args.episodes} | device={device}")
    print(f"  NO CURRICULUM | Minimal shaping | walls=True always")
    print(f"{'='*70}")

    for ep in range(1, args.episodes + 1):
        diff = random.choice([0, 0, 2, 2, 3, 3, 3])
        env.difficulty = diff
        env.box_blink_enabled = diff >= 2
        env.box_move_enabled = diff >= 3

        obs = env.reset()
        frame_stack = deque(maxlen=STACK_SIZE)
        frame_stack.append(obs.astype(np.float32))
        prev_action = np.zeros(ACTION_DIM, np.float32)
        action_hist = deque(maxlen=20)
        history = deque(maxlen=args.seq_len)
        aug = np.concatenate([build_stacked(frame_stack), prev_action])
        history.append(aug)

        hidden = None
        done = False
        ep_raw = 0.0
        fw_count, turn_dir = 0, 1
        zero_run = 0

        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-total_steps / args.eps_decay)

        while not done:
            obs_seq = build_seq(history, args.seq_len, INPUT_DIM)
            sensor_sum = float(np.sum(obs[:17]))

            if sensor_sum == 0:
                zero_run += 1
                if zero_run >= 5:
                    hidden = None
                if random.random() < 0.15:
                    action_idx = random.randrange(ACTION_DIM)
                else:
                    action_idx, fw_count, turn_dir = heuristic_action(obs, fw_count, turn_dir)
            elif random.random() < eps:
                zero_run = 0
                action_idx = random.randrange(ACTION_DIM)
            else:
                zero_run = 0
                s = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q, hidden = online(s, hidden)
                action_idx = int(q[:, -1, :].argmax(dim=1).item())

            action_str = ACTIONS[action_idx]
            next_obs, env_reward, done = env.step(action_str, render=False)
            next_obs = next_obs.astype(np.float32)
            shaped = shape_reward(next_obs, obs, action_idx, env_reward, action_hist)

            fs_n = deque(frame_stack, maxlen=STACK_SIZE)
            fs_n.append(next_obs)
            pa_new = np.zeros(ACTION_DIM, np.float32)
            pa_new[action_idx] = 1.0
            aug_n = np.concatenate([build_stacked(fs_n), pa_new])
            hist_n = deque(history, maxlen=args.seq_len)
            hist_n.append(aug_n)
            next_obs_seq = build_seq(hist_n, args.seq_len, INPUT_DIM)

            buffer.push(obs_seq, action_idx, shaped, next_obs_seq, done)

            if len(buffer) >= args.batch:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(args.batch)
                S = torch.as_tensor(s_b, dtype=torch.float32, device=device)
                A = torch.as_tensor(a_b, dtype=torch.long, device=device)
                R = torch.as_tensor(r_b, dtype=torch.float32, device=device)
                NS = torch.as_tensor(ns_b, dtype=torch.float32, device=device)
                D = torch.as_tensor(d_b, dtype=torch.float32, device=device)

                q_all, _ = online(S)
                q_sa = q_all[:, -1, :].gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq_online, _ = online(NS)
                    na = nq_online[:, -1, :].argmax(dim=1, keepdim=True)
                    nq_target, _ = target(NS)
                    nq = nq_target[:, -1, :].gather(1, na).squeeze(1)
                    tgt = R + args.gamma * nq * (1 - D)

                loss = F.smooth_l1_loss(q_sa, tgt)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()

            total_steps += 1
            if total_steps % args.target_update == 0:
                target.load_state_dict(online.state_dict())

            obs = next_obs
            frame_stack = fs_n
            history = hist_n
            prev_action = pa_new
            action_hist.append(action_idx)
            ep_raw += env_reward

        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-50:]))
        if avg50 > best_avg:
            best_avg = avg50
            torch.save({"online": online.state_dict()}, args.save)
            tag_avg = " avg50*"
        else:
            tag_avg = ""

        if ep_raw > best_raw:
            best_raw = ep_raw
            torch.save({"online": online.state_dict()}, args.save_raw)
            tag_raw = " raw*"
        else:
            tag_raw = ""

        if ep % args.checkpoint_every == 0:
            torch.save({"online": online.state_dict()},
                       os.path.join(args.checkpoint_dir, f"d3qn_v2_ep{ep}.pth"))

        if ep <= 300 or ep % 200 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} best_raw={best_raw:+8.1f} | "
                  f"avg50={avg50:+8.1f} best_avg50={best_avg:+8.1f} | "
                  f"eps={eps:.3f} d={diff} | {elapsed:.1f}m eta={eta:.1f}m{tag_avg}{tag_raw}")

    print(f"\nDone. Best avg50={best_avg:+.1f} -> {args.save}")
    print(f"Done. Best raw={best_raw:+.1f} -> {args.save_raw}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=75000)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=500000)
    p.add_argument("--target_update", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--save_raw", type=str, default="weights_best_raw.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_d3qn_v2")
    p.add_argument("--checkpoint_every", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
