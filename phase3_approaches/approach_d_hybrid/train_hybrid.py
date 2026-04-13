"""
train_hybrid.py — Hybrid Heuristic + DQN trainer for OBELIX (Phase 3)

Strategy:
  - Heuristic handles SEARCH phase (no sensors active → biased forward walk)
  - DQN learns APPROACH + PUSH behavior (when sensors detect the box)
  - Fastest to train: only learns near-box behavior, heuristic handles exploration

Key features:
  1) Simple MLP DQN (no recurrence needed — heuristic handles temporal aspects)
  2) Training uses curriculum (start near box → progressively farther)
  3) Dense reward shaping for approach and push
  4) Only transitions with sensor activity are stored in replay buffer

Usage:
  cd phase3_approaches/approach_d_hybrid
  python train_hybrid.py --episodes 5000 --device auto
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
HIDDEN_DIM = 64


# ═══════════════ Model ═══════════════
class SimpleQNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ═══════════════ Replay Buffer ═══════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((np.array(s, dtype=np.float32), int(a), float(r),
                         np.array(ns, dtype=np.float32), bool(d)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a, dtype=np.int64), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ═══════════════ Helpers ═══════════════
def build_stacked(frame_stack):
    frames = list(frame_stack)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, dtype=np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:], axis=0).astype(np.float32)


def curriculum_reset(env, progress):
    obs = env.reset()
    if progress > 0.55:
        return obs
    box_x, box_y = env.box_center_x, env.box_center_y
    margin = env.bot_radius + 15
    if progress < 0.15:
        dist = random.randint(15, 35)    # Very close
    elif progress < 0.30:
        dist = random.randint(30, 80)
    elif progress < 0.40:
        dist = random.randint(60, 140)
    else:
        dist = random.randint(100, 200)
    angle = random.uniform(0, 360)
    bx = int(box_x + dist * math.cos(math.radians(angle)))
    by = int(box_y + dist * math.sin(math.radians(angle)))
    env.bot_center_x = max(margin, min(env.frame_size[1] - margin, bx))
    env.bot_center_y = max(margin, min(env.frame_size[0] - margin, by))
    env.facing_angle = int(math.degrees(math.atan2(box_y - env.bot_center_y,
                                                     box_x - env.bot_center_x)) + random.uniform(-20, 20))
    env._update_frames(show=False)
    env.get_feedback()
    env.update_reward()
    return env.sensor_feedback.copy()


def shape_reward(obs, prev_obs, action_idx, env_reward, action_hist):
    shaped = env_reward
    s = float(np.sum(obs[:17]))
    ps = float(np.sum(prev_obs[:17]))
    shaped += 3.0 * (s - ps)
    if obs[16] == 1 and action_idx == 2:
        shaped += 10.0
    if s > 0 and action_idx == 2:
        shaped += 4.0
    if s == 0:
        shaped += 12.0
        if action_idx == 2:
            shaped += 2.0
    if len(action_hist) >= 6 and all(a != 2 for a in list(action_hist)[-6:]):
        shaped -= 5.0
    if obs[17] == 1:
        shaped -= 30.0
    return shaped


def heuristic_search(obs, step, rng, fw_count, turn_dir):
    """Biased random walk for exploration (used during training too)."""
    if obs[17] == 1:
        return (random.choice([0, 4]), 0, -turn_dir)

    segment = 12 + (step // 150) * 4
    if fw_count < segment:
        return (2, fw_count + 1, turn_dir)  # FW
    else:
        return (4 if turn_dir > 0 else 0, 0, -turn_dir)  # R45 or L45


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


# ═══════════════ Training Loop ═══════════════
def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps)

    online = SimpleQNet().to(device)
    target = SimpleQNet().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    eps = args.eps_start
    best_avg = -float('inf')
    best_raw = -float('inf')
    all_rewards = []
    total_steps = 0
    start = time.time()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or '.', exist_ok=True)
    best_reward_path = os.path.join(
        os.path.dirname(os.path.abspath(args.save)) or ".",
        "weights_best_reward.pth",
    )

    print(f"\n{'='*70}")
    print(f"  Hybrid Heuristic + DQN | episodes={args.episodes}")
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

        done = False
        ep_raw = 0.0
        fw_count = 0
        turn_dir = 1

        while not done:
            stacked = build_stacked(frame_stack)
            aug = np.concatenate([stacked, prev_action])

            sensor_sum = float(np.sum(obs[:17]))

            if sensor_sum == 0:
                # Heuristic search
                action_idx, fw_count, turn_dir = heuristic_search(obs, total_steps, None, fw_count, turn_dir)
                # Small chance of using NN even during search (for learning)
                if random.random() < 0.1 and len(buffer) > args.batch:
                    x = torch.as_tensor(aug, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        q = online(x)
                        action_idx = int(q.argmax(dim=1).item())
            else:
                # NN-guided with epsilon-greedy
                eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-total_steps / args.eps_decay)
                if random.random() < eps:
                    action_idx = random.randrange(ACTION_DIM)
                else:
                    x = torch.as_tensor(aug, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        q = online(x)
                        action_idx = int(q.argmax(dim=1).item())

            action_str = ACTIONS[action_idx]
            next_obs, env_reward, done = env.step(action_str, render=False)
            next_obs = next_obs.astype(np.float32)
            shaped = shape_reward(next_obs, obs, action_idx, env_reward, action_hist)

            frame_stack_n = deque(frame_stack, maxlen=STACK_SIZE)
            frame_stack_n.append(next_obs)
            stacked_n = build_stacked(frame_stack_n)
            pa_new = np.zeros(ACTION_DIM, dtype=np.float32)
            pa_new[action_idx] = 1.0
            aug_n = np.concatenate([stacked_n, pa_new])

            # Store ALL transitions (both heuristic and NN)
            buffer.push(aug, action_idx, shaped, aug_n, done)

            # Learn
            if len(buffer) >= args.batch:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(args.batch)
                S = torch.as_tensor(s_b, dtype=torch.float32, device=device)
                A = torch.as_tensor(a_b, dtype=torch.long, device=device)
                R = torch.as_tensor(r_b, dtype=torch.float32, device=device)
                NS = torch.as_tensor(ns_b, dtype=torch.float32, device=device)
                D = torch.as_tensor(d_b, dtype=torch.float32, device=device)

                q_sa = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a = online(NS).argmax(dim=1, keepdim=True)
                    next_q = target(NS).gather(1, next_a).squeeze(1)
                    tgt = R + args.gamma * next_q * (1 - D)

                loss = F.smooth_l1_loss(q_sa, tgt)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()

            total_steps += 1
            if total_steps % args.target_update == 0:
                target.load_state_dict(online.state_dict())

            obs = next_obs
            frame_stack = frame_stack_n
            prev_action = pa_new
            action_hist.append(action_idx)
            ep_raw += env_reward

        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-50:]))
        if avg50 > best_avg:
            best_avg = avg50
            torch.save({"online": online.state_dict()}, args.save)
            tag = " <-- BEST"
        else:
            tag = ""

        raw_tag = ""
        if ep_raw > best_raw:
            best_raw = ep_raw
            torch.save({"online": online.state_dict()}, best_reward_path)
            raw_tag = " <-- BEST_RAW"

        if ep % args.checkpoint_every == 0:
            torch.save({"online": online.state_dict()},
                       os.path.join(args.checkpoint_dir, f"hybrid_ep{ep}.pth"))

        if ep <= 200 or ep % 100 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} avg50={avg50:+8.1f} best={best_avg:+8.1f} | "
                f"d={diff} w={int(walls)} | {elapsed:.1f}m eta={eta:.1f}m{tag}{raw_tag}")

    print(f"\nDone. Best avg50={best_avg:+.1f}  Saved: {args.save}")
    print(f"Best raw episode reward={best_raw:+.1f}  Saved: {best_reward_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--eps_start", type=float, default=0.5)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=100000)
    p.add_argument("--target_update", type=int, default=1000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_hybrid")
    p.add_argument("--checkpoint_every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
