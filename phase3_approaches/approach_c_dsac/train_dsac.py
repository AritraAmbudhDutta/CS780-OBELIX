"""
train_dsac.py — Discrete SAC + GRU trainer for OBELIX (Phase 3)

Key features:
  1) Discrete Soft Actor-Critic (entropy-maximizing → best exploration)
  2) GRU-based policy and Q-networks for POMDP
  3) Auto-tuned temperature α
  4) Two Q-networks for stability (clipped double Q)
  5) Curriculum learning + dense reward shaping
  6) Multi-difficulty training schedule

Usage:
  cd phase3_approaches/approach_c_dsac
  python train_dsac.py --episodes 8000 --device auto
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


# ═══════════════ Networks ═══════════════
class SACPolicyNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.gru(x, hidden)
        return self.out(x), hidden

    def get_action(self, x, hidden=None):
        logits, hidden = self.forward(x, hidden)
        logits_2d = logits[:, -1, :]
        probs = F.softmax(logits_2d, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = torch.log(probs + 1e-8)
        return action, probs, log_prob, hidden


class SACQNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.gru(x, hidden)
        return self.out(x), hidden


# ═══════════════ Replay Buffer ═══════════════
class SeqReplayBuffer:
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


# ═══════════════ Common Helpers ═══════════════
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
    env.bot_center_x = max(margin, min(env.frame_size[1] - margin, bx))
    env.bot_center_y = max(margin, min(env.frame_size[0] - margin, by))
    env.facing_angle = int(math.degrees(math.atan2(box_y - env.bot_center_y,
                                                     box_x - env.bot_center_x)) + random.uniform(-25, 25))
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


# ═══════════════ Training Loop ═══════════════
def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps)

    policy_net = SACPolicyNet().to(device)
    q1 = SACQNet().to(device)
    q2 = SACQNet().to(device)
    q1_target = SACQNet().to(device)
    q2_target = SACQNet().to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    policy_opt = optim.Adam(policy_net.parameters(), lr=args.lr)
    q1_opt = optim.Adam(q1.parameters(), lr=args.lr)
    q2_opt = optim.Adam(q2.parameters(), lr=args.lr)

    # Auto-tuned temperature
    target_entropy = -0.5 * math.log(ACTION_DIM)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=args.lr)

    buffer = SeqReplayBuffer(args.buffer)

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
    print(f"  Discrete SAC + GRU | episodes={args.episodes}")
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

        hidden_p = None
        done = False
        ep_raw = 0.0

        while not done:
            obs_seq = build_seq(history, args.seq_len, INPUT_DIM)
            s_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, hidden_p = policy_net.get_action(s_t, hidden_p)
            action_idx = action.item()
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
            history_n = deque(history, maxlen=args.seq_len)
            history_n.append(aug_n)
            next_obs_seq = build_seq(history_n, args.seq_len, INPUT_DIM)

            buffer.push(obs_seq, action_idx, shaped, next_obs_seq, done)

            obs = next_obs
            frame_stack = frame_stack_n
            history = history_n
            prev_action = pa_new
            action_hist.append(action_idx)
            ep_raw += env_reward

        # ── SAC Update ──
        if len(buffer) >= args.batch:
            for _ in range(min(4, max(1, len(buffer) // 1000))):
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(args.batch)
                S = torch.as_tensor(s_b, dtype=torch.float32, device=device)
                A = torch.as_tensor(a_b, dtype=torch.long, device=device)
                R = torch.as_tensor(r_b, dtype=torch.float32, device=device)
                NS = torch.as_tensor(ns_b, dtype=torch.float32, device=device)
                D = torch.as_tensor(d_b, dtype=torch.float32, device=device)

                alpha = log_alpha.exp().detach()

                # Q-value targets
                with torch.no_grad():
                    next_logits, _ = policy_net(NS)
                    next_logits_2d = next_logits[:, -1, :]
                    next_probs = F.softmax(next_logits_2d, dim=-1)
                    next_log_probs = torch.log(next_probs + 1e-8)

                    q1t_all, _ = q1_target(NS)
                    q2t_all, _ = q2_target(NS)
                    q1t_2d = q1t_all[:, -1, :]
                    q2t_2d = q2t_all[:, -1, :]
                    next_q = torch.min(q1t_2d, q2t_2d)
                    next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
                    target = R + args.gamma * next_v * (1 - D)

                # Q-network updates
                q1_all, _ = q1(S)
                q2_all, _ = q2(S)
                q1_sa = q1_all[:, -1, :].gather(1, A.unsqueeze(1)).squeeze(1)
                q2_sa = q2_all[:, -1, :].gather(1, A.unsqueeze(1)).squeeze(1)

                q1_loss = F.mse_loss(q1_sa, target)
                q2_loss = F.mse_loss(q2_sa, target)

                q1_opt.zero_grad()
                q1_loss.backward()
                q1_opt.step()

                q2_opt.zero_grad()
                q2_loss.backward()
                q2_opt.step()

                # Policy update
                curr_logits, _ = policy_net(S)
                curr_logits_2d = curr_logits[:, -1, :]
                probs = F.softmax(curr_logits_2d, dim=-1)
                log_probs = torch.log(probs + 1e-8)

                with torch.no_grad():
                    q1_curr, _ = q1(S)
                    q2_curr, _ = q2(S)
                    q_min = torch.min(q1_curr[:, -1, :], q2_curr[:, -1, :])

                policy_loss = (probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()

                policy_opt.zero_grad()
                policy_loss.backward()
                policy_opt.step()

                # Temperature update
                alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy).mean()).mean()
                # Simplify: use entropy of current policy
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                alpha_loss = -(log_alpha.exp() * (entropy.detach() - target_entropy))

                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # Soft target update
                for tp, p in zip(q1_target.parameters(), q1.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for tp, p in zip(q2_target.parameters(), q2.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        # ── Logging ──
        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-50:]))
        if avg50 > best_avg:
            best_avg = avg50
            torch.save({"policy": policy_net.state_dict()}, args.save)
            tag = " <-- BEST"
        else:
            tag = ""

        raw_tag = ""
        if ep_raw > best_raw:
            best_raw = ep_raw
            torch.save({"policy": policy_net.state_dict()}, best_reward_path)
            raw_tag = " <-- BEST_RAW"

        if ep % args.checkpoint_every == 0:
            torch.save({"policy": policy_net.state_dict()},
                       os.path.join(args.checkpoint_dir, f"dsac_ep{ep}.pth"))

        if ep <= 200 or ep % 100 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            a_val = log_alpha.exp().item()
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} avg50={avg50:+8.1f} best={best_avg:+8.1f} | "
                f"α={a_val:.3f} d={diff} w={int(walls)} | {elapsed:.1f}m eta={eta:.1f}m{tag}{raw_tag}")

    print(f"\nDone. Best avg50={best_avg:+.1f}  Saved: {args.save}")
    print(f"Best raw episode reward={best_raw:+.1f}  Saved: {best_reward_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=75000)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_dsac")
    p.add_argument("--checkpoint_every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
