"""
train_final.py — Simple DDQN trainer (NO CURRICULUM, minimal shaping)

Root cause fix for Phase 3 failure:
  - NO curriculum learning (random spawns from step 1)
  - Minimal reward shaping (only anti-circling & anti-stuck)
  - Always wall_obstacles=True (matching Codabench)
  - Mixed difficulty (0/2/3) every episode
  - Heuristic-guided exploration during training

Usage:
  python launch.py --approach final --episodes 15000
  OR directly:
  cd phase4_submission && python train_final.py --episodes 15000 --device auto
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


class QNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ReplayBuffer:
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


def build_stacked(frame_stack):
    frames = list(frame_stack)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:], axis=0).astype(np.float32)


                                                                     
def heuristic_action(obs, fw_count, turn_dir):
    """Biased forward walk when no sensors active during training."""
    if obs[17] == 1:         
        return (random.choice([0, 4]), 0, -turn_dir)

    segment = 15
    if fw_count < segment:
        return (2, fw_count + 1, turn_dir)      
    elif fw_count < segment + 2:
        idx = 4 if turn_dir > 0 else 0             
        return (idx, fw_count + 1, turn_dir)
    else:
        return (2, 0, -turn_dir)                     


def shape_reward(obs, prev_obs, action_idx, env_reward, action_hist):
    """MINIMAL shaping — only penalize bad patterns, don't add fake positives."""
    shaped = env_reward

                                                 
    if len(action_hist) >= 6:
        recent = list(action_hist)[-6:]
        if all(a != 2 for a in recent):                                
            shaped -= 3.0

                                                                       
    if obs[17] == 1:
        shaped -= 10.0

    return shaped


def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    online = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = optim.Adam(online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

                                     
    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps,
                 wall_obstacles=True, difficulty=0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_avg = -float('inf')
    best_raw = -float('inf')
    all_rewards = []
    total_steps = 0
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  DDQN Final | episodes={args.episodes} | device={device}")
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

        done = False
        ep_raw = 0.0
        fw_count, turn_dir = 0, 1

                          
        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-total_steps / args.eps_decay)

        while not done:
            stacked = build_stacked(frame_stack)
            aug = np.concatenate([stacked, prev_action])
            sensor_sum = float(np.sum(obs[:17]))

                                    
            if sensor_sum == 0:
                                                                              
                if random.random() < 0.15:
                                                                          
                    action_idx = random.randrange(ACTION_DIM)
                else:
                    action_idx, fw_count, turn_dir = heuristic_action(obs, fw_count, turn_dir)
            elif random.random() < eps:
                                                         
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
            pa_new = np.zeros(ACTION_DIM, np.float32)
            pa_new[action_idx] = 1.0
            aug_n = np.concatenate([stacked_n, pa_new])

            buffer.push(aug, action_idx, shaped, aug_n, done)

                         
            if len(buffer) >= args.batch:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(args.batch)
                S = torch.as_tensor(s_b, dtype=torch.float32, device=device)
                A = torch.as_tensor(a_b, dtype=torch.long, device=device)
                R = torch.as_tensor(r_b, dtype=torch.float32, device=device)
                NS = torch.as_tensor(ns_b, dtype=torch.float32, device=device)
                D = torch.as_tensor(d_b, dtype=torch.float32, device=device)

                q_sa = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                                                                          
                    next_act = online(NS).argmax(dim=1, keepdim=True)
                    next_q = target(NS).gather(1, next_act).squeeze(1)
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
            ckpt = os.path.join(args.checkpoint_dir, f"final_ep{ep}.pth")
            torch.save({"online": online.state_dict()}, ckpt)

        if ep <= 300 or ep % 200 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} best_raw={best_raw:+8.1f} | "
                  f"avg50={avg50:+8.1f} best_avg50={best_avg:+8.1f} | "
                  f"eps={eps:.3f} d={diff} | {elapsed:.1f}m eta={eta:.1f}m{tag_avg}{tag_raw}")

    print(f"\nDone. Best avg50={best_avg:+.1f} -> {args.save}")
    print(f"Done. Best raw={best_raw:+.1f} -> {args.save_raw}")
    print(f"Total: {(time.time()-start)/60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=75000)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=500000)
    p.add_argument("--target_update", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--save_raw", type=str, default="weights_best_raw.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_final")
    p.add_argument("--checkpoint_every", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
