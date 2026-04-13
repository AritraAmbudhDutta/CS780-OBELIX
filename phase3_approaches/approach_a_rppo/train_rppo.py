"""Recurrent PPO trainer for OBELIX Phase 3."""

import argparse, os, sys, time, random, math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM, ACTION_DIM, STACK_SIZE = 18, 5, 4
INPUT_DIM = OBS_DIM * STACK_SIZE + ACTION_DIM  # 77
HIDDEN_DIM = 128


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.lstm(x, hidden)
        return self.actor(x), self.critic(x), hidden


def build_stacked(frame_stack):
    frames = list(frame_stack)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, dtype=np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:], axis=0).astype(np.float32)


def curriculum_reset(env, progress):
    """Spawn bot closer to box early in training, fully random after 60%."""
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
    face = math.degrees(math.atan2(box_y - by, box_x - bx))
    env.facing_angle = int(face + random.uniform(-25, 25))

    env._update_frames(show=False)
    env.get_feedback()
    env.update_reward()
    return env.sensor_feedback.copy()


def shape_reward(obs, prev_obs, action_idx, env_reward, action_hist):
    """Training-only reward shaping."""
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

    if len(action_hist) >= 6:
        recent = list(action_hist)[-6:]
        if all(a != 2 for a in recent):
            shaped -= 4.0

    if obs[17] == 1:
        shaped -= 30.0

    return shaped


def get_difficulty_schedule(progress):
    if progress < 0.20:
        return 0, False
    elif progress < 0.35:
        return 0, True
    elif progress < 0.50:
        return 2, random.random() < 0.5
    elif progress < 0.65:
        return 3, False
    else:
        d = random.choice([0, 2, 3, 3, 3])
        w = random.random() < 0.5
        return d, w


def train(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RecurrentActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=args.max_steps,
                 difficulty=0, wall_obstacles=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or '.', exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_reward_path = os.path.join(
        os.path.dirname(os.path.abspath(args.save)) or ".",
        "weights_best_reward.pth",
    )

    best_avg = -float('inf')
    best_raw = -float('inf')
    all_rewards = []
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  Recurrent PPO Training | episodes={args.episodes} | lr={args.lr}")
    print(f"  entropy_coeff={args.entropy_coeff} | clip={args.clip}")
    print(f"{'='*70}")

    for ep in range(1, args.episodes + 1):
        progress = ep / args.episodes

        diff, walls = get_difficulty_schedule(progress)
        env.difficulty = diff
        env.box_blink_enabled = diff >= 2
        env.box_move_enabled = diff >= 3
        env.wall_obstacles = walls

        obs = curriculum_reset(env, progress)

        frame_stack = deque(maxlen=STACK_SIZE)
        frame_stack.append(obs.astype(np.float32))
        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        action_hist = deque(maxlen=20)
        hidden = None

        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []
        ep_raw = 0.0

        done = False
        while not done:
            stacked = build_stacked(frame_stack)
            aug = np.concatenate([stacked, prev_action])

            x = torch.as_tensor(aug, dtype=torch.float32, device=device).view(1, 1, INPUT_DIM)
            with torch.no_grad():
                logits, value, hidden = model(x, hidden)
                logits_2d = logits[:, -1, :]
                value_2d = value[:, -1, :]

            dist = Categorical(logits=logits_2d)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_idx = action.item()
            action_str = ACTIONS[action_idx]

            next_obs, env_reward, done = env.step(action_str, render=False)
            next_obs = next_obs.astype(np.float32)

            shaped = shape_reward(next_obs, obs, action_idx, env_reward, action_hist)

            obs_list.append(aug.copy())
            act_list.append(action_idx)
            logp_list.append(log_prob.item())
            val_list.append(value_2d.item())
            rew_list.append(shaped)
            done_list.append(done)

            frame_stack.append(next_obs)
            prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
            prev_action[action_idx] = 1.0
            action_hist.append(action_idx)

            obs = next_obs
            ep_raw += env_reward

        T = len(rew_list)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        last_val = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_val
            else:
                next_val = val_list[t + 1]
            delta = rew_list[t] + args.gamma * next_val * (1 - done_list[t]) - val_list[t]
            gae = delta + args.gamma * args.gae_lambda * (1 - done_list[t]) * gae
            advantages[t] = gae
            returns[t] = gae + val_list[t]

        obs_t = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_list, dtype=torch.long, device=device)
        old_logp = torch.as_tensor(logp_list, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        adv_std = adv_t.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_std + 1e-8)
        else:
            adv_t = torch.zeros_like(adv_t)

        for _ in range(args.ppo_epochs):
            seq = obs_t.unsqueeze(0)
            logits_seq, val_seq, _ = model(seq)
            logits_seq = logits_seq.squeeze(0)
            val_seq = val_seq.squeeze(0).squeeze(-1)

            if not torch.isfinite(logits_seq).all() or not torch.isfinite(val_seq).all():
                break

            dist = Categorical(logits=logits_seq)
            new_logp = dist.log_prob(act_t)
            entropy = dist.entropy()

            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(val_seq, ret_t)
            entropy_loss = -entropy.mean()

            loss = policy_loss + args.value_coeff * value_loss + args.entropy_coeff * entropy_loss

            if not torch.isfinite(loss):
                break

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-50:]))

        if avg50 > best_avg:
            best_avg = avg50
            torch.save({"actor_critic": model.state_dict()}, args.save)
            tag = " <-- BEST"
        else:
            tag = ""

        raw_tag = ""
        if ep_raw > best_raw:
            best_raw = ep_raw
            torch.save({"actor_critic": model.state_dict()}, best_reward_path)
            raw_tag = " <-- BEST_RAW"

        if ep % args.checkpoint_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"rppo_ep{ep}.pth")
            torch.save({"actor_critic": model.state_dict()}, ckpt)

        if ep <= 200 or ep % 100 == 0:
            elapsed = (time.time() - start) / 60
            eta = elapsed / ep * (args.episodes - ep)
            print(f"Ep {ep:6d} | raw={ep_raw:+8.1f} avg50={avg50:+8.1f} best={best_avg:+8.1f} | "
                f"d={diff} w={int(walls)} | {elapsed:.1f}m eta={eta:.1f}m{tag}{raw_tag}")

    print(f"\nDone. Best avg50={best_avg:+.1f}  Saved: {args.save}")
    print(f"Best raw episode reward={best_raw:+.1f}  Saved: {best_reward_path}")
    print(f"Total: {(time.time()-start)/60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--entropy_coeff", type=float, default=0.02)
    p.add_argument("--value_coeff", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_rppo")
    p.add_argument("--checkpoint_every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
