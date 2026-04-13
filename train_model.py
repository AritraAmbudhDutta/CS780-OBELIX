"""
train_model.py  —  OBELIX DRQN trainer (anti-rotation shaping)

Key ideas implemented:
1) Sequence-based replay (DRQN) for partial observability.
2) Anti-circling reward shaping based on turn streak + stagnant observations.
3) Forward bonus when observation is all-zero (search during blink/invisible).
4) Double-DQN target selection with GRU model.

Submit with matching `agent.py` + generated `weights.pth`.
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACT_L45, ACT_L22, ACT_FW, ACT_R22, ACT_R45 = 0, 1, 2, 3, 4
TURN_ACTIONS = {ACT_L45, ACT_L22, ACT_R22, ACT_R45}
OBS_DIM = 18
ACTION_DIM = 5


class DRQNNet(nn.Module):
    def __init__(self, input_dim=OBS_DIM, hidden_dim=64, action_dim=ACTION_DIM):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, seq, hidden=None):
        x = F.relu(self.in_proj(seq))
        h, hidden_out = self.gru(x, hidden)
        q = self.out(h)
        return q, hidden_out


class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs_seq, action, reward, next_obs_seq, done):
        self.buf.append(
            (
                np.array(obs_seq, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_obs_seq, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DRQNAgent:
    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        seq_len=8,
        batch_size=128,
        buffer_capacity=300_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=400_000,
        target_update_every=1_000,
        grad_clip=10.0,
        use_amp=True,
        device="auto",
        input_dim=OBS_DIM,
    ):
        self.gamma = gamma
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_every = target_update_every
        self.grad_clip = grad_clip
        self.steps_done = 0

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested --device cuda but CUDA is not available.")
            self.device = torch.device("cuda")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.online = DRQNNet(input_dim=input_dim).to(self.device)
        self.target = DRQNNet(input_dim=input_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = SequenceReplayBuffer(buffer_capacity)

    def select_action(self, obs_seq):
        self.eps = self.eps_end + (1.0 - self.eps_end) * np.exp(
            -self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if random.random() < self.eps:
            return random.randrange(ACTION_DIM)

        s = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_seq, _ = self.online(s)
            q = q_seq[:, -1, :]
        return int(q.argmax(dim=1).item())

    def step(self, obs_seq, action, reward, next_obs_seq, done):
        self.buffer.push(obs_seq, action, reward, next_obs_seq, done)
        if len(self.buffer) >= self.batch_size:
            self.learn()
        if self.steps_done > 0 and self.steps_done % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

    def learn(self):
        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        S = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(a, dtype=torch.long, device=self.device)
        R = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        NS = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        D = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            q_seq, _ = self.online(S)
            q_last = q_seq[:, -1, :]
            q_sa = q_last.gather(1, A.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_online_seq, _ = self.online(NS)
                next_online_last = next_online_seq[:, -1, :]
                next_actions = next_online_last.argmax(dim=1, keepdim=True)

                next_target_seq, _ = self.target(NS)
                next_target_last = next_target_seq[:, -1, :]
                next_q = next_target_last.gather(1, next_actions).squeeze(1)

                target = R + self.gamma * next_q * (1.0 - D)

            loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
            self.optimizer.step()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.online.load_state_dict(sd)
        self.target.load_state_dict(sd)
        print(f"  [resumed from {path}]")


def _build_seq(history, seq_len, feature_dim):
    if len(history) >= seq_len:
        return np.array(list(history)[-seq_len:], dtype=np.float32)
    pad = [np.zeros(feature_dim, dtype=np.float32) for _ in range(seq_len - len(history))]
    return np.array(pad + list(history), dtype=np.float32)


def _build_stacked_obs(frame_history, stack_size):
    frames = list(frame_history)
    if len(frames) < stack_size:
        pad = [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(stack_size - len(frames))]
        frames = pad + frames
    else:
        frames = frames[-stack_size:]
    return np.concatenate(frames, axis=0).astype(np.float32)


def train(args):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    stacked_obs_dim = OBS_DIM * args.stack_size

    agent = DRQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        seq_len=args.seq_len,
        batch_size=args.batch,
        buffer_capacity=args.buffer,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_update_every=args.target_update_every,
        use_amp=not args.no_amp,
        device=args.device,
        input_dim=stacked_obs_dim,
    )

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)

    TURN_STREAK_PENALTY = args.turn_penalty
    SAME_OBS_TURN_PENALTY = args.same_obs_turn_penalty
    ZERO_OBS_FW_BONUS = args.zero_obs_fw_bonus

    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_reward = -float("inf")
    all_rewards = []

    print()
    print("=" * 78)
    print("  OBELIX DRQN Training (anti-rotation + sequence replay)")
    print(f"  episodes={args.episodes}  difficulty={args.difficulty}  walls={args.wall_obstacles}")
    print(
        f"  device={agent.device}  amp={agent.use_amp}  seq_len={args.seq_len}  stack={args.stack_size}  batch={args.batch}"
    )
    print(f"  save={args.save}")
    print(f"  checkpoint_dir={checkpoint_dir}")
    print("=" * 78)
    print(
        f"{'Ep':>7}  {'raw':>10}  {'shaped':>10}  {'avg50':>10}  {'best':>10}  {'eps':>6}  {'steps':>12}  {'elapsed':>9}  {'eta':>9}"
    )
    print("-" * 112)

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        obs = env.reset().astype(np.float32)
        frame_history = deque(maxlen=args.stack_size)
        frame_history.append(obs)
        stacked_obs = _build_stacked_obs(frame_history, args.stack_size)

        history = deque(maxlen=args.seq_len)
        history.append(stacked_obs)

        done = False
        ep_raw = 0.0
        ep_shaped = 0.0

        turn_streak = 0
        same_obs_count = 0

        while not done:
            obs_seq = _build_seq(history, args.seq_len, stacked_obs_dim)
            action_idx = agent.select_action(obs_seq)
            action_str = ACTIONS[action_idx]

            next_obs, env_reward, done = env.step(action_str, render=False)
            next_obs = next_obs.astype(np.float32)

            shaped_reward = float(env_reward)

            if action_idx in TURN_ACTIONS:
                turn_streak += 1
            else:
                turn_streak = 0

            if np.array_equal(next_obs, obs):
                same_obs_count += 1
            else:
                same_obs_count = 0

            if turn_streak >= 8 and same_obs_count >= 3:
                shaped_reward -= TURN_STREAK_PENALTY

            if turn_streak >= 12 and same_obs_count >= 5:
                shaped_reward -= SAME_OBS_TURN_PENALTY

            if float(obs.sum()) == 0.0 and action_idx == ACT_FW:
                shaped_reward += ZERO_OBS_FW_BONUS

            frame_history_next = deque(frame_history, maxlen=args.stack_size)
            frame_history_next.append(next_obs)
            stacked_next_obs = _build_stacked_obs(frame_history_next, args.stack_size)

            history_next = deque(history, maxlen=args.seq_len)
            history_next.append(stacked_next_obs)

            next_obs_seq = _build_seq(history_next, args.seq_len, stacked_obs_dim)
            agent.step(obs_seq, action_idx, shaped_reward, next_obs_seq, done)

            obs = next_obs
            frame_history = frame_history_next
            history = history_next
            ep_raw += float(env_reward)
            ep_shaped += shaped_reward

        all_rewards.append(ep_raw)
        avg50 = float(np.mean(all_rewards[-min(50, len(all_rewards)) :]))

        if avg50 > best_reward:
            best_reward = avg50
            agent.save(args.save)
            tag = "  <-- NEW BEST"
        else:
            tag = ""

        if ep % args.checkpoint_every == 0:
            base_name = os.path.splitext(os.path.basename(args.save))[0]
            ckpt = os.path.join(checkpoint_dir, f"{base_name}_ep{ep}.pth")
            agent.save(ckpt)
            print(f"  [checkpoint saved: {ckpt}]")

        if ep <= 1000 or ep % 500 == 0:
            elapsed_s = max(0.0, time.time() - start_time)
            avg_ep_s = elapsed_s / ep
            eta_s = avg_ep_s * (args.episodes - ep)
            elapsed_m = elapsed_s / 60.0
            eta_m = eta_s / 60.0
            print(
                f"{ep:7d}  {ep_raw:+10.1f}  {ep_shaped:+10.1f}  {avg50:+10.1f}  "
                f"{best_reward:+10.1f}  {agent.eps:6.3f}  {agent.steps_done:12,}  "
                f"{elapsed_m:8.1f}m  {eta_m:8.1f}m{tag}"
            )

    print()
    print("Training complete")
    print(f"Best avg50      : {best_reward:+.2f}")
    print(f"Weights saved   : {args.save}")
    print(f"Submit          : zip submission.zip agent.py {args.save}")
    total_s = max(0.0, time.time() - start_time)
    print(f"Total time      : {total_s / 60.0:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(
        description="OBELIX DRQN trainer with anti-circling shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--episodes", type=int, default=4_000)
    p.add_argument("--checkpoint_every", type=int, default=20)

    p.add_argument("--scaling_factor", type=int, default=5)
    p.add_argument("--arena_size", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1_000)
    p.add_argument("--wall_obstacles", action="store_true")
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--box_speed", type=int, default=2)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--buffer", type=int, default=300_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument(
        "--stack_size",
        type=int,
        default=4,
        help="Number of recent raw observations to concatenate as one stacked input.",
    )
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=200_000)
    p.add_argument("--target_update_every", type=int, default=1_000)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device: auto selects cuda if available else cpu.",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision training (AMP).",
    )

    p.add_argument("--turn_penalty", type=float, default=0.2)
    p.add_argument("--same_obs_turn_penalty", type=float, default=0.5)
    p.add_argument("--zero_obs_fw_bonus", type=float, default=1.5)

    p.add_argument("--save", type=str, default="weights.pth")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_drqn",
        help="Directory where periodic checkpoints are written.",
    )
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
