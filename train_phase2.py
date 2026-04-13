"""
train_phase2.py  —  OBELIX Phase 2: Blinking Box (difficulty=2)
================================================================

What this file does
-------------------
1.  Trains a Dueling Double-DQN agent on difficulty=2 (blinking box).
2.  Saves the BEST weights seen so far to  weights_phase2.pth  every time
    a new high score is achieved.
3.  Saves a periodic checkpoint every --checkpoint_every episodes.
4.  Prints one clear progress line per episode so you can watch it learn.
5.  At the end, writes  agent_phase2.py  (the submission file you zip and
    upload to Codabench — it already embeds the network definition and
    loads weights_phase2.pth automatically).

Files you submit to Codabench
------------------------------
    agent_phase2.py   +   weights_phase2.pth

Nothing else needs to be changed.  obelix.py / evaluate.py / etc. are
untouched — exactly as the professor provided.

The blinking-box problem
------------------------
When the box is invisible every sensor bit = 0.  A plain 18-bit agent
has zero gradient signal and just spins.  We fix this with:

  AUGMENTED STATE  (18 raw bits  →  22-dim input)
  ─────────────────────────────────────────────────
  [0..17]  raw sensor bits from env
  [18]     1 if box was visible on THIS step, else 0
  [19]     steps_since_box_last_seen / 50   (0 → 1)
  [20]     sin of angle robot→box when last seen
  [21]     cos of angle robot→box when last seen

  The network always has a "compass" toward the last known position even
  while the box is invisible.

The circling / orbiting problem
---------------------------------
The raw env reward fires for ANY sensor seeing the box, so a naive agent
learns to orbit the box (keeps sensors active, avoids the harder task of
going straight in).  We fix this with:

  1.  Potential-based distance shaping   F = γ·Φ(s') − Φ(s)
      Φ(s) = −0.05 × dist(robot, last_known_box)
      → every step that reduces distance gives positive shaped reward
      → decays when box is invisible so stale info isn't over-rewarded

  2.  Orbit penalty   −3 per step when OrbitDetector fires
      (constant distance + rotating angle = orbiting)

  3.  IR bonus  +15 when IR sensor fires (approach is well rewarded)

  4.  Forward-search bonus  +0.3 when box invisible and agent picks FW
      (discourages spinning in place during invisible phases)

  5.  Push shaping  +0.15 × (box moved closer to nearest wall)
      (guides the push-to-boundary phase once attached)

Architecture
------------
Dueling Double DQN with 3 shared hidden layers (128 units each).
Input dim = 22.  Output dim = 5 (one Q-value per action).

Usage
-----
# default: 4000 episodes, no walls
python train_phase2.py

# longer run with wall obstacles (matches Codabench evaluation)
python train_phase2.py --episodes 6000 --wall_obstacles

# resume from a previous checkpoint
python train_phase2.py --resume weights_phase2.pth
"""

import argparse
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from obelix import OBELIX

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
RAW_OBS_DIM = 18
EXTRA_DIM = 4
OBS_DIM = RAW_OBS_DIM + EXTRA_DIM  # 22
ACTION_DIM = 5
MAX_STEPS_SINCE_SEEN = 50  # normalisation cap


# ---------------------------------------------------------------------------
# Network — Dueling DQN
# ---------------------------------------------------------------------------


class DuelingNet(nn.Module):
    """
    Dueling DQN head.
    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

    Input : 22-dim augmented state
    Output: 5 Q-values (one per action)
    """

    def __init__(self, input_dim=OBS_DIM, output_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        h = self.shared(x)
        v = self.value_head(h)  # (B, 1)
        a = self.advantage_head(h)  # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append(
            (
                np.array(s, dtype=np.float32),
                int(a),
                float(r),
                np.array(ns, dtype=np.float32),
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


# ---------------------------------------------------------------------------
# Box memory — survives invisible phases
# ---------------------------------------------------------------------------


class BoxMemory:
    """
    Maintains the last-known direction to the box.
    Trainer calls update() every step with TRUE env positions.
    augment() appends 4 memory features to the raw 18-bit obs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_sin = 0.0  # sin of angle robot→box when last visible
        self.last_cos = 1.0  # cos of angle robot→box when last visible
        self.steps_since_seen = 0
        self.box_visible_now = False

    def update(self, rx, ry, bx, by, visible):
        """Call once per step with the true robot and box positions."""
        self.box_visible_now = visible
        if visible:
            angle = math.atan2(by - ry, bx - rx)
            self.last_sin = math.sin(angle)
            self.last_cos = math.cos(angle)
            self.steps_since_seen = 0
        else:
            self.steps_since_seen = min(self.steps_since_seen + 1, MAX_STEPS_SINCE_SEEN)

    def augment(self, raw_obs):
        """Return 22-dim augmented state from raw 18-dim obs."""
        extra = np.array(
            [
                float(self.box_visible_now),  # [18]
                self.steps_since_seen / MAX_STEPS_SINCE_SEEN,  # [19]
                self.last_sin,  # [20]
                self.last_cos,  # [21]
            ],
            dtype=np.float32,
        )
        return np.concatenate([raw_obs.astype(np.float32), extra])


# ---------------------------------------------------------------------------
# Orbit detector
# ---------------------------------------------------------------------------


class OrbitDetector:
    """
    Detects when the robot circles the box at a roughly constant distance.
    Fires when:
      - distance variance over last `window` steps is below dist_var_thresh
      - AND the robot-box angle has rotated more than angle_thresh_deg total
    Only updated while the box is visible (no point detecting orbit otherwise).
    """

    def __init__(self, window=20, dist_var_thresh=300.0, angle_thresh_deg=15.0):
        self.window = window
        self.dist_var_thresh = dist_var_thresh
        self.angle_thresh = math.radians(angle_thresh_deg)
        self._dists = deque(maxlen=window)
        self._angles = deque(maxlen=window)

    def update(self, rx, ry, bx, by):
        d = math.sqrt((rx - bx) ** 2 + (ry - by) ** 2)
        a = math.atan2(ry - by, rx - bx)
        self._dists.append(d)
        self._angles.append(a)

    def is_orbiting(self):
        if len(self._dists) < self.window:
            return False
        d_var = float(np.var(self._dists))
        angles_unwrapped = np.unwrap(list(self._angles))
        total_rotation = abs(float(angles_unwrapped[-1] - angles_unwrapped[0]))
        return d_var < self.dist_var_thresh and total_rotation > self.angle_thresh

    def reset(self):
        self._dists.clear()
        self._angles.clear()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _boundary_dist(bx, by, arena):
    """
    Distance of box centre from the nearest inner wall boundary (at 10 px).
    Smaller = closer to wall = closer to episode-ending success.
    """
    return float(min(bx - 10, (arena - 10) - bx, by - 10, (arena - 10) - by))


# ---------------------------------------------------------------------------
# Double Dueling DQN agent
# ---------------------------------------------------------------------------


class Phase2Agent:
    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=300_000,
        batch_size=128,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=400_000,
        target_hard_every=2_000,
        grad_clip=10.0,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_hard_every = target_hard_every
        self.grad_clip = grad_clip
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingNet(obs_dim, action_dim).to(self.device)
        self.target_net = DuelingNet(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    # ---- epsilon-greedy action selection ----------------------------------

    def select_action(self, aug_state):
        self.eps = self.eps_end + (1.0 - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if random.random() > self.eps:
            t = torch.as_tensor(
                aug_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                return int(self.policy_net(t).argmax(1).item())
        return random.randrange(ACTION_DIM)

    # ---- store transition and maybe learn ---------------------------------

    def step(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)
        if len(self.memory) >= self.batch_size:
            self._learn()
        # Hard-copy target network periodically
        if self.steps_done % self.target_hard_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---- Double DQN learning step -----------------------------------------

    def _learn(self):
        s, a, r, ns, d = self.memory.sample(self.batch_size)

        S = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(a, dtype=torch.long, device=self.device)
        R = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        NS = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        D = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        q_vals = self.policy_net(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # Double DQN: policy_net selects action, target_net evaluates it
        with torch.no_grad():
            best_actions = self.policy_net(NS).argmax(1, keepdim=True)
            next_q = self.target_net(NS).gather(1, best_actions).squeeze(1)
            target_q = R + self.gamma * next_q * (1.0 - D)

        loss = F.smooth_l1_loss(q_vals, target_q)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

    # ---- save / load ------------------------------------------------------

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)
        print(f"  [resumed from {path}]")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):

    # ---- environment ------------------------------------------------------
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=2,  # always difficulty=2 for Phase 2
        box_speed=args.box_speed,
    )
    arena = args.arena_size

    # ---- agent and helpers ------------------------------------------------
    agent = Phase2Agent(
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        eps_decay=args.eps_decay,
        eps_end=args.eps_end,
    )
    box_mem = BoxMemory()
    orbit = OrbitDetector()

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)

    # ---- reward shaping constants -----------------------------------------
    DIST_SCALE = 0.05  # Phi(s) = -DIST_SCALE * dist(robot, last_known_box)
    INVISIBLE_DECAY = 0.90  # decay Phi each step the box is invisible
    ORBIT_PENALTY = -3.0  # per step when orbit is detected
    IR_BONUS = 15.0  # extra bonus when IR sensor (bit 16) fires
    VIS_BONUS = 0.5  # small per-step bonus while box is visible
    SEARCH_FWD_BONUS = 0.3  # bonus for choosing FW while box is invisible
    PUSH_SCALE = 0.15  # bonus for moving attached box toward wall

    # ---- logging ----------------------------------------------------------
    best_reward = -float("inf")
    all_rewards = []

    # make sure the directory for the save file exists
    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  OBELIX  —  Phase 2 Training  (Blinking Box, difficulty=2)")
    print(f"  episodes      : {args.episodes}")
    print(f"  max_steps/ep  : {args.max_steps}")
    print(f"  arena_size    : {arena}   wall_obstacles: {args.wall_obstacles}")
    print(f"  device        : {agent.device}")
    print(f"  replay buffer : {args.buffer:,}   batch: {args.batch}")
    print(f"  eps_decay     : {args.eps_decay:,}   eps_end: {args.eps_end}")
    print(f"  weights saved : {args.save}")
    print("=" * 70)
    print()
    print(
        f"{'Ep':>7}  {'raw_reward':>11}  {'shaped':>10}  "
        f"{'avg50':>9}  {'best':>9}  {'eps':>6}  {'total_steps':>12}"
    )
    print("-" * 80)

    for episode in range(1, args.episodes + 1):
        # ---- reset --------------------------------------------------------
        raw_obs = env.reset()
        box_mem.reset()
        orbit.reset()

        # initialise memory with the true starting positions
        box_mem.update(
            env.bot_center_x,
            env.bot_center_y,
            env.box_center_x,
            env.box_center_y,
            env.box_visible,
        )
        aug_obs = box_mem.augment(raw_obs)

        # snapshot for potential shaping
        rx_prev = float(env.bot_center_x)
        ry_prev = float(env.bot_center_y)
        bx_last = float(env.box_center_x)  # last-known box position
        by_last = float(env.box_center_y)
        phi_prev = -DIST_SCALE * _dist(rx_prev, ry_prev, bx_last, by_last)
        vis_decay = 1.0  # multiplied by INVISIBLE_DECAY each invisible step

        episode_raw = 0.0
        episode_shaped = 0.0
        done = False
        attached = False  # have we attached to the box yet this episode?

        # ---- step loop ----------------------------------------------------
        while not done:
            action_idx = agent.select_action(aug_obs)
            action_str = ACTIONS[action_idx]

            raw_next, env_reward, done = env.step(action_str, render=False)

            # true env state (trainer has access; the submission agent does not)
            rx = float(env.bot_center_x)
            ry = float(env.bot_center_y)
            bx_true = float(env.box_center_x)
            by_true = float(env.box_center_y)
            visible = bool(env.box_visible)

            # update memory and build augmented next-obs
            box_mem.update(rx, ry, bx_true, by_true, visible)
            aug_next = box_mem.augment(raw_next)

            # ---- potential-based distance shaping -------------------------
            if visible:
                bx_last = bx_true
                by_last = by_true
                vis_decay = 1.0
            else:
                # bx_last / by_last keep their previous values (stale)
                vis_decay = vis_decay * INVISIBLE_DECAY

            phi_curr = -DIST_SCALE * _dist(rx, ry, bx_last, by_last) * vis_decay
            dist_shape = args.gamma * phi_curr - phi_prev
            phi_prev = phi_curr

            # ---- orbit penalty (only meaningful when box is visible) ------
            if visible:
                orbit.update(rx, ry, bx_true, by_true)
            orbit_shape = ORBIT_PENALTY if orbit.is_orbiting() else 0.0

            # ---- IR bonus -------------------------------------------------
            # raw_next[16] = 1 when the front IR sensor detects the box
            ir_shape = IR_BONUS if raw_next[16] else 0.0

            # ---- visible-proximity bonus ----------------------------------
            vis_shape = VIS_BONUS if visible else 0.0

            # ---- forward-search bonus (box invisible, agent picks FW) -----
            search_shape = (
                SEARCH_FWD_BONUS if (not visible and action_idx == 2) else 0.0
            )

            # ---- push-phase shaping (once attached) -----------------------
            push_shape = 0.0
            if env.enable_push:
                if not attached:
                    attached = True
                # positive when box moves closer to the nearest wall
                bd_before = _boundary_dist(bx_last, by_last, arena)
                bd_after = _boundary_dist(bx_true, by_true, arena)
                push_shape = PUSH_SCALE * (bd_before - bd_after)

            # ---- total shaped reward --------------------------------------
            shaped_reward = (
                env_reward
                + dist_shape
                + orbit_shape
                + ir_shape
                + vis_shape
                + search_shape
                + push_shape
            )

            # ---- store and learn ------------------------------------------
            agent.step(aug_obs, action_idx, shaped_reward, aug_next, done)

            # ---- advance --------------------------------------------------
            aug_obs = aug_next
            rx_prev, ry_prev = rx, ry
            episode_raw += env_reward  # raw env reward for leaderboard
            episode_shaped += shaped_reward

        # ---- episode bookkeeping ------------------------------------------
        all_rewards.append(episode_raw)
        window = min(50, len(all_rewards))
        avg_50 = float(np.mean(all_rewards[-window:]))

        # save best weights
        if episode_raw > best_reward:
            best_reward = episode_raw
            agent.save(args.save)
            star = "  <-- NEW BEST"
        else:
            star = ""

        # periodic named checkpoint
        if episode % args.checkpoint_every == 0:
            ckpt_path = args.save.replace(".pth", f"_ep{episode}.pth")
            agent.save(ckpt_path)
            print(f"  [checkpoint saved: {ckpt_path}]")

        # progress line every episode
        print(
            f"{episode:7d}  "
            f"{episode_raw:+11.1f}  "
            f"{episode_shaped:+10.1f}  "
            f"{avg_50:+9.1f}  "
            f"{best_reward:+9.1f}  "
            f"{agent.eps:6.3f}  "
            f"{agent.steps_done:12,}"
            f"{star}"
        )

    print()
    print(f"  Training complete.")
    print(f"  Best raw reward : {best_reward:+.2f}")
    print(f"  Weights saved to: {args.save}")
    print()
    print("  To submit to Codabench:")
    print(f"    zip submission.zip agent.py {args.save}")
    print()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="OBELIX Phase 2 — Blinking Box — Dueling DDQN with memory augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training length
    p.add_argument(
        "--episodes", type=int, default=4_000, help="Total number of training episodes."
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Save a named checkpoint every N episodes.",
    )

    # Environment settings
    p.add_argument("--scaling_factor", type=int, default=5)
    p.add_argument("--arena_size", type=int, default=500)
    p.add_argument(
        "--max_steps", type=int, default=1_000, help="Max steps per episode."
    )
    p.add_argument(
        "--wall_obstacles",
        action="store_true",
        help="Add wall obstacles (recommended for Codabench submission).",
    )
    p.add_argument(
        "--box_speed",
        type=int,
        default=2,
        help="Unused at difficulty=2, kept for API compatibility.",
    )

    # Agent hyperparameters
    p.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument(
        "--buffer", type=int, default=300_000, help="Replay buffer capacity."
    )
    p.add_argument("--batch", type=int, default=128, help="Batch size for learning.")
    p.add_argument(
        "--eps_decay",
        type=int,
        default=400_000,
        help="Epsilon decay (in total env steps).",
    )
    p.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon value.")

    # I/O
    p.add_argument(
        "--save",
        type=str,
        default="weights.pth",
        help="Path to save the best weights (submit this as weights.pth).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing weights to resume training from.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
