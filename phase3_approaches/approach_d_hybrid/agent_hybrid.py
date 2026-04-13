"""
agent_hybrid.py — OBELIX submission agent (Hybrid Heuristic + DQN)

Submit this file (renamed to agent.py) + weights.pth to Codabench.

Strategy:
  - When NO sensors are active: use a heuristic search pattern (biased random walk)
  - When sensors ARE active:  use a trained DQN to approach & push the box
  - When stuck: immediate turn-away escape

Architecture (DQN for near-box decisions):
  Linear(77→64) → ReLU → Linear(64→64) → ReLU → Linear(64→5)

Input: 4-frame stack (18×4=72) + previous action one-hot (5) = 77
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
OBS_DIM = 18
ACTION_DIM = 5
HIDDEN_DIM = 64
STACK_SIZE = 4
INPUT_DIM = OBS_DIM * STACK_SIZE + ACTION_DIM  # 77


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


# ── Global inference state ──────────────────────────────────────────
_MODEL = None
_LAST_RNG_ID = None
_FRAME_STACK = deque(maxlen=STACK_SIZE)
_PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
_STEP = 0
_FW_COUNT = 0
_TURN_DIR = 1  # 1=right, -1=left


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    model = SimpleQNet()
    sd = torch.load(wpath, map_location="cpu")
    if "online" in sd:
        model.load_state_dict(sd["online"])
    else:
        model.load_state_dict(sd)
    model.eval()
    _MODEL = model


def _build_stacked(frame_stack):
    frames = list(frame_stack)
    if len(frames) < STACK_SIZE:
        pad = [np.zeros(OBS_DIM, dtype=np.float32)] * (STACK_SIZE - len(frames))
        frames = pad + frames
    return np.concatenate(frames[-STACK_SIZE:], axis=0).astype(np.float32)


def _heuristic_search(obs, rng):
    """Biased random walk / lawnmower pattern when no sensors active."""
    global _FW_COUNT, _TURN_DIR

    # If stuck, turn away
    if obs[17] == 1:
        _FW_COUNT = 0
        _TURN_DIR *= -1
        return "L45" if _TURN_DIR < 0 else "R45"

    # Lawnmower pattern: go forward for N steps, then turn
    segment = 15 + (_STEP // 200) * 5  # Increasing segment length
    if _FW_COUNT < segment:
        _FW_COUNT += 1
        return "FW"
    else:
        _FW_COUNT = 0
        _TURN_DIR *= -1  # Alternate turn direction
        return "R45" if _TURN_DIR > 0 else "L45"


def _heuristic_with_sensors(obs):
    """Quick reactive heuristic when sensors detect the box."""
    # IR sensor active → go forward (aligned with box)
    if obs[16] == 1:
        return "FW"

    # Forward near sensors active → go forward
    if any(obs[5:12:2]):  # near bits of forward sensors
        return "FW"

    # Forward far sensors active → go forward
    if any(obs[4:12:2]):  # far bits of forward sensors
        return "FW"

    # Left sensors active → turn left
    if any(obs[0:4]):
        if obs[2] or obs[3]:  # near-left-front
            return "L22"
        return "L45"

    # Right sensors active → turn right
    if any(obs[12:16]):
        if obs[12] or obs[13]:  # near-right-front
            return "R22"
        return "R45"

    return None  # No clear heuristic → use NN


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _FRAME_STACK, _PREV_ACTION, _STEP, _FW_COUNT, _TURN_DIR

    _load_once()

    rid = id(rng)
    if _LAST_RNG_ID is None or rid != _LAST_RNG_ID:
        _FRAME_STACK.clear()
        _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
        _STEP = 0
        _FW_COUNT = 0
        _TURN_DIR = 1
        _LAST_RNG_ID = rid

    _STEP += 1
    sensor_sum = float(np.sum(obs[:17]))

    # ── Stuck escape ──
    if obs[17] == 1:
        act = "L45" if rng.random() < 0.5 else "R45"
        _update_prev(act)
        return act

    # ── No sensors → heuristic search ──
    if sensor_sum == 0:
        act = _heuristic_search(obs, rng)
        _update_prev(act)
        return act

    # ── Sensors active → try heuristic first, then NN ──
    h_act = _heuristic_with_sensors(obs)
    if h_act is not None:
        # Use NN to potentially override heuristic (blend)
        _FRAME_STACK.append(obs.astype(np.float32))
        stacked = _build_stacked(_FRAME_STACK)
        augmented = np.concatenate([stacked, _PREV_ACTION])
        x = torch.as_tensor(augmented, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = _MODEL(x)
            nn_idx = int(q.argmax(dim=1).item())
            nn_q_max = float(q.max())
            nn_q_heuristic = float(q[0, ACTIONS.index(h_act)])

        # If NN is confident and disagrees with heuristic, use NN
        if nn_q_max > nn_q_heuristic + 2.0:
            act = ACTIONS[nn_idx]
        else:
            act = h_act
    else:
        _FRAME_STACK.append(obs.astype(np.float32))
        stacked = _build_stacked(_FRAME_STACK)
        augmented = np.concatenate([stacked, _PREV_ACTION])
        x = torch.as_tensor(augmented, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = _MODEL(x)
            act = ACTIONS[int(q.argmax(dim=1).item())]

    _update_prev(act)
    return act


def _update_prev(action_str):
    global _PREV_ACTION, _FRAME_STACK
    idx = ACTIONS.index(action_str)
    _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
    _PREV_ACTION[idx] = 1.0
