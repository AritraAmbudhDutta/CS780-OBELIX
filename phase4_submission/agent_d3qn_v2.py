"""
agent_d3qn_v2.py — OBELIX submission agent (Fixed D3QN + GRU + Heuristic Fallback)

Fix vs Phase 3 version:
  - Heuristic fallback when sensors=0 (don't rely on NN for search)
  - Hidden state reset after 5+ zero-sensor steps
  - Deterministic stuck escape

Architecture: Linear(77→128) → ReLU → GRU(128→128) → Dueling(V+A)
Policy: heuristic when sensors=0, greedy Q when sensors active.
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
HIDDEN_DIM = 128
STACK_SIZE = 4
INPUT_DIM = OBS_DIM * STACK_SIZE + ACTION_DIM


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


_MODEL = None
_HIDDEN = None
_LAST_RNG_ID = None
_FRAME_STACK = deque(maxlen=STACK_SIZE)
_PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
_STEP = 0
_ZERO_RUN = 0
_FW_COUNT = 0
_TURN_DIR = 1
_STUCK_ESCAPE = 0


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    model = DuelingDRQN()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd["online"] if "online" in sd else sd)
    model.eval()
    _MODEL = model


def _build_stacked(fs):
    frames = list(fs)
    if len(frames) < STACK_SIZE:
        frames = [np.zeros(OBS_DIM, np.float32)] * (STACK_SIZE - len(frames)) + frames
    return np.concatenate(frames[-STACK_SIZE:]).astype(np.float32)


def _heuristic_search():
    global _FW_COUNT, _TURN_DIR
    seg = 15
    if _FW_COUNT < seg:
        _FW_COUNT += 1
        return "FW"
    elif _FW_COUNT < seg + 2:
        _FW_COUNT += 1
        return "R45" if _TURN_DIR > 0 else "L45"
    else:
        _FW_COUNT = 0
        _TURN_DIR *= -1
        return "FW"


def _heuristic_sensors(obs):
    if obs[16] == 1:
        return "FW"
    if any(obs[5:12:2]) or any(obs[4:12:2]):
        return "FW"
    left = sum(obs[0:4])
    right = sum(obs[12:16])
    if left > 0 and right == 0:
        return "L22" if obs[2] or obs[3] else "L45"
    if right > 0 and left == 0:
        return "R22" if obs[12] or obs[13] else "R45"
    if left > 0 and right > 0:
        return "FW"
    return None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN, _LAST_RNG_ID, _FRAME_STACK, _PREV_ACTION
    global _STEP, _ZERO_RUN, _FW_COUNT, _TURN_DIR, _STUCK_ESCAPE

    _load_once()

    rid = id(rng)
    if _LAST_RNG_ID is None or rid != _LAST_RNG_ID:
        _HIDDEN = None
        _FRAME_STACK.clear()
        _PREV_ACTION = np.zeros(ACTION_DIM, np.float32)
        _STEP = _ZERO_RUN = _FW_COUNT = _STUCK_ESCAPE = 0
        _TURN_DIR = 1
        _LAST_RNG_ID = rid

    _STEP += 1
    sensor_sum = float(np.sum(obs[:17]))

                  
    if obs[17] == 1:
        _STUCK_ESCAPE = 4
        _FW_COUNT = 0
    if _STUCK_ESCAPE > 0:
        _STUCK_ESCAPE -= 1
        act = ("L45" if _TURN_DIR > 0 else "R45") if _STUCK_ESCAPE >= 2 else "FW"
        _set_prev(act)
        return act

                                                        
    if sensor_sum == 0:
        _ZERO_RUN += 1
        if _ZERO_RUN >= 5:
            _HIDDEN = None                          
        if _ZERO_RUN >= 2:
            act = _heuristic_search()
            _set_prev(act)
            return act
    else:
        _ZERO_RUN = 0

                                                            
    _FRAME_STACK.append(obs.astype(np.float32))
    stacked = _build_stacked(_FRAME_STACK)
    aug = np.concatenate([stacked, _PREV_ACTION])
    x = torch.as_tensor(aug, dtype=torch.float32).view(1, 1, INPUT_DIM)

    with torch.no_grad():
        q, _HIDDEN = _MODEL(x, _HIDDEN)
        nn_idx = int(q[:, -1, :].argmax(dim=1).item())

    h_act = _heuristic_sensors(obs)
    if h_act is not None:
        nn_q = float(q[0, 0, nn_idx])
        h_q = float(q[0, 0, ACTIONS.index(h_act)])
        act = ACTIONS[nn_idx] if nn_q > h_q + 3.0 else h_act
    else:
        act = ACTIONS[nn_idx]

    _set_prev(act)
    return act


def _set_prev(a):
    global _PREV_ACTION
    _PREV_ACTION = np.zeros(ACTION_DIM, np.float32)
    _PREV_ACTION[ACTIONS.index(a)] = 1.0
