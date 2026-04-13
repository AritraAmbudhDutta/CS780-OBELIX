"""
Visualization-only agent variant for RPPO checkpoint.
Adds a safe no-sensor heuristic fallback to reduce spinning loops in local visual tests.
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
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value, hidden


_MODEL = None
_HIDDEN = None
_LAST_RNG_ID = None
_FRAME_STACK = deque(maxlen=STACK_SIZE)
_PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
_ZERO_RUN = 0
_FW_COUNT = 0
_TURN_DIR = 1
_STUCK_ESCAPE = 0


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    model = RecurrentActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    if "actor_critic" in sd:
        model.load_state_dict(sd["actor_critic"])
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


def _heuristic_search():
    global _FW_COUNT, _TURN_DIR
    segment_len = 15
    if _FW_COUNT < segment_len:
        _FW_COUNT += 1
        return "FW"
    elif _FW_COUNT < segment_len + 2:
        _FW_COUNT += 1
        return "R45" if _TURN_DIR > 0 else "L45"
    else:
        _FW_COUNT = 0
        _TURN_DIR *= -1
        return "FW"


def _set_prev(action_str):
    global _PREV_ACTION
    idx = ACTIONS.index(action_str)
    _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
    _PREV_ACTION[idx] = 1.0


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN, _LAST_RNG_ID, _FRAME_STACK, _PREV_ACTION
    global _ZERO_RUN, _FW_COUNT, _TURN_DIR, _STUCK_ESCAPE

    _load_once()

    rid = id(rng)
    if _LAST_RNG_ID is None or rid != _LAST_RNG_ID:
        _HIDDEN = None
        _FRAME_STACK.clear()
        _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
        _ZERO_RUN = 0
        _FW_COUNT = 0
        _TURN_DIR = 1
        _STUCK_ESCAPE = 0
        _LAST_RNG_ID = rid

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
        if _ZERO_RUN >= 2:
            act = _heuristic_search()
            _set_prev(act)
            return act
    else:
        _ZERO_RUN = 0

    _FRAME_STACK.append(obs.astype(np.float32))
    stacked = _build_stacked(_FRAME_STACK)
    augmented = np.concatenate([stacked, _PREV_ACTION])

    x = torch.as_tensor(augmented, dtype=torch.float32).view(1, 1, INPUT_DIM)
    with torch.no_grad():
        logits, _, _HIDDEN = _MODEL(x, _HIDDEN)
        action_idx = int(logits[:, -1, :].argmax(dim=1).item())

    act = ACTIONS[action_idx]
    _set_prev(act)
    return act
