"""
agent_dsac.py — OBELIX submission agent (Discrete SAC + GRU)

Submit this file (renamed to agent.py) + weights.pth to Codabench.

Architecture (Policy Network):
  Linear(77→128) → ReLU → GRU(128→128) → Linear(128→5)

Input: 4-frame stack (18×4=72) + previous action one-hot (5) = 77
Policy: greedy (argmax on policy logits; entropy only used in training).
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


class SACPolicyNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc(x))
        x, hidden = self.gru(x, hidden)
        logits = self.out(x)
        return logits, hidden


                                                                      
_MODEL = None
_HIDDEN = None
_LAST_RNG_ID = None
_FRAME_STACK = deque(maxlen=STACK_SIZE)
_PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    model = SACPolicyNet()
    sd = torch.load(wpath, map_location="cpu")
    if "policy" in sd:
        model.load_state_dict(sd["policy"])
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


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN, _LAST_RNG_ID, _FRAME_STACK, _PREV_ACTION

    _load_once()

    rid = id(rng)
    if _LAST_RNG_ID is None or rid != _LAST_RNG_ID:
        _HIDDEN = None
        _FRAME_STACK.clear()
        _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
        _LAST_RNG_ID = rid

    _FRAME_STACK.append(obs.astype(np.float32))
    stacked = _build_stacked(_FRAME_STACK)
    augmented = np.concatenate([stacked, _PREV_ACTION])

    x = torch.as_tensor(augmented, dtype=torch.float32).view(1, 1, INPUT_DIM)
    with torch.no_grad():
        logits, _HIDDEN = _MODEL(x, _HIDDEN)
        action_idx = int(logits[:, -1, :].argmax(dim=1).item())

    _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
    _PREV_ACTION[action_idx] = 1.0

    return ACTIONS[action_idx]
