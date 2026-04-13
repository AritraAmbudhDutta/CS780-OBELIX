"""
agent_final.py — OBELIX submission agent (Simple DDQN + Heuristic Fallback)

CRITICAL FIX: When no sensors detect the box → use heuristic biased-forward
search instead of relying on the NN (which outputs garbage for all-zero obs).

Architecture: Linear(77→128) → ReLU → Linear(128→128) → ReLU → Linear(128→5)
Input: 4-frame stack (18×4=72) + prev-action one-hot (5) = 77
Policy: heuristic when sensors=0, greedy Q otherwise.
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
STACK_SIZE = 4
INPUT_DIM = OBS_DIM * STACK_SIZE + ACTION_DIM      


class QNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


                                                          
_MODEL = None
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
    model = QNet()
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


def _heuristic_search():
    """Biased forward walk with periodic turns (lawnmower search)."""
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


def _heuristic_with_sensors(obs):
    """Reactive heuristic when sensors detect the box."""
                                                            
    if obs[16] == 1:
        return "FW"

                                                            
    fwd_near = obs[5] + obs[7] + obs[9] + obs[11]
    if fwd_near > 0:
        return "FW"

                                                         
    fwd_far = obs[4] + obs[6] + obs[8] + obs[10]
    if fwd_far > 0:
        return "FW"

                             
    left_sum = obs[0] + obs[1] + obs[2] + obs[3]
                                
    right_sum = obs[12] + obs[13] + obs[14] + obs[15]

    if left_sum > 0 and right_sum == 0:
        return "L22" if obs[2] or obs[3] else "L45"                           
    if right_sum > 0 and left_sum == 0:
        return "R22" if obs[12] or obs[13] else "R45"
    if left_sum > 0 and right_sum > 0:
        return "FW"                               

    return None                                       


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_RNG_ID, _FRAME_STACK, _PREV_ACTION
    global _STEP, _ZERO_RUN, _FW_COUNT, _TURN_DIR, _STUCK_ESCAPE

    _load_once()

                        
    rid = id(rng)
    if _LAST_RNG_ID is None or rid != _LAST_RNG_ID:
        _FRAME_STACK.clear()
        _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
        _STEP = 0
        _ZERO_RUN = 0
        _FW_COUNT = 0
        _TURN_DIR = 1
        _STUCK_ESCAPE = 0
        _LAST_RNG_ID = rid

    _STEP += 1
    sensor_sum = float(np.sum(obs[:17]))

                                                          
    if obs[17] == 1:
        _STUCK_ESCAPE = 4                        
        _FW_COUNT = 0

    if _STUCK_ESCAPE > 0:
        _STUCK_ESCAPE -= 1
        if _STUCK_ESCAPE >= 2:
            act = "L45" if _TURN_DIR > 0 else "R45"
        else:
            act = "FW"
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

                                                             
    if sensor_sum > 0:
        h_act = _heuristic_with_sensors(obs)
        if h_act is not None:
                                                               
            _FRAME_STACK.append(obs.astype(np.float32))
            stacked = _build_stacked(_FRAME_STACK)
            augmented = np.concatenate([stacked, _PREV_ACTION])
            x = torch.as_tensor(augmented, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = _MODEL(x)
                nn_idx = int(q.argmax(dim=1).item())
                nn_q = float(q[0, nn_idx])
                h_q = float(q[0, ACTIONS.index(h_act)])

                                                         
            if nn_q > h_q + 5.0:
                act = ACTIONS[nn_idx]
            else:
                act = h_act
            _set_prev(act)
            return act

                              
    _FRAME_STACK.append(obs.astype(np.float32))
    stacked = _build_stacked(_FRAME_STACK)
    augmented = np.concatenate([stacked, _PREV_ACTION])
    x = torch.as_tensor(augmented, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = _MODEL(x)
        act = ACTIONS[int(q.argmax(dim=1).item())]
    _set_prev(act)
    return act


def _set_prev(action_str):
    global _PREV_ACTION
    idx = ACTIONS.index(action_str)
    _PREV_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)
    _PREV_ACTION[idx] = 1.0
