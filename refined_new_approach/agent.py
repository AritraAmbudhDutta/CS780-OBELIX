"""
agent.py — Codabench submission agent for OBELIX.

Supports both checkpoint formats:
  · Dueling Double DQN  (DualStreamQNet)
  · ActorCritic / PPO   (PolicyValueNet)

The model architecture is auto-detected from the checkpoint keys.

Temporal context is provided by stacking the last STACK_SIZE observations
into a single FULL_OBS_DIM-dimensional vector (short-term memory).
"""

import numpy as np
import os
from collections import deque

                                                         
_NET         = None                   
_TH          = None                           
_HISTORY     = None                         
_LAST_RAW    = None                                                              
_NET_KIND    = None                    

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

                                                  
STACK_SIZE    = 6
SINGLE_OBS_DIM = 18
FULL_OBS_DIM  = STACK_SIZE * SINGLE_OBS_DIM        


                                                                             
                            
                                                                             

class _RollingBuffer:
    """Maintains the last STACK_SIZE observations as a flat concatenated vector."""

    def __init__(self, k: int = 6, d: int = 18):
        self.k = k
        self.d = d
        self._window = deque(maxlen=k)

    def reset(self, obs) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).ravel()
        self._window.clear()
        for _ in range(self.k):
            self._window.append(flat.copy())
        return np.concatenate(list(self._window))

    def push(self, obs) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).ravel()
        if not self._window:
            return self.reset(obs)
        self._window.append(flat.copy())
        return np.concatenate(list(self._window))


                                                                             
                           
                                                                             

def _is_new_episode(prev_obs, curr_obs) -> bool:
    """Heuristic: if ≥10 sensor bits changed, assume a new episode started."""
    if prev_obs is None:
        return True
    a = np.asarray(prev_obs, dtype=np.float32).ravel()[:SINGLE_OBS_DIM]
    b = np.asarray(curr_obs, dtype=np.float32).ravel()[:SINGLE_OBS_DIM]
    return int(np.sum(a != b)) >= 10


                                                                             
                                                                       
                                                                             

class _DualStreamNet:
    """
    Factory: creates a DualStreamQNet matching D3QN training.

    Architecture:
      shared_encoder (Linear→ReLU→Linear→ReLU) → h1→h2
      value_head     (Linear→ReLU→Linear)       → h2//2 → 1
      advantage_head (Linear→ReLU→Linear)       → h2//2 → num_actions
      Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    """

    def __new__(cls, obs_dim: int = 108, num_actions: int = 5, layer_sizes=(256, 128)):
        import torch.nn as nn

        h1, h2 = layer_sizes

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_encoder = nn.Sequential(
                    nn.Linear(obs_dim, h1), nn.ReLU(),
                    nn.Linear(h1, h2),     nn.ReLU(),
                )
                self.value_head = nn.Sequential(
                    nn.Linear(h2, h2 // 2), nn.ReLU(),
                    nn.Linear(h2 // 2, 1),
                )
                self.advantage_head = nn.Sequential(
                    nn.Linear(h2, h2 // 2), nn.ReLU(),
                    nn.Linear(h2 // 2, num_actions),
                )

            def forward(self, x):
                enc = self.shared_encoder(x)
                V   = self.value_head(enc)
                A   = self.advantage_head(enc)
                return V + (A - A.mean(dim=1, keepdim=True))

        return _Net()


class _PolicyValueNet:
    """
    Factory: creates a PPO ActorCritic matching training.

    Architecture:
      backbone (Linear→ReLU→Linear→ReLU) → h→h//2
      actor    (Linear→ReLU→Linear)       → h//4 → num_actions  (logits)
      critic   (Linear→ReLU→Linear)       → h//4 → 1
    """

    def __new__(cls, obs_dim: int = 108, num_actions: int = 5, hidden: int = 256):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(obs_dim, hidden),     nn.ReLU(),
                    nn.Linear(hidden, hidden // 2), nn.ReLU(),
                )
                self.actor = nn.Sequential(
                    nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
                    nn.Linear(hidden // 4, num_actions),
                )
                self.critic = nn.Sequential(
                    nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
                    nn.Linear(hidden // 4, 1),
                )

            def forward(self, x):
                return self.actor(self.backbone(x))                       

        return _Net()


                                                                             
                                           
                                                                             

def _fallback(obs, rng: np.random.Generator) -> str:
    """Used when weights cannot be loaded."""
    stuck = len(obs) > 17 and obs[17] == 1
    probs = np.array([0.25, 0.25, 0.00, 0.25, 0.25]) if stuck\
        else np.array([0.05, 0.10, 0.70, 0.10, 0.05])
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]


                                                                             
                       
                                                                             

def _load_once() -> None:
    global _NET, _TH, _HISTORY, _NET_KIND

    if _NET is not None:
        return

    try:
        import torch
        _TH      = torch
        _HISTORY = _RollingBuffer(k=STACK_SIZE, d=SINGLE_OBS_DIM)

        weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
        if not os.path.exists(weights_path):
            print("[agent] weights.pth not found — using fallback policy")
            return

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

        if isinstance(ckpt, dict) and "network" in ckpt:
                                                                 
            _NET_KIND = "ppo"
            _NET = _PolicyValueNet(obs_dim=FULL_OBS_DIM, num_actions=5, hidden=256)
            _NET.load_state_dict(ckpt["network"])
            print("[agent] Loaded PPO weights")
        else:
                                                                                 
            _NET_KIND = "ddqn"
            _NET = _DualStreamNet(obs_dim=FULL_OBS_DIM, num_actions=5, layer_sizes=(256, 128))
            sd  = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            _NET.load_state_dict(sd)
            print("[agent] Loaded D3QN weights")

        _NET.eval()

    except Exception as err:
        _NET = None
        print(f"[agent] Weight load failed: {err}")


                                                                             
                                              
                                                                             

def policy(obs, rng=None) -> str:
    """
    Select the next action given the current observation.

    Args:
        obs : array-like, length 18 (raw sensor reading)
        rng : numpy Generator (optional, created if None)

    Returns:
        One of {"L45", "L22", "FW", "R22", "R45"}
    """
    global _LAST_RAW

    if rng is None:
        rng = np.random.default_rng()

    _load_once()

    if _NET is None or _TH is None or _HISTORY is None:
        return _fallback(obs, rng)

                                                                   
    if _is_new_episode(_LAST_RAW, obs):
        stacked = _HISTORY.reset(obs)
    else:
        stacked = _HISTORY.push(obs)

    _LAST_RAW = np.asarray(obs, dtype=np.float32).ravel()[:SINGLE_OBS_DIM].copy()

    t = _TH.from_numpy(stacked).unsqueeze(0)
    with _TH.no_grad():
        logits = _NET(t)
        act_idx = int(_TH.argmax(logits, dim=1).item())

    return ACTIONS[act_idx]
