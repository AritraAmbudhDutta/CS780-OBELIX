"""
agent_ddqn_simple.py — Lightweight single-observation D3QN submission agent.

Unlike agent.py, this version feeds the raw 18-dim observation directly
to the network (no frame stacking).  It is simpler but lacks temporal context.

Use agent.py for best performance.  This file exists as a minimal fallback
that is also valid for Codabench submission (just rename to agent.py).
"""

import numpy as np
import os

_NET         = None
_TH          = None
_LOAD_ERR    = None

ACTIONS = ("L45", "L22", "FW", "R22", "R45")


# ===========================================================================
# Inline DualStream network (no external imports required)
# ===========================================================================

class _InlineNet:
    """Factory that produces a DualStreamQNet compatible with single-obs training."""

    def __new__(cls, obs_dim: int = 18, num_actions: int = 5, layer_sizes=(256, 128)):
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


# ===========================================================================
# Fallback policy
# ===========================================================================

def _safe_fallback(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Biased random walk used when weights cannot be loaded."""
    stuck = len(obs) > 17 and obs[17] == 1
    if stuck:
        probs = np.array([0.25, 0.25, 0.00, 0.25, 0.25], dtype=float)
    else:
        probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]


# ===========================================================================
# One-time loader
# ===========================================================================

def _load_once() -> None:
    global _NET, _TH, _LOAD_ERR
    if _NET is not None:
        return
    try:
        import torch
        _TH  = torch
        _NET = _InlineNet(obs_dim=18, num_actions=5, layer_sizes=(256, 128))

        path = os.path.join(os.path.dirname(__file__), "weights.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            sd   = ckpt["model_state_dict"] if (isinstance(ckpt, dict)
                                                and "model_state_dict" in ckpt) else ckpt
            _NET.load_state_dict(sd)

        _NET.eval()

    except Exception as exc:
        _NET     = None
        _LOAD_ERR = str(exc)


# ===========================================================================
# Public API
# ===========================================================================

def policy(obs, rng=None) -> str:
    """
    Greedy action selection from the loaded DualStream Q-network.

    Args:
        obs : array-like, length 18 (raw sensor reading)
        rng : numpy Generator (optional)

    Returns:
        Action string: one of {"L45", "L22", "FW", "R22", "R45"}
    """
    if rng is None:
        rng = np.random.default_rng()

    _load_once()

    if _NET is None or _TH is None:
        return _safe_fallback(np.asarray(obs), rng)

    t = _TH.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    with _TH.no_grad():
        q_vals  = _NET(t)
        act_idx = int(_TH.argmax(q_vals, dim=1).item())

    return ACTIONS[act_idx]
