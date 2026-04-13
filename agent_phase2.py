"""
agent.py  —  OBELIX Phase 2 (Blinking Box) submission agent.

Submit this file + weights.pth to Codabench.

How it works
------------
The box randomly disappears for 10-30 steps.  When invisible, all 18 sensor
bits go to zero and a memoryless agent has no idea where the box is.

We fix this with a 4-feature memory that survives invisible phases:
  [18]  was box visible on this step? (1 / 0)
  [19]  steps since box last seen, normalised to [0, 1]
  [20]  sin of angle to box when last seen
  [21]  cos of angle to box when last seen

The network (Dueling DQN, 22-dim input) was trained with:
  - Potential-based distance shaping  → pulls robot toward last-known box
  - Orbit penalty                     → stops the robot circling the box
  - IR bonus                          → strongly rewards closing in / attaching
  - Forward-search bonus              → keeps robot moving while box invisible
  - Push shaping                      → guides box to boundary once attached

The policy() function is called by the Codabench evaluator every step.
Episode resets are detected by watching for a new rng object (the evaluator
creates a fresh rng = np.random.default_rng(seed) before each episode).
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Constants  — must match what was used in train_phase2.py
# ---------------------------------------------------------------------------

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
MAX_STEPS_SINCE_SEEN = 50  # normalisation cap for steps-since-seen feature

# ---------------------------------------------------------------------------
# Network definition  (Dueling DQN, input_dim=22)
# Must be byte-for-byte identical to the DuelingNet in train_phase2.py so
# that the saved state_dict loads correctly.
# ---------------------------------------------------------------------------

_MODEL = None  # loaded once, reused for every policy() call


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class DuelingNet(nn.Module):
        def __init__(self, input_dim=22, output_dim=5, hidden=128):
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
            a = self.advantage_head(h)  # (B, 5)
            return v + a - a.mean(dim=1, keepdim=True)

    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    net = DuelingNet()
    net.load_state_dict(__import__("torch").load(wpath, map_location="cpu"))
    net.eval()
    _MODEL = net


# ---------------------------------------------------------------------------
# Per-episode memory  (module-level globals, reset each episode)
# ---------------------------------------------------------------------------

_mem_last_sin = 0.0  # sin of angle robot→box when last visible
_mem_last_cos = 1.0  # cos of angle robot→box when last visible
_mem_steps_gone = 0  # steps since box was last seen
_mem_vis_now = False  # was box visible on the most recent step
_last_rng_id = None  # id() of the rng from the previous call


def _reset_memory():
    """Reset per-episode memory to neutral defaults."""
    global _mem_last_sin, _mem_last_cos, _mem_steps_gone, _mem_vis_now
    _mem_last_sin = 0.0
    _mem_last_cos = 1.0
    _mem_steps_gone = 0
    _mem_vis_now = False


# ---------------------------------------------------------------------------
# policy()  —  called by Codabench every step
# ---------------------------------------------------------------------------


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Choose an action from the 18-bit OBELIX observation.

    Parameters
    ----------
    obs : np.ndarray, shape (18,), dtype float32
        Binary sensor vector from the environment.
    rng : np.random.Generator
        Fresh generator object at the start of each episode.
        We use id(rng) to detect episode resets.

    Returns
    -------
    str — one of "L45", "L22", "FW", "R22", "R45"
    """
    global _mem_last_sin, _mem_last_cos, _mem_steps_gone, _mem_vis_now, _last_rng_id

    # ---- load network once ------------------------------------------------
    _load_model()

    # ---- detect episode reset ---------------------------------------------
    # The evaluator (evaluate.py / evaluate_on_codabench.py) creates a brand
    # new rng object before every episode:
    #   rng = np.random.default_rng(seed)
    # When id(rng) changes we know a new episode has started.
    current_rng_id = id(rng)
    if current_rng_id != _last_rng_id:
        _reset_memory()
        _last_rng_id = current_rng_id

    # ---- update memory from sensor bits -----------------------------------
    # The box is "visible" when any sonar or IR bit is active.
    # When invisible the env sets the box frame to zero → all bits = 0.
    visible = bool(np.any(obs[:17]))

    if visible:
        # Estimate the lateral angle to the box from the sensor activation
        # pattern.  Sensor pairs (far/near) by position:
        #   [0,1]  far-left-rear    [2,3]  left
        #   [4,5]  fwd-left         [6,7]  fwd-left-inner
        #   [8,9]  fwd-right-inner  [10,11] fwd-right
        #   [12,13] right           [14,15] far-right-rear
        #   [16]   IR front
        lateral_weights = np.array(
            [
                -1.5,
                -1.5,  # far-left-rear  → strongly left
                -0.8,
                -0.8,  # left
                -0.3,
                -0.3,  # fwd-left
                -0.1,
                -0.1,  # fwd-left-inner
                0.1,
                0.1,  # fwd-right-inner
                0.3,
                0.3,  # fwd-right
                0.8,
                0.8,  # right
                1.5,
                1.5,  # far-right-rear → strongly right
                0.0,  # IR (front — no lateral component)
            ],
            dtype=np.float32,
        )

        active = obs[:17].astype(np.float32)
        w_total = float(np.sum(active))
        if w_total > 0.0:
            lateral = float(np.dot(lateral_weights, active) / w_total)
            angle_est = lateral * (math.pi / 3.0)  # map [-1.5,1.5] → [-π/2, π/2]
            _mem_last_sin = math.sin(angle_est)
            _mem_last_cos = math.cos(angle_est)

        _mem_steps_gone = 0
        _mem_vis_now = True
    else:
        _mem_steps_gone = min(_mem_steps_gone + 1, MAX_STEPS_SINCE_SEEN)
        _mem_vis_now = False

    # ---- build 22-dim augmented state -------------------------------------
    extra = np.array(
        [
            float(_mem_vis_now),  # [18]
            _mem_steps_gone / MAX_STEPS_SINCE_SEEN,  # [19]
            _mem_last_sin,  # [20]
            _mem_last_cos,  # [21]
        ],
        dtype=np.float32,
    )
    aug = np.concatenate([obs.astype(np.float32), extra])  # (22,)

    # ---- greedy action from network ---------------------------------------
    import torch

    x = torch.from_numpy(aug).unsqueeze(0)  # (1, 22)
    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0)  # (5,)
        action_idx = int(q_values.argmax().item())

    return ACTIONS[action_idx]
