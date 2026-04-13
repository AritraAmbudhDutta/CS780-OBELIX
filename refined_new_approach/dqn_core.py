"""
dqn_core.py — Core components for Dueling Double DQN with PER and n-step returns.

Classes:
  FrameBuffer           : Temporal observation stacker (frame stacking)
  Experience            : Named tuple for a single transition
  PERBuffer             : Prioritized Experience Replay (sum-based priorities)
  DuelDQNAgent          : Full D3QN agent (Dueling + Double + PER + n-step + biased explore)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qnet import DualStreamQNet


                                                                             
                              
                                                                             

class FrameBuffer:
    """
    Maintains a rolling window of the last K sensor observations and
    concatenates them into a single flat vector for the Q-network.

    Motivation:
      A single OBELIX sensor reading (18 bits) is Markovian only in a
      fully-observable world.  In practice the agent cannot infer:
        · its recent trajectory (was it turning? going straight?)
        · temporal persistence of objects (box just appeared vs. long visible)
        · being newly stuck vs. repeatedly stuck

      Stacking K consecutive readings restores approximate Markovian-ness
      and gives the network implicit short-term memory — identical in spirit
      to the frame-stack used by Atari DQN (Mnih et al. 2015).

    Example (K=4, 18-dim observations):
      Single observation   →  18 dims
      4-frame stack        →  72 dims  [t-3 | t-2 | t-1 | t]

    Usage:
      buf = FrameBuffer(stack_size=4, single_dim=18)
      stacked = buf.reset(initial_obs)   # returns 72-dim array
      stacked = buf.advance(next_obs)    # returns 72-dim array
    """

    def __init__(self, stack_size: int = 4, single_dim: int = 18) -> None:
        self.stack_size = stack_size
        self.single_dim = single_dim
        self._queue: deque = deque(maxlen=stack_size)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the concatenated output."""
        return self.stack_size * self.single_dim

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Initialise buffer by repeating the first observation K times."""
        flat = np.asarray(obs, dtype=np.float32).ravel()
        self._queue.clear()
        for _ in range(self.stack_size):
            self._queue.append(flat.copy())
        return self._flatten()

    def advance(self, obs: np.ndarray) -> np.ndarray:
        """Append a new observation and return the updated stack."""
        flat = np.asarray(obs, dtype=np.float32).ravel()
        self._queue.append(flat.copy())
        return self._flatten()

    def _flatten(self) -> np.ndarray:
        return np.concatenate(list(self._queue), axis=0)


                                                                             
                           
                                                                             

@dataclass
class Experience:
    """Stores one environment transition (s, a, r, s', done)."""
    obs:      np.ndarray
    action:   int
    reward:   float
    next_obs: np.ndarray
    terminal: bool


                                                                             
                                      
                                                                             

class PERBuffer:
    """
    Replay buffer that samples transitions proportional to their TD-error.

    Priority of transition i:  p_i = |δ_i| + ε
    Sampling probability:      P(i) ∝ p_i ^ α
    Importance-sampling weight: w_i = (N · P(i))^{-β}  (normalised)

    α controls prioritisation strength  (0 → uniform, 1 → full priority)
    β anneals 0.4 → 1.0 to correct the bias introduced by non-uniform sampling.
    """

    def __init__(self, max_size: int, priority_exponent: float = 0.6) -> None:
        self.max_size = int(max_size)
        self.alpha = float(priority_exponent)
        self._storage: list[Experience] = []
        self._priorities = np.zeros((self.max_size,), dtype=np.float64)
        self._write_pos = 0

    def __len__(self) -> int:
        return len(self._storage)

    def store(self, exp: Experience) -> None:
        """Insert a transition, assigning it the current maximum priority."""
        max_p = float(self._priorities.max()) if self._storage else 1.0

        if len(self._storage) < self.max_size:
            self._storage.append(exp)
        else:
            self._storage[self._write_pos] = exp

        self._priorities[self._write_pos] = max(max_p, 1e-6)
        self._write_pos = (self._write_pos + 1) % self.max_size

    def draw(
        self,
        num_samples: int,
        importance_weight_exponent: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a mini-batch using priority-proportional probabilities."""
        if not self._storage:
            raise RuntimeError("Cannot draw from an empty buffer")

        raw_p = self._priorities[: len(self._storage)]
        scaled = raw_p ** self.alpha
        probs  = scaled / scaled.sum()

        chosen = np.random.choice(len(self._storage), num_samples, p=probs)
        batch  = [self._storage[i] for i in chosen]

        obs_batch      = np.stack([e.obs      for e in batch], axis=0)
        action_batch   = np.array([e.action   for e in batch], dtype=np.int64)
        reward_batch   = np.array([e.reward   for e in batch], dtype=np.float32)
        next_obs_batch = np.stack([e.next_obs for e in batch], axis=0)
        done_batch     = np.array([e.terminal for e in batch], dtype=np.float32)

        N       = len(self._storage)
        raw_w   = (N * probs[chosen]) ** (-importance_weight_exponent)
        raw_w  /= raw_w.max()
        is_weights = raw_w.astype(np.float32)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, chosen, is_weights

    def refresh_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """Write updated TD-error-based priorities back to storage."""
        for idx, p in zip(indices, new_priorities):
            self._priorities[int(idx)] = float(max(p, 1e-6))


                                                                             
                                                                              
                                                                             

class DuelDQNAgent:
    """
    Full agent combining:
      · Dueling Q-Network       — separates V(s) and A(s,a) for better value estimation
      · Double DQN              — decouples action selection & evaluation to reduce overestimation
      · Prioritized Replay      — focuses learning on high-TD-error transitions
      · n-step returns          — propagates reward signal N steps ahead, faster learning
      · Forward-biased ε-greedy — exploration favours 'FW' to encourage forward motion

    Epsilon anneals linearly from ε_start → ε_end over epsilon_decay steps.
    IS-weight exponent β anneals from β_start → 1.0 over beta_horizon steps.
    """

                                                              
    FWD_BIAS_PROBS  = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)
                                                          
    TURN_ONLY_PROBS = np.array([0.25, 0.25, 0.00, 0.25, 0.25], dtype=np.float64)

    def __init__(
        self,
        obs_dim: int = 18,
        num_actions: int = 5,
        arch_sizes: tuple = (256, 128),
        learning_rate: float = 1e-4,
        discount: float = 0.99,
        buffer_size: int = 50_000,
        minibatch: int = 64,
        eps_high: float = 1.0,
        eps_low: float = 0.05,
        eps_horizon: int = 200_000,
        target_sync_every: int = 2000,
                    
        per_alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_horizon: int = 200_000,
        per_min_priority: float = 1e-5,
                       
        n_step: int = 3,
        device: Optional[str] = None,
    ) -> None:

        self.obs_dim    = int(obs_dim)
        self.num_actions = int(num_actions)
        self.discount    = float(discount)
        self.minibatch   = int(minibatch)

        self.eps_high    = float(eps_high)
        self.eps_low     = float(eps_low)
        self.eps_horizon = int(max(1, eps_horizon))
        self.epsilon     = float(eps_high)

        self.target_sync_every = int(max(1, target_sync_every))

                       
        self.beta_start   = float(beta_start)
        self.beta_horizon = int(max(1, beta_horizon))
        self.per_min_p    = float(per_min_priority)

                       
        self.n_step = int(max(1, n_step))
        self._nstep_queue: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] =\
            deque(maxlen=self.n_step)

                          
        if device is not None:
            chosen = device
        elif torch.cuda.is_available():
            chosen = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
        self.device = torch.device(chosen)

                  
        self.online_net = DualStreamQNet(
            sensor_dim=self.obs_dim,
            num_actions=self.num_actions,
            layer_sizes=arch_sizes,
        ).to(self.device)

        self.frozen_net = DualStreamQNet(
            sensor_dim=self.obs_dim,
            num_actions=self.num_actions,
            layer_sizes=arch_sizes,
        ).to(self.device)
        self.frozen_net.load_state_dict(self.online_net.state_dict())
        self.frozen_net.eval()

        self.opt    = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay = PERBuffer(max_size=buffer_size, priority_exponent=per_alpha)

        self.total_steps  = 0                                  
        self.update_steps = 0                                                     

                                                                        
               
                                                                        

    def _current_epsilon(self) -> float:
        frac = min(1.0, self.total_steps / float(self.eps_horizon))
        return self.eps_high + frac * (self.eps_low - self.eps_high)

    def _current_beta(self) -> float:
        frac = min(1.0, self.update_steps / float(self.beta_horizon))
        return self.beta_start + frac * (1.0 - self.beta_start)

                                                                        
                      
                                                                        

    def choose_action(
        self,
        obs: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        training: bool = True,
        greedy_eps: float = 0.0,
    ) -> int:
        """
        ε-greedy action selection with forward-biased random exploration.

        During training:   ε follows the annealing schedule
        During inference:  ε = greedy_eps (default 0 → fully greedy)
        """
        gen = rng if rng is not None else np.random.default_rng()

        self.epsilon = self._current_epsilon()
        threshold = self.epsilon if training else float(max(0.0, greedy_eps))

        self.total_steps += 1

        if gen.random() < threshold:
            stuck = (obs is not None and len(obs) >= 18 and obs[-1] == 1)
            dist  = self.TURN_ONLY_PROBS if stuck else self.FWD_BIAS_PROBS
            return int(gen.choice(self.num_actions, p=dist))

        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_net(t).argmax(dim=1).item())

                                                                        
                                 
                                                                        

    def _build_multistep_transition(self) -> Optional[Experience]:
        """Convert the n-step queue into a single discounted Experience."""
        if len(self._nstep_queue) < self.n_step:
            return None

        cumulative_r   = 0.0
        final_next_obs = self._nstep_queue[-1][3]
        final_done     = self._nstep_queue[-1][4]

        for step_i, (_, _, r, ns, d) in enumerate(self._nstep_queue):
            cumulative_r   += (self.discount ** step_i) * float(r)
            final_next_obs  = ns
            if d:
                final_done = True
                break

        root_obs, root_act, _, _, _ = self._nstep_queue[0]
        return Experience(
            obs      = np.array(root_obs,       dtype=np.float32, copy=True),
            action   = int(root_act),
            reward   = float(cumulative_r),
            next_obs = np.array(final_next_obs, dtype=np.float32, copy=True),
            terminal = bool(final_done),
        )

    def _drain_nstep_buffer(self) -> None:
        """Flush remaining items from the n-step queue at episode end."""
        while self._nstep_queue:
            exp = self._build_multistep_transition()
            if exp is not None:
                self.replay.store(exp)
            self._nstep_queue.popleft()

                                                                        
                              
                                                                        

    def record_and_learn(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition (handling n-step) and trigger a learning update."""
        if self.n_step == 1:
            self.replay.store(Experience(
                obs      = np.array(obs,      dtype=np.float32, copy=True),
                action   = int(action),
                reward   = float(reward),
                next_obs = np.array(next_obs, dtype=np.float32, copy=True),
                terminal = bool(done),
            ))
        else:
            self._nstep_queue.append((
                np.array(obs,      dtype=np.float32, copy=True),
                int(action),
                float(reward),
                np.array(next_obs, dtype=np.float32, copy=True),
                bool(done),
            ))
            exp = self._build_multistep_transition()
            if exp is not None:
                self.replay.store(exp)
                self._nstep_queue.popleft()
            if done:
                self._drain_nstep_buffer()

        if len(self.replay) >= self.minibatch:
            return self.update_weights()

        return None

                                                                        
                                                     
                                                                        

    def update_weights(self) -> Optional[float]:
        """One gradient step using a priority-sampled mini-batch."""
        if len(self.replay) < self.minibatch:
            return None

        beta = self._current_beta()
        obs_b, act_b, rew_b, nobs_b, done_b, idx_b, w_b =\
            self.replay.draw(self.minibatch, importance_weight_exponent=beta)

        s  = torch.from_numpy(obs_b).to(self.device)
        a  = torch.from_numpy(act_b).to(self.device)
        r  = torch.from_numpy(rew_b).to(self.device)
        s2 = torch.from_numpy(nobs_b).to(self.device)
        d  = torch.from_numpy(done_b).to(self.device)
        w  = torch.from_numpy(w_b).to(self.device)

                          
        q_pred = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                                                                           
        disc_n = self.discount ** self.n_step
        with torch.no_grad():
            best_next_a = self.online_net(s2).argmax(dim=1, keepdim=True)
            q_next      = self.frozen_net(s2).gather(1, best_next_a).squeeze(1)
            td_target   = r + (1.0 - d) * disc_n * q_next

                                         
        td_errors   = td_target - q_pred
        elem_loss   = F.smooth_l1_loss(q_pred, td_target, reduction="none")
        total_loss  = (w * elem_loss).mean()

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.opt.step()

                                                    
        refreshed_p = td_errors.detach().abs().cpu().numpy() + self.per_min_p
        self.replay.refresh_priorities(idx_b, refreshed_p)

        self.update_steps += 1
        if self.update_steps % self.target_sync_every == 0:
            self.sync_target()

        return float(total_loss.item())

                                                                        
               
                                                                        

    def sync_target(self) -> None:
        """Copy online network weights into the frozen target network."""
        self.frozen_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist the full agent state to disk."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.online_net.state_dict(),
                "target_state_dict":    self.frozen_net.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "epsilon":              self.epsilon,
                "total_steps":          self.total_steps,
                "update_steps":         self.update_steps,
            },
            dest,
        )

    def load_checkpoint(self, path: Union[str, Path], map_location: str = "cpu") -> None:
        """Restore agent state from a saved checkpoint."""
        ckpt = torch.load(Path(path), map_location=map_location)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.online_net.load_state_dict(ckpt["model_state_dict"])
            self.frozen_net.load_state_dict(
                ckpt.get("target_state_dict", ckpt["model_state_dict"])
            )
            opt_state = ckpt.get("optimizer_state_dict")
            if opt_state is not None:
                self.opt.load_state_dict(opt_state)
            self.epsilon      = float(ckpt.get("epsilon",      self.epsilon))
            self.total_steps  = int(ckpt.get("total_steps",    self.total_steps))
            self.update_steps = int(ckpt.get("update_steps",   self.update_steps))
        else:
                                     
            self.online_net.load_state_dict(ckpt)
            self.frozen_net.load_state_dict(self.online_net.state_dict())

        self.online_net.to(self.device)
        self.frozen_net.to(self.device)
        self.frozen_net.eval()
