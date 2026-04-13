"""
qnet.py — Neural network architectures for the OBELIX robot agent.

Architectures provided:
  BasicQNet          : Shallow 3-layer MLP (baseline)
  DualStreamQNet     : Dueling Q-Network with shared encoder + separate value/advantage heads
  NoisyDualStreamQNet: Dueling Q-Network with learnable parametric noise for exploration
  ParametricNoiseLinear: Factorized-noise linear layer (Fortunato et al. 2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline Q-Network  (small MLP, 3 layers)
# ---------------------------------------------------------------------------

class BasicQNet(nn.Module):
    """
    Simple 3-layer MLP Q-Network.

    Input : sensor_dim bits (16 sonar + 1 IR + 1 attachment = 18)
    Output: num_actions Q-values  ("L45", "L22", "FW", "R22", "R45")
    """

    def __init__(self, sensor_dim: int = 18, num_actions: int = 5):
        super(BasicQNet, self).__init__()
        units = 64
        self.layer1 = nn.Linear(sensor_dim, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ---------------------------------------------------------------------------
# Parametric-Noise Linear Layer  (Fortunato et al. 2017 — "Noisy Networks")
# ---------------------------------------------------------------------------

class ParametricNoiseLinear(nn.Module):
    """
    Linear layer where the weight noise magnitude is itself learned.

    Standard linear:  y = W x + b
    Noisy linear:     y = (μ_W + σ_W ⊙ ε_W) x + (μ_b + σ_b ⊙ ε_b)

    Parameters:
      μ_W, μ_b  — learnable mean (same role as a normal weight/bias)
      σ_W, σ_b  — learnable noise scale, initialised to σ₀ / √fan_in
      ε_W, ε_b  — random noise, resampled every forward pass at train time

    Exploration behaviour:
      Large σ  → noisy outputs  → agent explores that region of state space
      Small σ  → deterministic  → agent exploits its current best estimate
      The network LEARNS when each regime is appropriate.

    Factorised noise (cheaper than fully-independent noise):
      ε_W[i,j] = f(p_i) · f(q_j)   where f(x) = sign(x) √|x|
    """

    def __init__(self, fan_in: int, fan_out: int, initial_sigma: float = 0.5):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self._init_sigma = initial_sigma

        # Learnable tensors
        self.weight_mu    = nn.Parameter(torch.empty(fan_out, fan_in))
        self.weight_sigma = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias_mu      = nn.Parameter(torch.empty(fan_out))
        self.bias_sigma   = nn.Parameter(torch.empty(fan_out))

        # Non-learnable noise buffers (kept on same device as params)
        self.register_buffer("noise_weight", torch.zeros(fan_out, fan_in))
        self.register_buffer("noise_bias",   torch.zeros(fan_out))

        self._init_parameters()
        self.resample_noise()

    def _init_parameters(self) -> None:
        spread = 1.0 / math.sqrt(self.fan_in)
        self.weight_mu.data.uniform_(-spread, spread)
        self.bias_mu.data.uniform_(-spread, spread)
        fill_val = self._init_sigma / math.sqrt(self.fan_in)
        self.weight_sigma.data.fill_(fill_val)
        self.bias_sigma.data.fill_(fill_val)

    @staticmethod
    def _scaled_noise(size: int) -> torch.Tensor:
        """Generate factorised noise: f(x) = sign(x) · √|x|."""
        raw = torch.randn(size)
        return raw.sign() * raw.abs().sqrt()

    def resample_noise(self) -> None:
        """Draw fresh factorised noise for weights and biases."""
        row_noise = self._scaled_noise(self.fan_in)
        col_noise = self._scaled_noise(self.fan_out)
        self.noise_weight.copy_(col_noise.outer(row_noise))
        self.noise_bias.copy_(col_noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            W = self.weight_mu + self.weight_sigma * self.noise_weight
            b = self.bias_mu   + self.bias_sigma   * self.noise_bias
        else:
            # Inference: disable noise, use learned mean only
            W = self.weight_mu
            b = self.bias_mu
        return F.linear(x, W, b)


# ---------------------------------------------------------------------------
# Dueling Q-Network  (standard linear, no noise)
# ---------------------------------------------------------------------------

class DualStreamQNet(nn.Module):
    """
    Dueling architecture separating state-value V(s) and action-advantage A(s,a).

    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

    The mean-subtraction trick (Wang et al. 2015) ensures identifiability
    and leads to more stable training than a single-head Q-network.

    Network layout:
      shared_encoder  → hidden_dim1 → hidden_dim2
      value_head      → hidden_dim2 // 2 → 1
      advantage_head  → hidden_dim2 // 2 → num_actions
    """

    def __init__(
        self,
        sensor_dim: int = 18,
        num_actions: int = 5,
        layer_sizes: tuple = (256, 128),
    ):
        super().__init__()
        h1, h2 = layer_sizes

        self.shared_encoder = nn.Sequential(
            nn.Linear(sensor_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.shared_encoder(x)
        V   = self.value_head(enc)
        A   = self.advantage_head(enc)
        return V + (A - A.mean(dim=1, keepdim=True))

    def resample_noise(self) -> None:
        """No-op — present for API compatibility with noisy variant."""
        pass


# ---------------------------------------------------------------------------
# Noisy Dueling Q-Network  (ParametricNoiseLinear + Dueling heads)
# ---------------------------------------------------------------------------

class NoisyDualStreamQNet(nn.Module):
    """
    Dueling Q-Network whose value/advantage heads use ParametricNoiseLinear,
    replacing ε-greedy exploration with learned, state-dependent exploration.

    The shared encoder uses plain Linear layers (noise in the heads suffices
    to produce diverse action outputs without over-randomising representations).

    Behaviour:
      Training  — noise active  → implicit exploration without explicit ε schedule
      Inference — noise removed → greedy best-action selection

    Architecture identical to DualStreamQNet except noisy heads:
      shared_encoder     : Linear → ReLU → Linear → ReLU
      value_hidden       : ParametricNoiseLinear (h2 → h2//2)
      value_out          : ParametricNoiseLinear (h2//2 → 1)
      advantage_hidden   : ParametricNoiseLinear (h2 → h2//2)
      advantage_out      : ParametricNoiseLinear (h2//2 → num_actions)
    """

    def __init__(
        self,
        sensor_dim: int = 72,
        num_actions: int = 5,
        layer_sizes: tuple = (256, 128),
    ):
        super().__init__()
        h1, h2 = layer_sizes

        # Deterministic feature extractor
        self.shared_encoder = nn.Sequential(
            nn.Linear(sensor_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        # Noisy value head
        self.value_hidden = ParametricNoiseLinear(h2, h2 // 2)
        self.value_out    = ParametricNoiseLinear(h2 // 2, 1)

        # Noisy advantage head
        self.advantage_hidden = ParametricNoiseLinear(h2, h2 // 2)
        self.advantage_out    = ParametricNoiseLinear(h2 // 2, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc  = self.shared_encoder(x)

        V    = F.relu(self.value_hidden(enc))
        V    = self.value_out(V)

        A    = F.relu(self.advantage_hidden(enc))
        A    = self.advantage_out(A)

        return V + (A - A.mean(dim=1, keepdim=True))

    def resample_noise(self) -> None:
        """Resample factorised noise in all four ParametricNoiseLinear layers."""
        self.value_hidden.resample_noise()
        self.value_out.resample_noise()
        self.advantage_hidden.resample_noise()
        self.advantage_out.resample_noise()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_q_network(sensor_dim: int, num_actions: int) -> BasicQNet:
    """Convenience constructor — returns a BasicQNet."""
    return BasicQNet(sensor_dim, num_actions)
