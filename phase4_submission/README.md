# OBELIX Phase 4 Submission — Fixed RL Approaches

## Why Phase 3 Approaches Failed (-318K scores)

| Root Cause | Impact |
|---|---|
| **Curriculum learning**: Robot spawned near box during training → never learned to explore/search | Agent outputs garbage when spawned randomly at eval time |
| **Aggressive reward shaping**: +12.0 zero-obs bonus dominated real signals | Agent learned to maximize fake rewards, not real task |
| **Codabench always uses `wall_obstacles=True`** | Training with inconsistent wall settings hurt generalization |
| **NN alone for all states** | NN outputs deterministic bad actions for all-zero observations |

## What Changed (Phase 4 Fixes)

1. **NO curriculum** — fully random spawns from episode 1
2. **Minimal reward shaping** — only anti-circling (-3) and anti-stuck (-10)
3. **Heuristic fallback at inference** — when sensors=0, use biased-forward search (lawnmower pattern) instead of NN
4. **Always `wall_obstacles=True`** during training (matches Codabench)
5. **Mixed difficulty** every episode (0/2/3 randomly)
6. **Stuck escape** — deterministic 90° turn + forward sequence

## Directory Structure

```
phase4_submission/
├── agent_final.py          ← Simple DDQN + heuristic fallback (SUBMIT THIS)
├── train_final.py          ← Training script for agent_final
├── agent_d3qn_v2.py        ← D3QN+GRU + heuristic fallback (alternative)
├── train_d3qn_v2.py        ← Training script for D3QN v2
├── eval_all.py             ← Codabench-identical local evaluation
├── make_submission.py      ← Creates submission.zip for Codabench
├── checkpoints_final/      ← Checkpoints from DDQN training
├── checkpoints_d3qn_v2/    ← Checkpoints from D3QN training
└── weights.pth             ← Best weights (auto-saved during training)

obl/
├── launch.py               ← Top-level launcher (run from obl/)
└── ...
```

## Quick Start

### 1. Train (from obl/ directory)

```bash
# Approach 1: Simple DDQN (recommended — faster, more predictable)
python launch.py --approach final --episodes 15000 --device auto

# Approach 2: D3QN+GRU (try if DDQN isn't good enough)
python launch.py --approach d3qn_v2 --episodes 12000 --device auto

# Or run directly from phase4_submission/
cd phase4_submission
python train_final.py --episodes 15000 --device cuda
```

### 2. Evaluate Locally (Codabench-identical)

```bash
cd phase4_submission
python eval_all.py --agent agent_final.py --runs 10
python eval_all.py --agent agent_d3qn_v2.py --runs 10
```

This reports:
- Per-difficulty scores (d=0, d=2, d=3)
- With/without walls
- Cumulative scores in Codabench format
- Weighted score estimate

### 3. Submit to Codabench

```bash
cd phase4_submission
python make_submission.py --approach final
# → Creates submission_final.zip (contains agent.py + weights.pth)

# To submit a specific checkpoint:
python make_submission.py --approach final --weights checkpoints_final/final_ep5000.pth
```

### 4. Rank Multiple Checkpoints

```bash
cd phase4_submission

# Find best checkpoint from DDQN checkpoints:
cd ..
python rank_checkpoints_codabench.py \
    --agent_file phase4_submission/agent_final.py \
    --weights_glob "phase4_submission/checkpoints_final/final_ep*.pth" \
    --include_base_weights --max_candidates 10
```

## Approach Details

### Approach 1: Simple DDQN (`agent_final.py`)

**Architecture**: MLP with 3 layers (77→128→128→5)

**Why no LSTM/GRU?** With only 18 binary sensor bits, the observation space is small enough that frame stacking (4 frames = 72 dims) captures sufficient temporal info. LSTM hidden states cause problems when they saturate on all-zero inputs.

**Inference Logic:**
1. Stuck (bit 17 = 1) → Deterministic escape: turn 90° + forward
2. All sensors = 0 for 2+ steps → Heuristic lawnmower search
3. Sensors active → Heuristic reactive turn + NN confirmation
4. Fallback → NN greedy Q-value

**Training:**
- Double DQN (online selects, target evaluates)
- Heuristic-guided exploration for search (not ε-greedy for zero-sensor states)
- ε-greedy only when sensors are active (near-box decisions)
- Slow ε decay (500K steps) for thorough exploration
- All transitions stored (both heuristic and NN-selected)

### Approach 2: D3QN v2 (`agent_d3qn_v2.py`)

**Architecture**: Dueling DQN with GRU (77→128→GRU→V+A)

**Fix vs Phase 3 version:**
- Hidden state reset after 5 consecutive zero-sensor steps
- Heuristic fallback (identical to Approach 1)
- No curriculum whatsoever
- Minimal shaping only

## Hyperparameters

| Parameter | DDQN (final) | D3QN v2 |
|---|---|---|
| Episodes | 15,000 | 12,000 |
| Max steps | 1,000 | 1,000 |
| Learning rate | 2e-4 | 1e-4 |
| Gamma | 0.99 | 0.99 |
| Batch size | 64 | 64 |
| Buffer | 300K | 300K |
| ε start→end | 1.0→0.05 | 1.0→0.05 |
| ε decay | 500K steps | 500K steps |
| Target update | 2000 steps | 2000 steps |
| Seq length | — | 16 |
| Walls | Always True | Always True |
| Difficulty | Mixed (0,2,3) | Mixed (0,2,3) |

## Expected Results

| Metric | Phase 3 (broken) | Phase 4 (fixed) |
|---|---|---|
| Per-episode score | -10,000 | -3,000 to -500 |
| cumulative_with_wall (30 runs) | -318,000 | -90,000 to -15,000 |
| Behavior | Spinning in place | Searches → finds → pushes box |

The heuristic alone (without any NN) should score ~-5000 per episode (from exploring efficiently but not optimally approaching the box). The NN improves on this by learning to approach and push effectively.
