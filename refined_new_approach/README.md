# CS780 OBELIX — D3QN Agent  (Refactored)

## File Map

| Your file               | Replaces (professor's) | Purpose                                    |
|-------------------------|------------------------|--------------------------------------------|
| `qnet.py`               | `model.py`             | Neural network architectures               |
| `dqn_core.py`           | `d3qnagent.py`         | FrameBuffer, PERBuffer, DuelDQNAgent       |
| `train.py`              | `train_d3qn.py`        | Training loop (PBRS + curriculum)          |
| `agent.py`              | `agent (2).py`         | **Submission agent** (stacked obs, auto-detect) |
| `agent_ddqn_simple.py`  | `agent_d3qn.py`        | Simple single-obs agent (fallback)         |

---

## How to Run Training

### Step 1 — Place files in your workspace

Copy into your OBELIX project directory (same folder as `obelix.py`):

```
qnet.py
dqn_core.py
train.py
```

### Step 2 — Run training

```bash
# Recommended (matches professor's hyper-parameters)
python train.py \
    --episodes 8000 \
    --lr 1e-4 \
    --batch_size 64 \
    --buffer_size 50000 \
    --n_stack 6 \
    --n_step 3 \
    --per_alpha 0.6 \
    --shaping_scale 10.0 \
    --eps_decay_steps 200000 \
    --target_update 2000 \
    --eval_interval 200 \
    --save_interval 50 \
    --out_dir ./checkpoints
```

**If you have more compute** (department server), run longer:

```bash
python train.py \
    --episodes 12000 \
    --max_steps 2000 \
    --lr 1e-4 \
    --batch_size 128 \
    --buffer_size 200000 \
    --n_stack 6 \
    --n_step 3 \
    --shaping_scale 10.0 \
    --eps_decay_steps 400000 \
    --target_update 2000 \
    --eval_interval 200 \
    --out_dir ./checkpoints
```

**Resume from checkpoint** (if interrupted):

```bash
python train.py --resume --out_dir ./checkpoints --episodes 8000
```

---

## What Gets Saved

```
checkpoints/
  obelix_agent.pth          ← latest checkpoint (saved every --save_interval episodes)
  best_obelix_agent.pth     ← best checkpoint by evaluation mean reward
  weights.pth               ← copy of best checkpoint (READY TO SUBMIT)
  train_log.jsonl           ← per-episode training metrics
  eval_log.jsonl            ← periodic evaluation metrics
```

---

## What to Submit to Codabench

You need **exactly two files** in your submission zip:

```
agent.py          ← from this repo (unchanged)
weights.pth       ← auto-generated at checkpoints/weights.pth
```

### Steps to prepare submission

```bash
# After training completes:
cp agent.py submission/agent.py
cp checkpoints/weights.pth submission/weights.pth

# Zip for Codabench:
cd submission && zip ../my_submission.zip agent.py weights.pth
```

The `agent.py` file:
- Is completely **self-contained** (no imports from `qnet.py` or `dqn_core.py`)
- Auto-detects whether `weights.pth` is a D3QN or PPO checkpoint
- Uses 6-frame stacking (108-dim input) to match training
- Falls back to a forward-biased random walk if weights are missing

---

## Techniques Used (Same as Professor's Solution)

| Technique                       | Where                          |
|---------------------------------|--------------------------------|
| Dueling Q-Network (D3QN)        | `qnet.py` → `DualStreamQNet`   |
| Double DQN (decoupled target)   | `dqn_core.py` → `update_weights()` |
| Prioritized Experience Replay   | `dqn_core.py` → `PERBuffer`    |
| n-step bootstrapped returns     | `dqn_core.py` → `_build_multistep_transition()` |
| Frame stacking (temporal memory)| `dqn_core.py` → `FrameBuffer`  |
| Forward-biased ε-greedy         | `dqn_core.py` → `FWD_BIAS_PROBS` |
| Stuck avoidance (turn-only)     | `dqn_core.py` → `TURN_ONLY_PROBS` |
| PBRS (3-component shaping)      | `train.py` → `apply_pbrs_shaping()` |
| Curriculum learning (4 phases)  | `train.py` → `schedule_curriculum()` |
| NoisyNet (optional)             | `qnet.py` → `NoisyDualStreamQNet` |

---

## Quick Sanity Check (before full training)

```bash
python train.py --episodes 20 --max_steps 200 --eval_interval 10
```

You should see episode logs printing without errors.
