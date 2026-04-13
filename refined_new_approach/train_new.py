"""
train.py — Training entry point for OBELIX D3QN agent.
============================================================
Run:
    python train.py                        # default settings
    python train.py --episodes 8000 --lr 1e-4 --n_stack 6

Key techniques:
  1. Potential-Based Reward Shaping (PBRS) with three gradient signals:
       a) Bot → Box distance      (always active)
       b) Heading alignment       (pre-attachment: face the box)
       c) Box → Nearest wall      (post-attachment: push toward boundary)
  2. Curriculum learning:  no-wall warm-up → walls → blinking → all difficulties
  3. Prioritized Experience Replay  (PER)
  4. n-step bootstrapped returns
  5. Observation frame stacking for short-term memory
"""

import argparse
import json
import math
import os
import time

import numpy as np

from dqn_core import DuelDQNAgent, FrameBuffer
from obelix import OBELIX

ACTION_SET = ["L45", "L22", "FW", "R22", "R45"]


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


                                                                             
                                      
                                                                             

def bot_to_box_dist(env: OBELIX) -> float:
    """Euclidean distance between robot centre and box centre."""
    dx = env.bot_center_x - env.box_center_x
    dy = env.bot_center_y - env.box_center_y
    return math.sqrt(dx * dx + dy * dy)


def box_wall_distance(env: OBELIX) -> float:
    """
    Distance from the box centre to the nearest arena boundary wall.

    The arena has a 10-pixel border, so usable space is
    [margin, frame_w - margin] × [margin, frame_h - margin].
    """
    border = 10
    fh, fw = env.frame_size[0], env.frame_size[1]
    clearance_left   = env.box_center_x - border
    clearance_right  = (fw - border) - env.box_center_x
    clearance_top    = env.box_center_y - border
    clearance_bottom = (fh - border) - env.box_center_y
    return float(min(clearance_left, clearance_right, clearance_top, clearance_bottom))


def heading_alignment(env: OBELIX) -> float:
    """
    Cosine similarity between the robot's facing direction and the
    vector from robot centre to box centre.  Range: [-1, +1].

      +1 → robot is pointing directly at the box
       0 → robot is perpendicular to the box direction
      -1 → robot is pointing directly away from the box
    """
    dx = env.box_center_x - env.bot_center_x
    dy = env.box_center_y - env.bot_center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return 1.0                                  

                                
    theta = math.radians(env.facing_angle)
    fx, fy = math.cos(theta), math.sin(theta)

                                   
    tx, ty = dx / dist, dy / dist

    return fx * tx + fy * ty                                                


                                                                             
                                     
                                                                             

def apply_pbrs_shaping(
    env: OBELIX,
    base_reward: float,
    d_bot_box_prev: float,
    d_bot_box_curr: float,
    d_box_wall_prev: float,
    d_box_wall_curr: float,
    align_prev: float,
    align_curr: float,
    arena_diagonal: float,
    gamma: float,
    scale: float,
) -> float:
    """
    Add three Potential-Based Reward Shaping (PBRS) components to base_reward.

    PBRS formula:  F(s, s') = γ · Φ(s') − Φ(s)
    Property:      PBRS does NOT change the optimal policy (Ng et al. 1999).

    Component A — Approach (always active):
        Φ_A(s) = −dist(bot, box) / arena_diagonal
        Guides the agent toward the box regardless of push state.

    Component B — Heading (before attachment only):
        Φ_B(s) = cosine_similarity(facing, direction_to_box)
        Encourages the agent to FACE the box, not just be near it.
        Without this, the agent can spiral around the box and never attach.

    Component C — Push (after attachment only):
        Φ_C(s) = −dist(box, nearest_wall) / arena_diagonal
        Provides gradient for pushing the box toward a wall.
        Without this, the push phase has no shaping signal → random pushing.
    """
    shaped = base_reward

                                   
    phi_a_prev = -d_bot_box_prev / arena_diagonal
    phi_a_curr = -d_bot_box_curr / arena_diagonal
    shaped += scale * (gamma * phi_a_curr - phi_a_prev)

    if not env.enable_push:
                                                            
        shaped += (scale * 0.5) * (gamma * align_curr - align_prev)
    else:
                                                                     
        phi_c_prev = -d_box_wall_prev / arena_diagonal
        phi_c_curr = -d_box_wall_curr / arena_diagonal
        shaped += (scale * 2.0) * (gamma * phi_c_curr - phi_c_prev)

    return shaped


                                                                             
                      
                                                                             

def schedule_curriculum(current_episode: int) -> dict:
    """
    Four-phase difficulty schedule that gradually exposes the agent to
    harder conditions over the course of training.

    Phase 1 (ep    1 –  500)  no walls,  difficulty 0        — warmup
    Phase 2 (ep  501 – 2000)  with walls, difficulty 0       — obstacle nav
    Phase 3 (ep 2001 – 4000)  with walls, diff 0+2 (blink)   — handle blinking
    Phase 4 (ep 4001 – ∞   )  with walls, diff 0+2+3 (all)   — full challenge
    """
    if current_episode <= 500:
        return {"use_walls": False, "diff_pool": [0]}
    elif current_episode <= 2000:
        return {"use_walls": True,  "diff_pool": [0]}
    elif current_episode <= 4000:
        return {"use_walls": True,  "diff_pool": [0, 2]}
    else:
        return {"use_walls": True,  "diff_pool": [0, 2, 3]}


                                                                             
                                                           
                                                                             

def run_evaluation(
    agent: DuelDQNAgent,
    num_runs: int,
    seed_offset: int,
    scale: int,
    arena_sz: int,
    step_limit: int,
    walls: bool,
    box_spd: int,
    diff_list: list,
    stack_k: int = 4,
) -> dict:
    """
    Run num_runs greedy episodes and return summary statistics.

    Difficulty levels and seeds are cycled so each difficulty appears
    roughly equally often — same protocol as Codabench.
    """
    episode_returns = []

    for i in range(num_runs):
        diff = diff_list[i % len(diff_list)]
        seed = seed_offset + i

        env = OBELIX(
            scaling_factor=scale,
            arena_size=arena_sz,
            max_steps=step_limit,
            wall_obstacles=walls,
            difficulty=diff,
            box_speed=box_spd,
            seed=seed,
        )

        raw = env.reset(seed=seed)
        buf = FrameBuffer(stack_size=stack_k, single_dim=18)
        obs = buf.reset(raw)
        gen = np.random.default_rng(seed)
        done = False
        ep_ret = 0.0

        while not done:
            act_idx = agent.choose_action(obs, rng=gen, training=False, greedy_eps=0.0)
            raw, rew, done = env.step(ACTION_SET[act_idx], render=False)
            obs = buf.advance(raw)
            ep_ret += float(rew)

        episode_returns.append(ep_ret)

    arr = np.asarray(episode_returns)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


                                                                             
               
                                                                             

def run_training() -> None:
    p = argparse.ArgumentParser(description="Train OBELIX agent with Dueling Double DQN")

                               
    p.add_argument("--episodes",      type=int,   default=8000)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--resume",        action="store_true",
                   help="Continue from existing checkpoint")

                         
    p.add_argument("--scaling_factor", type=int,  default=5)
    p.add_argument("--arena_size",     type=int,  default=500)
    p.add_argument("--max_steps",      type=int,  default=2000)
    p.add_argument("--box_speed",      type=int,  default=2)

                                  
    p.add_argument("--h1", type=int, default=256, help="First hidden layer size")
    p.add_argument("--h2", type=int, default=128, help="Second hidden layer size")

                          
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--buffer_size",       type=int,   default=50_000)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--eps_start",         type=float, default=1.0)
    p.add_argument("--eps_end",           type=float, default=0.05)
    p.add_argument("--eps_decay_steps",   type=int,   default=200_000)
    p.add_argument("--target_update",     type=int,   default=2000)

                 
    p.add_argument("--per_alpha",         type=float, default=0.6)
    p.add_argument("--per_beta_init",     type=float, default=0.4)
    p.add_argument("--per_beta_steps",    type=int,   default=200_000)
    p.add_argument("--per_min_priority",  type=float, default=1e-5)

                            
    p.add_argument("--n_step",            type=int,   default=3)

                                  
    p.add_argument("--n_stack",           type=int,   default=6,
                   help="Number of frames to stack (1 = no stacking)")

                            
    p.add_argument("--shaping_scale",     type=float, default=10.0,
                   help="Multiplier applied to all PBRS shaping components")

                                 
    p.add_argument("--out_dir",           type=str,   default=".")
    p.add_argument("--ckpt_name",         type=str,   default="obelix_agent.pth")
    p.add_argument("--save_interval",     type=int,   default=50,
                   help="Save a periodic checkpoint every N episodes")
    p.add_argument("--eval_interval",     type=int,   default=200)
    p.add_argument("--eval_episodes",     type=int,   default=6)
    p.add_argument("--render_interval",   type=int,   default=0)

    cfg = p.parse_args()

                     
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path           = os.path.join(cfg.out_dir, cfg.ckpt_name)
    best_path           = os.path.join(cfg.out_dir, f"best_{cfg.ckpt_name}")
    best_raw_path       = os.path.join(cfg.out_dir, "best_raw_reward.pth")
    best_avg50_path     = os.path.join(cfg.out_dir, "best_avg50_reward.pth")

    master_rng  = np.random.default_rng(cfg.seed)
    stacked_dim = cfg.n_stack * 18                      

    agent = DuelDQNAgent(
        obs_dim=stacked_dim,
        num_actions=5,
        arch_sizes=(cfg.h1, cfg.h2),
        learning_rate=cfg.lr,
        discount=cfg.gamma,
        buffer_size=cfg.buffer_size,
        minibatch=cfg.batch_size,
        eps_high=cfg.eps_start,
        eps_low=cfg.eps_end,
        eps_horizon=cfg.eps_decay_steps,
        target_sync_every=cfg.target_update,
        per_alpha=cfg.per_alpha,
        beta_start=cfg.per_beta_init,
        beta_horizon=cfg.per_beta_steps,
        per_min_priority=cfg.per_min_priority,
        n_step=cfg.n_step,
    )

    if cfg.resume and os.path.exists(ckpt_path):
        agent.load_checkpoint(ckpt_path)
        print(f"Resumed training from: {ckpt_path}")

    arena_diag = math.sqrt(2) * cfg.arena_size                         

                                    
    top_mean        = -float("inf")                                      
    top_raw_reward  = -float("inf")                                                         
    top_avg50       = -float("inf")                                                              
    recent_rewards: list[float] = []                                          

    print("=" * 72)
    print("OBELIX Training  —  Dueling Double DQN + PER + n-step + PBRS")
    print("=" * 72)
    print(f"  Episodes        : {cfg.episodes}")
    print(f"  Learning rate   : {cfg.lr}")
    print(f"  Batch / Buffer  : {cfg.batch_size} / {cfg.buffer_size}")
    print(f"  PER α           : {cfg.per_alpha}")
    print(f"  n-step          : {cfg.n_step}")
    print(f"  Frame stack     : {cfg.n_stack} × 18 = {stacked_dim}-dim input")
    print(f"  Shaping scale   : {cfg.shaping_scale}")
    print(f"  ε decay steps   : {cfg.eps_decay_steps}")
    print(f"  Curriculum      : P1(no-wall,1-500) → P2(wall,501-2000) → "
          f"P3(blink,2001-4000) → P4(all,4001+)")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Best eval       : {best_path}")
    print(f"  Best raw reward : {best_raw_path}")
    print(f"  Best avg-50     : {best_avg50_path}")
    print(f"  Device          : {agent.device}")
    print("=" * 72)

    train_start_time = time.time()

    for ep in range(1, cfg.episodes + 1):
                              
        sched      = schedule_curriculum(ep)
        diff_pool  = sched["diff_pool"]
        use_walls  = sched["use_walls"]
        difficulty = int(master_rng.choice(diff_pool))
        ep_seed    = cfg.seed + ep

        env = OBELIX(
            scaling_factor=cfg.scaling_factor,
            arena_size=cfg.arena_size,
            max_steps=cfg.max_steps,
            wall_obstacles=use_walls,
            difficulty=difficulty,
            box_speed=cfg.box_speed,
            seed=ep_seed,
        )

        raw_obs   = env.reset(seed=ep_seed)
        frame_buf = FrameBuffer(stack_size=cfg.n_stack, single_dim=18)
        obs       = frame_buf.reset(raw_obs)

        done          = False
        ep_env_ret    = 0.0                                                   
        ep_shaped_ret = 0.0                                               
        ep_steps      = 0
        losses        = []
        detected_box  = False
        pushed_wall   = False
        detect_step   = -1
        act_histogram = [0] * 5

        render_this = cfg.render_interval > 0 and (ep % cfg.render_interval == 0)

                                              
        prev_d_bb    = bot_to_box_dist(env)
        prev_d_bwall = box_wall_distance(env)
        prev_align   = heading_alignment(env)

        while not done:
            act_idx = agent.choose_action(obs, rng=master_rng, training=True)
            act     = ACTION_SET[act_idx]
            act_histogram[act_idx] += 1

            raw_next, env_rew, done = env.step(act, render=render_this)
            next_obs = frame_buf.advance(raw_next)

                                
            if env.enable_push and not detected_box:
                detected_box = True
                detect_step  = ep_steps + 1
            if done and env.enable_push and env._box_touches_boundary(
                    env.box_center_x, env.box_center_y):
                pushed_wall = True

                                  
            curr_d_bb    = bot_to_box_dist(env)
            curr_d_bwall = box_wall_distance(env)
            curr_align   = heading_alignment(env)

            shaped_rew = apply_pbrs_shaping(
                env            = env,
                base_reward    = float(env_rew),
                d_bot_box_prev = prev_d_bb,
                d_bot_box_curr = curr_d_bb,
                d_box_wall_prev= prev_d_bwall,
                d_box_wall_curr= curr_d_bwall,
                align_prev     = prev_align,
                align_curr     = curr_align,
                arena_diagonal = arena_diag,
                gamma          = cfg.gamma,
                scale          = cfg.shaping_scale,
            )
            prev_d_bb    = curr_d_bb
            prev_d_bwall = curr_d_bwall
            prev_align   = curr_align

                   
            loss_val = agent.record_and_learn(obs, act_idx, shaped_rew, next_obs, done)
            if loss_val is not None:
                losses.append(loss_val)

            obs            = next_obs
            ep_env_ret    += float(env_rew)
            ep_shaped_ret += shaped_rew
            ep_steps      += 1

                           
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        flags = ""
        if detected_box:
            flags += " ✓BOX"
        if pushed_wall:
            flags += " ✓WALL"

        if not use_walls:
            phase = "P1-warmup"
        elif diff_pool == [0]:
            phase = "P2-walls"
        elif diff_pool == [0, 2]:
            phase = "P3-blink"
        else:
            phase = "P4-full"

        elapsed_seconds = time.time() - train_start_time
        avg_seconds_per_episode = elapsed_seconds / ep
        remaining_seconds = avg_seconds_per_episode * max(cfg.episodes - ep, 0)

        print(
            f"[Ep {ep:4d}/{cfg.episodes}] {phase} d={difficulty} "
            f"envR={ep_env_ret:9.1f} shapedR={ep_shaped_ret:9.1f} "
            f"steps={ep_steps:4d} ε={agent.epsilon:.4f} loss={avg_loss:.5f} "
            f"elapsed={format_hms(elapsed_seconds)} remaining={format_hms(remaining_seconds)}{flags}"
        )

                                         
        recent_rewards.append(ep_env_ret)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        current_avg50 = float(np.mean(recent_rewards))

                                                  
        if ep_env_ret > top_raw_reward:
            top_raw_reward = ep_env_ret
            agent.save_checkpoint(best_raw_path)
            print(
                f"  🏆 New best RAW reward  = {top_raw_reward:.1f}  "
                f"(ep {ep}) → {best_raw_path}"
            )
                                                                
            with open(best_raw_path + ".json", "w") as mf:
                json.dump({
                    "episode":    ep,
                    "raw_reward": top_raw_reward,
                    "phase":      phase,
                    "difficulty": difficulty,
                }, mf, indent=2)

                                              
        if len(recent_rewards) == 50 and current_avg50 > top_avg50:
            top_avg50 = current_avg50
            agent.save_checkpoint(best_avg50_path)
            print(
                f"  📈 New best AVG-50 reward = {top_avg50:.2f}  "
                f"(ep {ep}) → {best_avg50_path}"
            )
            with open(best_avg50_path + ".json", "w") as mf:
                json.dump({
                    "episode":     ep,
                    "avg50_reward": top_avg50,
                    "window":      recent_rewards[-50:],
                    "phase":       phase,
                }, mf, indent=2)

                                       
        record = {
            "episode":        ep,
            "phase":          phase,
            "difficulty":     difficulty,
            "walls":          use_walls,
            "env_reward":     ep_env_ret,
            "shaped_reward":  ep_shaped_ret,
            "steps":          ep_steps,
            "epsilon":        agent.epsilon,
            "avg_loss":       avg_loss if not math.isnan(avg_loss) else None,
            "detected_box":   detected_box,
            "pushed_wall":    pushed_wall,
            "detect_step":    detect_step,
            "actions":        act_histogram,
            "rolling_avg50":  current_avg50,
            "best_raw_so_far":   top_raw_reward,
            "best_avg50_so_far": top_avg50,
        }
        with open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a") as fh:
            fh.write(json.dumps(record) + "\n")

                                                                      
        if ep % cfg.save_interval == 0:
            ckpt_stem, ckpt_ext = os.path.splitext(cfg.ckpt_name)
            periodic_ckpt_path = os.path.join(
                cfg.out_dir,
                f"{ckpt_stem}_ep{ep:05d}{ckpt_ext or '.pth'}",
            )
            agent.save_checkpoint(periodic_ckpt_path)
            agent.save_checkpoint(ckpt_path)
            print(
                f"  💾 [Ep {ep}] Periodic checkpoint → {periodic_ckpt_path}  "
                f"(raw_best={top_raw_reward:.1f}, avg50_best={top_avg50:.2f})"
            )

                                       
        if cfg.eval_interval > 0 and ep % cfg.eval_interval == 0:
            eval_stats = run_evaluation(
                agent    = agent,
                num_runs = cfg.eval_episodes,
                seed_offset = cfg.seed + 10000 + ep,
                scale    = cfg.scaling_factor,
                arena_sz = cfg.arena_size,
                step_limit = cfg.max_steps,
                walls    = True,                                               
                box_spd  = cfg.box_speed,
                diff_list= [0, 2, 3],                                            
                stack_k  = cfg.n_stack,
            )
            print(
                f"  📊 [Eval] mean={eval_stats['mean']:.1f}  "
                f"std={eval_stats['std']:.1f}  "
                f"min={eval_stats['min']:.1f}  max={eval_stats['max']:.1f}"
            )

            if eval_stats["mean"] > top_mean:
                top_mean = eval_stats["mean"]
                agent.save_checkpoint(best_path)
                print(f"  ⭐ Best so far → {best_path}  (mean={top_mean:.1f})")

            with open(os.path.join(cfg.out_dir, "eval_log.jsonl"), "a") as fh:
                fh.write(json.dumps({
                    "episode":   ep,
                    "eval_mean": eval_stats["mean"],
                    "eval_std":  eval_stats["std"],
                    "eval_min":  eval_stats["min"],
                    "eval_max":  eval_stats["max"],
                }) + "\n")

                          
    agent.save_checkpoint(ckpt_path)
    print(f"\n{'=' * 72}")
    print(f"Training complete.")
    print(f"  Latest checkpoint  → {ckpt_path}")
    if os.path.exists(best_path):
        print(f"  Best eval-mean     → {best_path}  (mean={top_mean:.1f})")
    if os.path.exists(best_raw_path):
        print(f"  Best raw reward    → {best_raw_path}  (reward={top_raw_reward:.1f})")
    if os.path.exists(best_avg50_path):
        print(f"  Best avg-50 reward → {best_avg50_path}  (avg50={top_avg50:.2f})")
    print(f"{'=' * 72}")

                                                               
                                                                               
    import shutil
    submission_weights = os.path.join(cfg.out_dir, "weights.pth")
    if os.path.exists(best_path):
        shutil.copy2(best_path, submission_weights)
        print(f"\n→ Submission weights: BEST EVAL-MEAN copied to {submission_weights}")
    elif os.path.exists(best_avg50_path):
        shutil.copy2(best_avg50_path, submission_weights)
        print(f"\n→ Submission weights: BEST AVG-50 copied to {submission_weights}")
    elif os.path.exists(best_raw_path):
        shutil.copy2(best_raw_path, submission_weights)
        print(f"\n→ Submission weights: BEST RAW copied to {submission_weights}")


if __name__ == "__main__":
    run_training()
