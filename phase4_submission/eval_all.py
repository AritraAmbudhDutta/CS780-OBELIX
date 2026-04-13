#!/usr/bin/env python3
"""
eval_all.py — Evaluate agent using Codabench-identical settings

Runs the agent against all 3 difficulty levels (0, 2, 3) with wall_obstacles=True,
10 runs each, max_steps=1000. Reports per-difficulty and overall scores.

Also evaluates WITHOUT walls for comparison.

Usage:
  cd phase4_submission
  python eval_all.py --agent agent_final.py
  python eval_all.py --agent agent_d3qn_v2.py
"""

import argparse
import importlib.util
import os
import sys
from typing import Callable, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from obelix import OBELIX


def load_policy(path: str) -> Callable:
    spec = importlib.util.spec_from_file_location("agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "policy"):
        raise AttributeError(f"No policy() in {path}")
    return mod.policy


def evaluate_like_codabench(policy_fn, runs=10, base_seed=0, max_steps=1000,
                             wall_obstacles=True, box_speed=2):
    """Exact same settings as evaluate_on_codabench.py."""
    difficulty_levels = [0, 2, 3]
    results = {}
    all_scores = []

    for diff in difficulty_levels:
        level_scores = []
        for i in range(runs):
            seed = base_seed + i
            env = OBELIX(
                scaling_factor=5, arena_size=500, max_steps=max_steps,
                wall_obstacles=wall_obstacles, difficulty=diff,
                box_speed=box_speed, seed=seed,
            )
            obs = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            total = 0.0
            done = False
            while not done:
                action = policy_fn(obs, rng)
                obs, reward, done = env.step(action, render=False)
                total += float(reward)
            level_scores.append(total)
            all_scores.append(total)

        results[f"d{diff}_mean"] = float(np.mean(level_scores))
        results[f"d{diff}_std"] = float(np.std(level_scores))

    results["overall_mean"] = float(np.mean(all_scores))
    results["overall_std"] = float(np.std(all_scores))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, required=True, help="Path to agent file")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    agent_path = os.path.abspath(args.agent)
    print(f"\nAgent: {agent_path}")
    print(f"Runs: {args.runs} per difficulty level")

    policy_fn = load_policy(agent_path)

                                          
    print(f"\n{'='*60}")
    print("  Codabench-Identical: wall_obstacles=True, all difficulties")
    print(f"{'='*60}")

    res_wall = evaluate_like_codabench(policy_fn, runs=args.runs, base_seed=args.seed,
                                        wall_obstacles=True)
    cum_wall = res_wall["overall_mean"] * args.runs * 3                            

    print(f"\n  Difficulty 0 (static):           mean={res_wall['d0_mean']:+.1f}  std={res_wall['d0_std']:.1f}")
    print(f"  Difficulty 2 (blinking):         mean={res_wall['d2_mean']:+.1f}  std={res_wall['d2_std']:.1f}")
    print(f"  Difficulty 3 (moving+blinking):  mean={res_wall['d3_mean']:+.1f}  std={res_wall['d3_std']:.1f}")
    print(f"  Overall (with walls):            mean={res_wall['overall_mean']:+.1f}  std={res_wall['overall_std']:.1f}")
    print(f"  Cumulative (with walls, {args.runs*3} runs): {cum_wall:+.1f}")

                                 
    policy_fn = load_policy(agent_path)

                         
    print(f"\n{'='*60}")
    print("  Comparison: wall_obstacles=False, all difficulties")
    print(f"{'='*60}")

    res_nowall = evaluate_like_codabench(policy_fn, runs=args.runs, base_seed=args.seed,
                                          wall_obstacles=False)
    cum_nowall = res_nowall["overall_mean"] * args.runs * 3

    print(f"\n  Difficulty 0 (static):           mean={res_nowall['d0_mean']:+.1f}  std={res_nowall['d0_std']:.1f}")
    print(f"  Difficulty 2 (blinking):         mean={res_nowall['d2_mean']:+.1f}  std={res_nowall['d2_std']:.1f}")
    print(f"  Difficulty 3 (moving+blinking):  mean={res_nowall['d3_mean']:+.1f}  std={res_nowall['d3_std']:.1f}")
    print(f"  Overall (no walls):              mean={res_nowall['overall_mean']:+.1f}  std={res_nowall['overall_std']:.1f}")
    print(f"  Cumulative (no walls, {args.runs*3} runs):  {cum_nowall:+.1f}")

                   
    print(f"\n{'='*60}")
    print(f"  CODABENCH SUBMISSION ESTIMATE")
    print(f"{'='*60}")
    print(f"  cumulative_reward_with_wall:    {cum_wall:+.1f}")
    print(f"  cumulative_reward_no_wall:      {cum_nowall:+.1f}")
    w = 0.6 * cum_wall + 0.4 * cum_nowall
    print(f"  weighted_cumulative_reward:     {w:+.1f}")
    print()


if __name__ == "__main__":
    main()
