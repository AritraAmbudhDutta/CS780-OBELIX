import argparse
import glob
import importlib.util
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from evaluate import evaluate_agent


@dataclass
class CandidateScore:
    path: str
    score_nowall: float
    score_wall: float
    aggregate: float


def _extract_episode(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"_ep[_]?(\d+)|_ep(\d+)|ep[_]?(\d+)", name)
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def _evenly_pick(paths: List[str], k: int) -> List[str]:
    if len(paths) <= k:
        return paths
    idxs = np.linspace(0, len(paths) - 1, num=k, dtype=int)
    chosen = []
    seen = set()
    for i in idxs.tolist():
        p = paths[i]
        if p not in seen:
            chosen.append(p)
            seen.add(p)
    return chosen


def _load_policy(agent_file: str):
    module_name = f"submitted_agent_{os.getpid()}_{np.random.randint(1_000_000)}"
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module: {agent_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "policy"):
        raise AttributeError(f"No policy(obs, rng) in {agent_file}")
    return mod.policy


def _evaluate_one(
    agent_file: str,
    runs: int,
    seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    difficulty: int,
    box_speed: int,
) -> Tuple[float, float]:
    policy_fn = _load_policy(agent_file)

    print("    -> evaluating no-wall setting...", flush=True)
    res_nowall = evaluate_agent(
        policy_fn,
        agent_name="candidate",
        runs=runs,
        base_seed=seed,
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=False,
        difficulty=difficulty,
        box_speed=box_speed,
    )

    print("    -> evaluating wall_obstacles setting...", flush=True)
    res_wall = evaluate_agent(
        policy_fn,
        agent_name="candidate",
        runs=runs,
        base_seed=seed,
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=True,
        difficulty=difficulty,
        box_speed=box_speed,
    )

    return res_nowall.mean_score, res_wall.mean_score


def main():
    p = argparse.ArgumentParser(
        description="Evaluate and rank OBELIX checkpoints by aggregate mean score."
    )
    p.add_argument("--agent_file", type=str, default="agent.py")
    p.add_argument(
        "--weights_glob",
        type=str,
        default="weights_ep*.pth",
        help="Glob for candidate checkpoints (relative to cwd).",
    )
    p.add_argument(
        "--include_base_weights",
        action="store_true",
        help="Also include weights.pth as a candidate.",
    )
    p.add_argument("--max_candidates", type=int, default=10)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--scaling_factor", type=int, default=5)
    p.add_argument("--arena_size", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--box_speed", type=int, default=2)

    p.add_argument(
        "--write_best_to",
        type=str,
        default=None,
        help="If set, copy best checkpoint to this path (e.g., weights.pth).",
    )

    args = p.parse_args()

    cwd = os.getcwd()
    agent_file = os.path.abspath(args.agent_file)
    if not os.path.exists(agent_file):
        raise FileNotFoundError(f"agent_file not found: {agent_file}")

    candidates = sorted(glob.glob(args.weights_glob))
    if args.include_base_weights and os.path.exists("weights.pth"):
        candidates.append("weights.pth")

    candidates = sorted(set(candidates), key=lambda x: (_extract_episode(x) is None, _extract_episode(x) or 10**18, x))

    if not candidates:
        raise RuntimeError(
            "No checkpoint files found. Train first or pass --weights_glob with a valid pattern."
        )

    picked = _evenly_pick(candidates, args.max_candidates)

    print("\nPicked candidates:")
    for c in picked:
        print(f"  - {c}")

    weights_path = os.path.join(cwd, "weights.pth")
    backup_path = None
    if os.path.exists(weights_path):
        backup_path = os.path.join(cwd, ".weights_backup_for_ranking.pth")
        shutil.copy2(weights_path, backup_path)

    scored: List[CandidateScore] = []

    try:
        for idx, c in enumerate(picked, 1):
            print(f"\n[{idx}/{len(picked)}] candidate: {c}", flush=True)
            src_path = os.path.abspath(c)
            dst_path = os.path.abspath(weights_path)
            if src_path != dst_path:
                shutil.copy2(src_path, dst_path)
            else:
                print("    -> candidate is already weights.pth (no copy needed)", flush=True)
            mean_nowall, mean_wall = _evaluate_one(
                agent_file=agent_file,
                runs=args.runs,
                seed=args.seed,
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
            )
            agg = 0.5 * (mean_nowall + mean_wall)
            scored.append(CandidateScore(c, mean_nowall, mean_wall, agg))
            print(
                f"evaluated {c}: no-wall={mean_nowall:+.2f}, wall={mean_wall:+.2f}, agg={agg:+.2f}",
                flush=True,
            )
    finally:
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, weights_path)
            os.remove(backup_path)

    scored.sort(key=lambda x: x.aggregate, reverse=True)

    print("\nRanking (best first):")
    for i, s in enumerate(scored, 1):
        print(
            f"{i:2d}. {s.path} | agg={s.aggregate:+.2f} | no-wall={s.score_nowall:+.2f} | wall={s.score_wall:+.2f}"
        )

    best = scored[0]
    print(f"\nBest checkpoint: {best.path}")

    if args.write_best_to:
        out = os.path.abspath(args.write_best_to)
        shutil.copy2(best.path, out)
        print(f"Copied best checkpoint to: {out}")


if __name__ == "__main__":
    main()
