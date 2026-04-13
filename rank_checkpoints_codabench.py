import argparse
import glob
import importlib.util
import os
import re
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


@dataclass
class CandidateScore:
    path: str
    mean_score: float
    std_score: float
    mean_score_0: float
    mean_score_2: float
    mean_score_3: float


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


def _load_policy(agent_file: str) -> ActionFn:
    module_name = f"submitted_agent_{os.getpid()}_{np.random.randint(1_000_000)}"
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module: {agent_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "policy"):
        raise AttributeError(f"No policy(obs, rng) in {agent_file}")
    policy_fn = getattr(mod, "policy")
    if not callable(policy_fn):
        raise TypeError(f"policy is not callable in {agent_file}")
    return policy_fn


def evaluate_codabench_like(
    policy_fn: ActionFn,
    *,
    runs: int,
    base_seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    box_speed: int,
) -> dict:
    difficulty_levels = [0, 2, 3]
    wall_obstacles = True

    all_scores: List[float] = []
    results = {}

    for difficulty in difficulty_levels:
        level_scores: List[float] = []

        for i in range(runs):
            seed = base_seed + i
            env = OBELIX(
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=box_speed,
                seed=seed,
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

        results[f"mean_score_{difficulty}"] = float(np.mean(level_scores))
        results[f"std_score_{difficulty}"] = float(np.std(level_scores))

    results["mean_score"] = float(np.mean(all_scores))
    results["std_score"] = float(np.std(all_scores))
    return results


def main():
    p = argparse.ArgumentParser(
        description="Rank checkpoints using Codabench-like settings (difficulty 0/2/3, wall_obstacles=True)."
    )
    p.add_argument("--agent_file", type=str, default="agent.py")
    p.add_argument(
        "--weights_glob",
        type=str,
        default="checkpoints_drqn/weights_ep*.pth",
        help="Glob for candidate checkpoints.",
    )
    p.add_argument(
        "--include_base_weights",
        action="store_true",
        help="Also include current weights.pth as a candidate.",
    )
    p.add_argument("--max_candidates", type=int, default=10)

    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--scaling_factor", type=int, default=5)
    p.add_argument("--arena_size", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1000)
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

    candidates = sorted(
        set(candidates),
        key=lambda x: (_extract_episode(x) is None, _extract_episode(x) or 10**18, x),
    )

    if not candidates:
        raise RuntimeError("No candidate checkpoints found.")

    picked = _evenly_pick(candidates, args.max_candidates)

    print("\nPicked candidates:")
    for c in picked:
        print(f"  - {c}")

    weights_path = os.path.join(cwd, "weights.pth")
    backup_path = None
    if os.path.exists(weights_path):
        backup_path = os.path.join(cwd, ".weights_backup_for_codabench_rank.pth")
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

            policy_fn = _load_policy(agent_file)
            print("    -> evaluating Codabench-like runs...", flush=True)
            res = evaluate_codabench_like(
                policy_fn,
                runs=args.runs,
                base_seed=args.seed,
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                box_speed=args.box_speed,
            )

            score = CandidateScore(
                path=c,
                mean_score=res["mean_score"],
                std_score=res["std_score"],
                mean_score_0=res["mean_score_0"],
                mean_score_2=res["mean_score_2"],
                mean_score_3=res["mean_score_3"],
            )
            scored.append(score)

            print(
                f"evaluated {c}: mean={score.mean_score:+.2f}, std={score.std_score:.2f}, "
                f"d0={score.mean_score_0:+.2f}, d2={score.mean_score_2:+.2f}, d3={score.mean_score_3:+.2f}",
                flush=True,
            )
    finally:
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, weights_path)
            os.remove(backup_path)

    scored.sort(key=lambda x: x.mean_score, reverse=True)

    print("\nRanking (best first):")
    for i, s in enumerate(scored, 1):
        print(
            f"{i:2d}. {s.path} | mean={s.mean_score:+.2f} std={s.std_score:.2f} "
            f"| d0={s.mean_score_0:+.2f} d2={s.mean_score_2:+.2f} d3={s.mean_score_3:+.2f}"
        )

    best = scored[0]
    print(f"\nBest checkpoint: {best.path}")

    if args.write_best_to:
        out = os.path.abspath(args.write_best_to)
        shutil.copy2(best.path, out)
        print(f"Copied best checkpoint to: {out}")


if __name__ == "__main__":
    main()
