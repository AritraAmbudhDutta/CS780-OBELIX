#!/usr/bin/env python3
"""
launch.py — Top-level launcher for OBELIX Phase 4 training

Runs from the obl/ directory. Launches training for the chosen approach
inside phase4_submission/.

Usage:
  python launch.py --approach final --episodes 15000 --device auto
  python launch.py --approach d3qn_v2 --episodes 12000 --device cuda
  python launch.py --approach final --episodes 15000 --eval-after
"""

import argparse
import os
import sys
import subprocess


def main():
    p = argparse.ArgumentParser(
        description="Launch OBELIX Phase 4 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py --approach final --episodes 15000
  python launch.py --approach d3qn_v2 --episodes 12000 --device cuda
  python launch.py --approach final --eval-after
        """,
    )
    p.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=["final", "d3qn_v2"],
        help="Which approach to train: 'final' (Simple DDQN) or 'd3qn_v2' (D3QN+GRU).",
    )
    p.add_argument("--episodes", type=int, default=None, help="Override default episode count.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eval-after", action="store_true", help="Run Codabench-like evaluation after training.")
    p.add_argument("--extra-args", type=str, default="", help="Additional args to pass to the train script (quoted).")

    args = p.parse_args()

    obl_dir = os.path.dirname(os.path.abspath(__file__))
    phase4_dir = os.path.join(obl_dir, "phase4_submission")

    train_map = {
        "final": "train_final.py",
        "d3qn_v2": "train_d3qn_v2.py",
    }
    agent_map = {
        "final": "agent_final.py",
        "d3qn_v2": "agent_d3qn_v2.py",
    }

    train_script = os.path.join(phase4_dir, train_map[args.approach])
    agent_file = os.path.join(phase4_dir, agent_map[args.approach])

    if not os.path.exists(train_script):
        print(f"ERROR: {train_script} not found.")
        sys.exit(1)

    # Build command
    cmd = [sys.executable, train_script, "--device", args.device]
    if args.episodes is not None:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.extra_args:
        cmd.extend(args.extra_args.split())

    print(f"\n{'='*60}")
    print(f"  Launching: {args.approach}")
    print(f"  Script:    {train_script}")
    print(f"  Device:    {args.device}")
    print(f"  Command:   {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=phase4_dir)

    if result.returncode != 0:
        print(f"\nTraining failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✅ Training complete for approach: {args.approach}")

    if args.eval_after:
        print(f"\nRunning Codabench-like evaluation...")
        eval_cmd = [
            sys.executable,
            os.path.join(obl_dir, "phase4_submission", "eval_all.py"),
            "--agent", agent_file,
        ]
        subprocess.run(eval_cmd, cwd=obl_dir)


if __name__ == "__main__":
    main()
