#!/usr/bin/env python3
"""
make_submission.py — Create Codabench submission zip from specified approach.

Usage:
  cd phase4_submission
  python make_submission.py --approach final
  python make_submission.py --approach d3qn_v2 --weights checkpoints_d3qn_v2/d3qn_v2_ep5000.pth
"""

import argparse
import os
import shutil
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", required=True, choices=["final", "d3qn_v2"])
    p.add_argument("--weights", type=str, default=None,
                   help="Path to weights file. Default: weights.pth in current dir.")
    p.add_argument("--output", type=str, default=None,
                   help="Output zip filename. Default: submission_<approach>.zip")
    args = p.parse_args()

    agent_map = {"final": "agent_final.py", "d3qn_v2": "agent_d3qn_v2.py"}
    agent_src = agent_map[args.approach]
    weights_src = args.weights or "weights.pth"
    output = args.output or f"submission_{args.approach}.zip"

    if not os.path.exists(agent_src):
        print(f"ERROR: {agent_src} not found")
        return
    if not os.path.exists(weights_src):
        print(f"ERROR: {weights_src} not found")
        return

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(agent_src, "agent.py")                      
        zf.write(weights_src, "weights.pth")

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"✅ Created: {output} ({size_mb:.1f} MB)")
    print(f"   Contents: agent.py (from {agent_src}), weights.pth (from {weights_src})")
    print(f"   Upload this to Codabench.")


if __name__ == "__main__":
    main()
