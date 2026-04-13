"""Top-level launcher for phase 3 D3QN training."""

import os
import sys
from pathlib import Path
import runpy


def _has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent / "phase3_approaches" / "approach_b_d3qn"

    if os.access(script_dir, os.W_OK):
        os.chdir(script_dir)
    else:
        writable_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        out_dir = writable_root / "phase3_outputs" / "approach_b_d3qn"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not _has_flag("--save"):
            sys.argv.extend(["--save", str(out_dir / "weights.pth")])
        if not _has_flag("--checkpoint_dir"):
            sys.argv.extend(["--checkpoint_dir", str(out_dir / "checkpoints_d3qn")])

    runpy.run_path(str(script_dir / "train_d3qn.py"), run_name="__main__")
