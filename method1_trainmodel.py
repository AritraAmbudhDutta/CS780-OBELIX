"""Top-level launcher for phase 3 RPPO training."""

import os
import sys
from pathlib import Path
import runpy


def _has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent / "phase3_approaches" / "approach_a_rppo"

    if os.access(script_dir, os.W_OK):
        os.chdir(script_dir)
    else:
        writable_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        out_dir = writable_root / "phase3_outputs" / "approach_a_rppo"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not _has_flag("--save"):
            sys.argv.extend(["--save", str(out_dir / "weights.pth")])
        if not _has_flag("--checkpoint_dir"):
            sys.argv.extend(["--checkpoint_dir", str(out_dir / "checkpoints_rppo")])

    runpy.run_path(str(script_dir / "train_rppo.py"), run_name="__main__")
