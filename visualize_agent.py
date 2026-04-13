"""
visualize_agent.py  –  Watch your trained agent run live in the OBELIX arena.

Usage examples
--------------
# Watch the current best weights.pth  (default DQN / tabular agent.py)
python visualize_agent.py

# Pick a specific weights file
python visualize_agent.py --weights weights_best.pth

# Watch a specific submission file (no weights needed)
python visualize_agent.py --agent_file submission_template1.py

# Slower playback so you can see what's happening
python visualize_agent.py --delay 60

# Run with wall obstacles and difficulty 2 (blinking box)
python visualize_agent.py --wall_obstacles --difficulty 2

# Run several episodes back-to-back
python visualize_agent.py --episodes 5

Keys while watching
-------------------
  SPACE  – pause / resume
  N      – skip to next episode immediately
  Q / ESC – quit
"""

import argparse
import importlib.util
import os
import sys
import time
from typing import Callable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make sure obelix.py is importable regardless of where this script lives
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from obelix import OBELIX

# ─────────────────────────────────────────────────────────────────────────────
# HUD helpers
# ─────────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.50
FONT_MED = 0.60
FONT_LARGE = 0.75
WHITE = (255, 255, 255)
GREEN = (80, 220, 80)
YELLOW = (50, 220, 220)
RED = (80, 80, 240)
CYAN = (220, 220, 50)
GREY = (160, 160, 160)
DARK_GREY = (80, 80, 80)
BLACK = (0, 0, 0)
PANEL_BG = (25, 25, 30)

SIDEBAR_W = 280


def _draw_hud(arena_frame: np.ndarray, info: dict) -> np.ndarray:
    """Create a composite frame: arena on the left, clean HUD panel on the right."""
    h, w = arena_frame.shape[:2]

    # Create a separate dark panel (not overlaid on the arena)
    panel = np.full((h, SIDEBAR_W, 3), PANEL_BG, dtype=np.uint8)

    # Draw a thin vertical separator line
    cv2.line(panel, (0, 0), (0, h), DARK_GREY, 1)

    x0 = 16
    y = 30
    dy = 26

    def put(text, color=WHITE, scale=FONT_SMALL, bold=False):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(panel, text, (x0, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += dy

    def separator():
        nonlocal y
        y += 4
        cv2.line(panel, (x0, y), (SIDEBAR_W - 16, y), DARK_GREY, 1)
        y += 12

    # ── Title ──
    put("OBELIX Viewer", CYAN, FONT_LARGE, bold=True)
    separator()

    # ── Episode / Step ──
    put(f"Episode   {info['episode']}", WHITE)
    put(f"Step      {info['step']} / {info['max_steps']}", WHITE)
    separator()

    # ── Reward ──
    r = info["episode_reward"]
    r_col = GREEN if r >= 0 else RED
    put(f"Reward    {r:+.1f}", r_col, FONT_MED, bold=True)

    step_r = info["step_reward"]
    sr_col = GREEN if step_r >= 0 else RED
    put(f"Step R    {step_r:+.1f}", sr_col)
    separator()

    # ── State / Attached ──
    state_colors = {"F": YELLOW, "P": GREEN, "U": RED}
    state_labels = {"F": "FIND", "P": "PUSH", "U": "UNWEDGE"}
    s = info["state"]
    put(f"State     {state_labels.get(s, s)}", state_colors.get(s, WHITE), bold=True)
    att = info["attached"]
    put(f"Attached  {'YES' if att else 'no'}", GREEN if att else GREY)

    # ── Action ──
    put(f"Action    {info['action']}", CYAN)
    separator()

    # ── Sensor bar ──
    put("Sensors (18-bit obs)", GREY)
    obs = info["obs"]
    bar_x = x0
    bar_y = y
    cell_w = max(2, (SIDEBAR_W - 32) // 18)
    cell_h = 14
    for i, bit in enumerate(obs):
        color = GREEN if bit else (50, 50, 50)
        top_left = (bar_x + i * cell_w, bar_y)
        bot_right = (bar_x + i * cell_w + cell_w - 2, bar_y + cell_h)
        cv2.rectangle(panel, top_left, bot_right, color, -1)
        # Thin border for inactive cells
        if not bit:
            cv2.rectangle(panel, top_left, bot_right, (70, 70, 70), 1)
    y += cell_h + 8

    # Sensor group labels (compact)
    # Individual labels below bars
    labels = [
        "LL", "LL", "L", "L", "F1", "F1", "F2", "F2",
        "F3", "F3", "F4", "F4", "R", "R", "RR", "RR", "IR", "SK",
    ]
    for i, lbl in enumerate(labels):
        col = GREEN if obs[i] else (60, 60, 60)
        lx = bar_x + i * cell_w
        cv2.putText(panel, lbl, (lx, y + 6), FONT, 0.28, col, 1, cv2.LINE_AA)
    y += 18
    # Far/Near row
    fn_labels = ["F", "N"] * 8 + ["", ""]
    for i, lbl in enumerate(fn_labels):
        if lbl:
            col = GREEN if obs[i] else (55, 55, 55)
            lx = bar_x + i * cell_w
            cv2.putText(panel, lbl, (lx + 2, y + 6), FONT, 0.25, col, 1, cv2.LINE_AA)
    y += 16
    separator()

    # ── Stats ──
    put(f"Best      {info['best_reward']:+.1f}", YELLOW)
    put(f"Avg({info['n_done']:02d})   {info['avg_reward']:+.1f}", GREY)
    separator()

    # ── Controls ──
    put("Controls", GREY, scale=FONT_SMALL, bold=True)
    y += 2
    ctrl_scale = 0.42
    ctrl_dy = 22
    for key, desc in [("SPACE", "pause / resume"), ("N", "next episode"), ("Q / ESC", "quit")]:
        cv2.putText(panel, key, (x0, y), FONT, ctrl_scale, CYAN, 1, cv2.LINE_AA)
        cv2.putText(panel, desc, (x0 + 80, y), FONT, ctrl_scale, GREY, 1, cv2.LINE_AA)
        y += ctrl_dy

    # ── Compose: arena | panel ──
    canvas = np.hstack([arena_frame, panel])

    # ── Paused banner (over arena portion only) ──
    if info.get("paused"):
        bh = 50
        by = h // 2 - bh // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, by), (w, by + bh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(
            canvas, "PAUSED", (w // 2 - 80, h // 2 + 10),
            FONT, 1.0, YELLOW, 2, cv2.LINE_AA,
        )

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Policy loaders
# ─────────────────────────────────────────────────────────────────────────────


def _load_policy_from_file(agent_file: str) -> Callable:
    """Dynamically import a submission file and return its policy() function."""
    spec = importlib.util.spec_from_file_location("submitted_agent", agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {agent_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "policy") or not callable(mod.policy):
        raise AttributeError(f"No callable policy() found in {agent_file}")
    return mod.policy


def _load_dqn_policy(weights_path: str) -> Callable:
    """Load a DQN policy from a weights file using the Net architecture in model.py."""
    import torch

    # Try to import Net from model.py sitting next to this script
    model_path = os.path.join(os.path.dirname(__file__), "model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Net = mod.Net

    net = Net(input_dim=18, output_dim=5)
    state_dict = torch.load(weights_path, map_location="cpu")

    # weights.pth saved by TabularQAgent is just the Q-table tensor, not a state_dict
    if isinstance(state_dict, torch.Tensor):
        q_table = state_dict
        ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

        def tabular_policy(obs: np.ndarray, rng: np.random.Generator) -> str:
            idx = 0
            for bit in obs:
                idx = (idx << 1) | int(bit)
            return ACTIONS[int(torch.argmax(q_table[idx]).item())]

        return tabular_policy

    net.load_state_dict(state_dict)
    net.eval()

    import torch

    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

    def dqn_policy(obs: np.ndarray, rng: np.random.Generator) -> str:
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            idx = net(x).squeeze(0).argmax().item()
        return ACTIONS[int(idx)]

    return dqn_policy


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation loop
# ─────────────────────────────────────────────────────────────────────────────


def run_visualizer(
    policy_fn: Callable,
    *,
    episodes: int,
    seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    wall_obstacles: bool,
    difficulty: int,
    box_speed: int,
    delay_ms: int,
    policy_label: str,
    headless: bool = False,
    save_video: Optional[str] = None,
) -> None:

    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=seed,
    )

    full_width = arena_size + SIDEBAR_W

    window = "OBELIX – Agent Viewer"
    use_window = not headless
    if use_window:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, full_width, arena_size)
        except cv2.error as e:
            print("[WARN] OpenCV GUI backend not available; switching to headless mode.")
            print(f"       Details: {e}")
            use_window = False

    video_writer = None
    if save_video:
        out_path = os.path.abspath(save_video)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fps = max(1.0, 1000.0 / max(1, delay_ms)) if delay_ms > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            out_path,
            fourcc,
            fps,
            (full_width, arena_size),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {out_path}")
        print(f"  Saving video to: {out_path}")

    all_rewards = []
    best_reward = -float("inf")
    paused = False
    skip_episode = False

    print(f"\n{'=' * 60}")
    print(f"  OBELIX Visualiser  |  policy: {policy_label}")
    print(
        f"  arena={arena_size}  steps={max_steps}  diff={difficulty}"
        f"  walls={wall_obstacles}  delay={delay_ms}ms"
    )
    print(f"{'=' * 60}\n")

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep - 1
        obs = env.reset(seed=ep_seed)
        rng = np.random.default_rng(ep_seed)

        total_reward = 0.0
        prev_reward = 0.0
        done = False
        step = 0
        action_str = "—"
        skip_episode = False

        print(f"  Episode {ep}/{episodes} (seed={ep_seed}) …", end="", flush=True)

        while not done:
            # ── Handle key events ──────────────────────────────────────────
            key = -1
            if use_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # Q or ESC → quit entirely
                    print("\n  Quit by user.")
                    if video_writer is not None:
                        video_writer.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(" "):  # SPACE → pause / resume
                    paused = not paused
                elif key == ord("n"):  # N → next episode
                    skip_episode = True

            if skip_episode:
                break

            if paused:
                # Render a paused frame and wait
                env._update_frames(show=False)
                paused_arena = env.frame.copy()
                if paused_arena.shape[1] != arena_size or paused_arena.shape[0] != arena_size:
                    paused_arena = cv2.resize(paused_arena, (arena_size, arena_size), interpolation=cv2.INTER_AREA)
                hud_frame = _draw_hud(
                    paused_arena,
                    dict(
                        episode=ep,
                        step=step,
                        max_steps=max_steps,
                        episode_reward=total_reward,
                        step_reward=0.0,
                        state=env.active_state,
                        attached=env.enable_push,
                        action=action_str,
                        obs=obs,
                        best_reward=best_reward,
                        avg_reward=(np.mean(all_rewards) if all_rewards else 0.0),
                        n_done=len(all_rewards),
                        paused=True,
                    ),
                )
                if video_writer is not None:
                    video_writer.write(hud_frame)
                if use_window:
                    cv2.imshow(window, hud_frame)
                time.sleep(0.05)
                continue

            # ── Agent acts ─────────────────────────────────────────────────
            action_str = policy_fn(obs, rng)
            obs, reward, done = env.step(action_str, render=False)
            total_reward += float(reward)
            step_delta = float(reward) - prev_reward
            prev_reward = float(reward)
            step += 1

            # ── Build the frame to display ──────────────────────────────────
            # env.frame is already built inside env.step (render=False still
            # populates self.frame via _update_frames).
            display = env.frame.copy()

            # Resize to arena_size with high-quality interpolation
            if display.shape[1] != arena_size or display.shape[0] != arena_size:
                display = cv2.resize(display, (arena_size, arena_size), interpolation=cv2.INTER_AREA)

            hud_info = dict(
                episode=ep,
                step=step,
                max_steps=max_steps,
                episode_reward=total_reward,
                step_reward=step_delta,
                state=env.active_state,
                attached=env.enable_push,
                action=action_str,
                obs=obs,
                best_reward=best_reward,
                avg_reward=(np.mean(all_rewards) if all_rewards else 0.0),
                n_done=len(all_rewards),
                paused=False,
            )

            display = _draw_hud(display, hud_info)

            # Expand canvas for sidebar (sidebar drawn inside _draw_hud)
            if video_writer is not None:
                video_writer.write(display)
            if use_window:
                cv2.imshow(window, display)

            # ── Delay ──────────────────────────────────────────────────────
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

        # ── Episode summary ───────────────────────────────────────────────
        all_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward

        outcome = (
            "ATTACHED+DONE"
            if env.enable_push and env.done
            else (
                "ATTACHED"
                if env.enable_push
                else "skipped"
                if skip_episode
                else "timeout"
            )
        )
        print(f"  reward={total_reward:+.1f}  steps={step}  [{outcome}]")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Finished {len(all_rewards)} episode(s)")
    print(f"  Mean reward : {np.mean(all_rewards):+.2f}")
    print(f"  Std  reward : {np.std(all_rewards):+.2f}")
    print(f"  Best reward : {best_reward:+.2f}")
    print(f"{'=' * 60}\n")

    if video_writer is not None:
        video_writer.release()

    # Keep the last frame visible until the user closes the window
    if use_window:
        print("  Press any key in the viewer window to exit …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise a trained OBELIX agent in real-time."
    )

    # Policy source – mutually exclusive
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--agent_file",
        type=str,
        default=None,
        help="Path to a submission .py file that contains policy(obs, rng).",
    )
    src.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to a .pth weights file (uses Net architecture from model.py). "
        "If omitted and no --agent_file is given, tries weights.pth in this folder.",
    )

    # Environment knobs
    parser.add_argument("--scaling_factor", type=int, default=7)
    parser.add_argument("--arena_size", type=int, default=700)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="0=static, 2=blinking, 3=moving+blinking",
    )
    parser.add_argument("--box_speed", type=int, default=2)

    # Visualiser knobs
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to visualise."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (incremented per episode).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=20,
        help="Milliseconds to wait between steps (0 = as fast as possible).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a GUI window (for SSH/headless servers).",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Optional path to save rendered rollout as an .mp4 file.",
    )

    args = parser.parse_args()

    # ── Resolve policy ────────────────────────────────────────────────────
    if args.agent_file:
        agent_file = os.path.abspath(args.agent_file)
        policy_fn = _load_policy_from_file(agent_file)
        label = os.path.basename(agent_file)
    else:
        # Default to weights.pth next to this script if nothing else specified
        wpath = args.weights or os.path.join(os.path.dirname(__file__), "weights.pth")
        wpath = os.path.abspath(wpath)
        if not os.path.exists(wpath):
            print(f"[ERROR] Weights file not found: {wpath}")
            print("  Specify --agent_file or --weights, or put weights.pth here.")
            sys.exit(1)
        print(f"  Loading weights: {wpath}")
        policy_fn = _load_dqn_policy(wpath)
        label = os.path.basename(wpath)

    run_visualizer(
        policy_fn,
        episodes=args.episodes,
        seed=args.seed,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        delay_ms=args.delay,
        policy_label=label,
        headless=args.headless,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
