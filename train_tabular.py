import numpy as np
import torch
from obelix import OBELIX
from agent import TabularQAgent
import os

def train():
    SCALING_FACTOR = 5
    ARENA_SIZE = 500
    MAX_STEPS = 1000
    NUM_EPISODES = 1000000 
    SAVE_INTERVAL = 50000
    os.makedirs("checkpoints_tabular", exist_ok=True)
    BEST_SAVE_PATH = "weights_best_tabular.pth" 
    env = OBELIX(scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE, max_steps=MAX_STEPS, wall_obstacles=True)
    agent = TabularQAgent()
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    print("Starting Hyper-Speed Tabular Q-Learning (WITH WALLS)...")
    best_reward = -float('inf') 
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        while not done:
            action_idx = agent.select_action(state)
            action = ACTIONS[action_idx]
            next_state, reward, done = env.step(action, render=False)
            agent.learn(state, action_idx, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
        if episode % 500 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {steps}")
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(BEST_SAVE_PATH)
            print(f"🔥 NEW OVERALL HIGH SCORE: {best_reward:.2f}! Saved to {BEST_SAVE_PATH} 🔥")
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = f"checkpoints_tabular/weights_ep_{episode}.pth"
            agent.save(checkpoint_path)
    agent.save("weights_final.pth")
    print("🏁 Training complete. Final map saved to weights_final.pth 🏁")

if __name__ == "__main__":
    train()