import numpy as np
import torch
from obelix import OBELIX
from agent import A2CAgent
import os

def train():
    SCALING_FACTOR = 5
    ARENA_SIZE = 500
    MAX_STEPS = 1000
    NUM_EPISODES = 5000 
    SAVE_INTERVAL = 100
    os.makedirs("checkpoints_a2c", exist_ok=True)
    BEST_SAVE_PATH = "weights_best.pth" 
    env = OBELIX(scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE, max_steps=MAX_STEPS)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    agent = A2CAgent()
    agent.net.train()
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    print("Starting Advantage Actor-Critic (A2C) Training...")
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
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward
            steps += 1
        agent.learn()
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {steps}")
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(BEST_SAVE_PATH)
            print(f"🔥 NEW HIGH SCORE: {best_reward:.2f}! Saved to {BEST_SAVE_PATH} 🔥")
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = f"checkpoints_a2c/weights_ep_{episode}.pth"
            agent.save(checkpoint_path)
            print(f"💾 Saved intermediate checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train()