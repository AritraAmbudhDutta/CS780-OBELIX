import numpy as np
import torch
from obelix import OBELIX
# Importing directly from the agent.py file you will submit
from agent import DQNAgent 
import os

def train():
    # ==========================================
    # OPTIMIZED HYPERPARAMETERS
    # ==========================================
    SCALING_FACTOR = 5
    ARENA_SIZE = 500
    MAX_STEPS = 1000
    NUM_EPISODES = 1000
    TARGET_UPDATE_STEPS = 2000
    
    # Matching the exact filename your agent.py looks for
    BEST_SAVE_PATH = "weights.pth" 

    # Initialize environment
    env = OBELIX(scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE, max_steps=MAX_STEPS)
    
    # Initialize agent with the updated learning parameters
    agent = DQNAgent(
        state_dim=18, 
        action_dim=5, 
        lr=5e-4,                 # Helps pull it out of the negative-reward trap
        buffer_capacity=100000,  # Massively increased memory capacity
        epsilon_decay=200000     # Slower decay to force more exploration
    )
    
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    
    print("Starting training (CPU Mode)...")
    
    total_steps = 0
    best_reward = -float('inf')  # Tracker for the highest score
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action_idx = agent.select_action(state)
            action = ACTIONS[action_idx]
            
            next_state, reward, done = env.step(action, render=False)
            agent.step(state, action_idx, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TARGET_UPDATE_STEPS == 0:
                agent.update_target_network()
        
        print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Total Steps: {total_steps}")
        
        # Save ONLY the best performing weights
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(BEST_SAVE_PATH)
            print(f"*** New best score! Saved weights to {BEST_SAVE_PATH} ***")

if __name__ == "__main__":
    train()