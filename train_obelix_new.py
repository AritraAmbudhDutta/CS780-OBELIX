import numpy as np
import torch
from obelix import OBELIX
from agent import DQNAgent
import os

def train():
    SCALING_FACTOR = 5
    ARENA_SIZE = 500
    MAX_STEPS = 1000
    NUM_EPISODES = 2000 
    TARGET_UPDATE_STEPS = 5000 
    
    BEST_SAVE_PATH = "weights.pth" 

    env = OBELIX(scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE, max_steps=MAX_STEPS)
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Initialization (Hyperparameters are now defaulted in agent.py)
    agent = DQNAgent()
    
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    
    print("Starting High-Performance Training...")
    
    total_steps = 0
    best_reward = -float('inf') 
    
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
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {total_steps}")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(BEST_SAVE_PATH)
            print(f"New Best Score: {best_reward:.2f}! Saved to {BEST_SAVE_PATH} 🔥")

if __name__ == "__main__":
    train()