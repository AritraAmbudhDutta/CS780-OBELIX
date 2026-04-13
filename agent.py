import numpy as np
import torch
import random
import os

class TabularQAgent:
    def __init__(self, state_bits=18, action_dim=5, lr=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=50000):
        self.state_bits = state_bits
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.q_table = torch.zeros((2**state_bits, action_dim), dtype=torch.float32)

    def _get_state_index(self, state):
        state_int = 0
        for bit in state:
            state_int = (state_int << 1) | int(bit)
        return state_int

    def select_action(self, state, rng=None):
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        state_idx = self._get_state_index(state)
        if random.random() > self.epsilon:
            return int(torch.argmax(self.q_table[state_idx]).item())
        else:
            if rng:
                return int(rng.integers(0, self.action_dim))
            return random.randrange(self.action_dim)

    def learn(self, state, action, reward, next_state, done):
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)
        current_q = self.q_table[state_idx, action]
        if done:
            max_next_q = 0.0
        else:
            max_next_q = torch.max(self.q_table[next_state_idx]).item()
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action] = new_q

    def save(self, path):
        torch.save(self.q_table, path)

    def load(self, path):
        self.q_table = torch.load(path, map_location="cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def policy(obs, rng):
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
    q_table = torch.zeros((2**18, 5), dtype=torch.float32)
    if os.path.exists(weights_path):
        q_table = torch.load(weights_path, map_location="cpu")
    state_int = 0
    for bit in obs:
        state_int = (state_int << 1) | int(bit)
    action_idx = int(torch.argmax(q_table[state_int]).item())
    return ACTIONS[action_idx]