import torch
import torch.nn as nn
import numpy as np
from src.model.dqn import DQN, ReplayBuffer

ACTIONS = [
    (-1.0, 1.0),  # hard left
    (-0.5, 1.0),  # soft left
    ( 0.0, 1.0),  # straight
    ( 0.5, 1.0),  # soft right
    ( 1.0, 1.0),  # hard right
]

BATCH_SIZE    = 32
GAMMA         = 0.95
LR            = 1e-3
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.99


class DQNAgent:
    def __init__(self):
        self.device    = torch.device("cpu")  # simple baseline: just use CPU
        self.policy    = DQN().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer    = ReplayBuffer()
        self.epsilon   = EPSILON_START

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        # Current Q values
        q_values  = self.policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (no separate target network for simplicity)
        with torch.no_grad():
            max_next_q = self.policy(next_states_t).max(1)[0]
            target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path="src/model/dqn_weights.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="src/model/dqn_weights.pth"):
        self.policy.load_state_dict(torch.load(path))