import torch
import torch.nn as nn
import numpy as np
from src.model.dqn import DQN, ReplayBuffer

ACTIONS = [
    (-1.0, 1.0),
    (-0.5, 1.0),
    ( 0.0, 1.0),
    ( 0.5, 1.0),
    ( 1.0, 1.0),
]

BATCH_SIZE    = 64
GAMMA         = 0.99   # discount factor
LR            = 1e-3
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995  # multiplied each episode


class DQNAgent:
    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy  = DQN().to(self.device)   # network being trained
        self.target  = DQN().to(self.device)   # stable target network
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer    = ReplayBuffer()
        self.epsilon   = EPSILON_START
        self.steps     = 0

    def select_action(self, state: list[float]) -> int:
        """Epsilon-greedy action selection. Returns action index."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy(state_t)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target(next_states_t).max(1)[0]
            target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        # Sync target network every 100 steps
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path="model/dqn_weights.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="model/dqn_weights.pth"):
        self.policy.load_state_dict(torch.load(path))