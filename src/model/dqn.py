import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np


class DQN(nn.Module):
    """
    Improved DQN: 3-hidden-layer network with LayerNorm and Dropout.

    Architecture: 7 → 128 → 128 → 64 → output_size
    - LayerNorm (not BatchNorm) for stable normalisation over non-IID replay batches
    - Dropout(0.1) prevents co-adaptation without destabilising early Q-value estimates
    """

    def __init__(self, input_size: int = 7, output_size: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer with 50 000-transition capacity."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)
