import torch
import torch.nn as nn
import numpy as np
from src.model.dqn import DQN, ReplayBuffer

# ---------------------------------------------------------------------------
# Action space — 9 discrete actions covering steering × throttle combinations.
# Including partial throttle lets the agent brake into corners.
# ---------------------------------------------------------------------------

ACTIONS = [
    (-1.0, 0.6),  # hard left,  partial throttle  (corner braking)
    (-1.0, 1.0),  # hard left,  full throttle
    (-0.5, 1.0),  # soft left,  full throttle
    ( 0.0, 1.0),  # straight,   full throttle
    ( 0.5, 1.0),  # soft right, full throttle
    ( 1.0, 1.0),  # hard right, full throttle
    ( 1.0, 0.6),  # hard right, partial throttle  (corner braking)
    ( 0.0, 0.4),  # straight,   coast / decelerate
    ( 0.0, 0.0),  # neutral     (emergency stop)
]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE         = 128     # was 32  — lower gradient variance
GAMMA              = 0.99    # was 0.95 — longer horizon for corner planning
LR                 = 3e-4    # was 1e-3 — lower LR pairs with target-net stability
WEIGHT_DECAY       = 1e-4    # L2 regularisation via AdamW
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 0.995   # was 0.99 — 0.99^500≈0.007; 0.995^500≈0.08
TARGET_UPDATE_FREQ = 10      # hard target-network sync every N episodes
GRAD_CLIP_NORM     = 10.0    # max L2 norm of gradient vector
WARMUP_STEPS       = 1_000   # steps before training starts (fill buffer first)


class DQNAgent:
    """
    Improved DQN agent.

    Key improvements over baseline:
    - Separate target network (eliminates feedback-loop instability)
    - AdamW optimizer with L2 weight decay
    - Huber loss (SmoothL1) — clips gradient for large TD errors
    - Gradient clipping (max norm 10.0)
    - Larger replay buffer (50k) and batch (128)
    - Slower epsilon decay (0.995/episode)
    - Per-step and per-episode metrics tracking
    """

    def __init__(self):
        self.device  = torch.device("cpu")

        self.policy  = DQN(output_size=len(ACTIONS)).to(self.device)
        self.target  = DQN(output_size=len(ACTIONS)).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()  # target never trains

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.buffer  = ReplayBuffer()
        self.epsilon = EPSILON_START

        self.step_count    = 0
        self.episode_count = 0

        self.metrics: dict[str, list] = {
            "loss":             [],
            "mean_q":           [],
            "max_q":            [],
            "episode_reward":   [],
            "epsilon":          [],
            "checkpoints_hit":  [],
        }

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy(state_t).argmax().item()

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> None:
        if not self.buffer.can_sample(max(BATCH_SIZE, WARMUP_STEPS)):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for actions taken
        q_values = self.policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values from the frozen target network
        with torch.no_grad():
            max_next_q = self.target(next_states_t).max(1)[0]
            target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, target_q)   # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.step_count += 1

        # Record per-step metrics
        self.metrics["loss"].append(loss.item())
        self.metrics["mean_q"].append(q_values.mean().item())
        self.metrics["max_q"].append(q_values.max().item())

    # ------------------------------------------------------------------
    # End-of-episode bookkeeping
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.metrics["epsilon"].append(self.epsilon)

    def record_episode(self, total_reward: float, checkpoints_hit: int) -> None:
        """Call once per episode after decay_epsilon()."""
        self.metrics["episode_reward"].append(total_reward)
        self.metrics["checkpoints_hit"].append(checkpoints_hit)
        self.episode_count += 1
        if self.episode_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

    def update_target(self) -> None:
        """Hard-copy policy weights into the target network."""
        self.target.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "src/model/dqn_weights.pth") -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str = "src/model/dqn_weights.pth") -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())
