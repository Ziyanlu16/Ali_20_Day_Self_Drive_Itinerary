"""Deep Q-Network agent for Gomoku."""
from __future__ import annotations

import random
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import GomokuEnv


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class GomokuQNetwork(nn.Module):
    def __init__(self, board_size: int, channels: int = 64, blocks: int = 3) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, board_size * board_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.input_layer(x)
        out = self.residuals(out)
        return self.head(out)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    """A basic DQN agent using an experience replay buffer."""

    def __init__(
        self,
        env: GomokuEnv,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 10_000,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer: Deque[Transition] = deque(maxlen=buffer_size)
        self.device = torch.device(device)

        self.online_net = GomokuQNetwork(env.board_size).to(self.device)
        self.target_net = GomokuQNetwork(env.board_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.steps_done = 0
        self.target_update = target_update

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def load(self, checkpoint: str | bytes | os.PathLike[str]) -> None:
        """Load network weights from a checkpoint file."""
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def select_action(
        self,
        state: np.ndarray,
        legal_actions: List[Tuple[int, int]],
        explore: bool = True,
    ) -> Tuple[int, int]:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        if explore:
            self.steps_done += 1

        if explore and random.random() < epsilon:
            return random.choice(legal_actions)

        self.online_net.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor).view(self.env.board_size, self.env.board_size)
            mask = torch.full_like(q_values, -float("inf"))
            for row, col in legal_actions:
                mask[row, col] = q_values[row, col]
            best = torch.argmax(mask)
        self.online_net.train()
        row = int(best // self.env.board_size)
        col = int(best % self.env.board_size)
        return row, col

    def remember(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def experience_replay(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device).float()
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device).float()
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards - self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def state_action_to_index(self, action: Tuple[int, int]) -> int:
        return action[0] * self.env.board_size + action[1]

    def index_to_action(self, index: int) -> Tuple[int, int]:
        return divmod(index, self.env.board_size)

