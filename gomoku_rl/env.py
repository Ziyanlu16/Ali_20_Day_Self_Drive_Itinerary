"""Gomoku environment for reinforcement learning.

The environment models a square Gomoku board. The current player alternates
between 1 and -1. Empty cells are 0. A player wins by placing `n_to_win`
consecutive stones horizontally, vertically or diagonally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GomokuState:
    """Container for an environment step result."""

    board: np.ndarray
    current_player: int
    done: bool
    winner: Optional[int]
    reward: float


class GomokuEnv:
    """A minimal Gomoku environment.

    The environment uses a numpy array to represent the board. Cells are 0 when
    empty, 1 for the first player and -1 for the second player. After each move
    the environment evaluates whether either player has won or whether the
    board is full.
    """

    def __init__(
        self,
        board_size: int = 9,
        n_to_win: int = 5,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_draw: float = 0.0,
    ) -> None:
        self.board_size = board_size
        self.n_to_win = n_to_win
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.reset()

    def reset(self) -> GomokuState:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner: Optional[int] = None
        return GomokuState(self.board.copy(), self.current_player, False, None, 0.0)

    def legal_actions(self) -> List[Tuple[int, int]]:
        if self.done:
            return []
        positions = np.argwhere(self.board == 0)
        return [tuple(pos) for pos in positions]

    def step(self, action: Tuple[int, int]) -> GomokuState:
        if self.done:
            raise ValueError("Cannot step in a finished game")
        row, col = action
        if self.board[row, col] != 0:
            raise ValueError(f"Illegal action {action}; cell already occupied")

        player = self.current_player
        self.board[row, col] = player
        reward: float

        if self._is_winner(row, col):
            self.done = True
            self.winner = player
            reward = self.reward_win
        elif not (self.board == 0).any():
            self.done = True
            self.winner = None
            reward = self.reward_draw
        else:
            reward = 0.0

        if not self.done:
            self.current_player = -player
        else:
            # Keep current_player pointing to the player who would move next
            self.current_player = -player

        return GomokuState(
            self.board.copy(),
            self.current_player,
            self.done,
            self.winner,
            reward,
        )

    # Utility methods -----------------------------------------------------
    def board_to_tensor(self) -> np.ndarray:
        """Return a 3-channel representation of the board for neural nets."""
        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        empty = (self.board == 0).astype(np.float32)
        return np.stack([current, opponent, empty], axis=0)

    def clone(self) -> "GomokuEnv":
        copy_env = GomokuEnv(
            board_size=self.board_size,
            n_to_win=self.n_to_win,
            reward_win=self.reward_win,
            reward_loss=self.reward_loss,
            reward_draw=self.reward_draw,
        )
        copy_env.board = self.board.copy()
        copy_env.current_player = self.current_player
        copy_env.done = self.done
        copy_env.winner = self.winner
        return copy_env

    def _is_winner(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= self.n_to_win:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            else:
                break
        return count


def evaluate_winner(board: np.ndarray, n_to_win: int = 5) -> Optional[int]:
    """Standalone helper to evaluate a finished board."""
    env = GomokuEnv(board_size=board.shape[0], n_to_win=n_to_win)
    env.board = board.copy()
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] != 0 and env._is_winner(row, col):
                return board[row, col]
    return None
