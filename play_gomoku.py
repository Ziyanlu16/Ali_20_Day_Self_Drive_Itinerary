"""Play Gomoku games with a trained agent and visualize the matches."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from gomoku_rl.agent import DQNAgent
from gomoku_rl.env import GomokuEnv


Move = Tuple[int, Tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained model checkpoint")
    parser.add_argument("--board-size", type=int, default=9, help="Board size of the Gomoku environment")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for inference")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument(
        "--opponent",
        type=str,
        choices=("self", "random"),
        default="self",
        help="Opponent type: another copy of the trained agent or a random policy",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save GIF visualizations for each game",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the exported GIF visualizations",
    )
    return parser.parse_args()


def load_agent(checkpoint: Path, env: GomokuEnv, device: str) -> DQNAgent:
    agent = DQNAgent(env, device=device)
    agent.load(checkpoint)
    return agent


def random_policy(legal_actions: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    return random.choice(list(legal_actions))


def play_game(
    env: GomokuEnv,
    agent_first: DQNAgent,
    agent_second: DQNAgent | None,
    opponent_type: str,
) -> tuple[List[np.ndarray], List[Move], int]:
    env.reset()
    board_history: List[np.ndarray] = [env.board.copy()]
    move_history: List[Move] = []

    while not env.done:
        state = env.board_to_tensor()
        legal = env.legal_actions()
        player = env.current_player
        if player == 1:
            action = agent_first.select_action(state, legal, explore=False)
        else:
            if opponent_type == "self":
                assert agent_second is not None
                action = agent_second.select_action(state, legal, explore=False)
            else:
                action = random_policy(legal)
        env.step(action)
        board_history.append(env.board.copy())
        move_history.append((player, action))

    winner = env.winner or 0
    return board_history, move_history, winner


def visualize_game(
    board_history: Sequence[np.ndarray],
    moves: Sequence[Move],
    output_path: Path,
    fps: int = 2,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    board_size = board_history[0].shape[0]

    fig, ax = plt.subplots(figsize=(board_size / 1.8, board_size / 1.8))

    def draw_board(board: np.ndarray, highlight: Tuple[int, int] | None, move_idx: int | None) -> None:
        ax.clear()
        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(-0.5, board_size - 0.5)
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))
        ax.set_xticklabels(range(board_size))
        ax.set_yticklabels(range(board_size))
        ax.set_aspect("equal")
        ax.grid(True, which="both", linestyle="-", color="#CCCCCC")
        ax.set_title("Gomoku Game")

        black = np.argwhere(board == 1)
        white = np.argwhere(board == -1)
        if black.size:
            ax.scatter(black[:, 1], board_size - 1 - black[:, 0], s=300, c="black", marker="o")
        if white.size:
            ax.scatter(white[:, 1], board_size - 1 - white[:, 0], s=300, c="white", edgecolors="black", marker="o")

        if move_idx is not None:
            player, (row, col) = moves[move_idx]
            ax.text(
                col,
                board_size - 1 - row,
                str(move_idx + 1),
                color="red" if player == 1 else "blue",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        if highlight is not None:
            row, col = highlight
            ax.scatter(
                col,
                board_size - 1 - row,
                s=350,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
            )

    def update(frame: int):
        highlight = None
        move_idx = None
        if frame > 0:
            move_idx = frame - 1
            highlight = moves[move_idx][1]
        draw_board(board_history[frame], highlight, move_idx)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(board_history), interval=1000 // max(fps, 1), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=animation.PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    env = GomokuEnv(board_size=args.board_size)
    agent_first = load_agent(args.checkpoint, env, args.device)
    agent_second = None
    if args.opponent == "self":
        agent_second = load_agent(args.checkpoint, env, args.device)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for game_idx in range(1, args.games + 1):
        board_history, moves, winner = play_game(env, agent_first, agent_second, args.opponent)
        if winner == 0:
            result = "draw"
        elif winner == 1:
            result = "agent_first"
        else:
            result = "agent_second" if args.opponent == "self" else "opponent"
        print(f"Game {game_idx}: winner={result} in {len(moves)} moves")

        if args.save_dir is not None:
            output_file = args.save_dir / f"game_{game_idx}.gif"
            visualize_game(board_history, moves, output_file, fps=args.fps)
            print(f"Saved visualization to {output_file}")


if __name__ == "__main__":
    main()
