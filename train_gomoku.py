"""Train a reinforcement learning agent to play Gomoku."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from gomoku_rl.agent import DQNAgent, Transition
from gomoku_rl.env import GomokuEnv


def self_play_episode(agent: DQNAgent, env: GomokuEnv) -> Tuple[float, int]:
    """Play one self-play episode where the agent controls both players."""
    transitions: list[tuple[np.ndarray, int, np.ndarray, bool]] = []
    state = env.board_to_tensor()

    while not env.done:
        legal = env.legal_actions()
        action = agent.select_action(state, legal, explore=True)
        index = agent.state_action_to_index(action)
        result = env.step(action)
        next_state = env.board_to_tensor()

        transitions.append((state, index, next_state, result.done))
        state = next_state

        if result.done:
            break

    winner = env.winner or 0
    outcome = 0.0 if winner == 0 else 1.0

    player_reward = outcome
    # Assign rewards starting from the last move and alternating players.
    for state, action, next_state, done in reversed(transitions):
        agent.remember(Transition(state, action, player_reward, next_state, done))
        player_reward = -player_reward

    loss = agent.experience_replay()
    env.reset()
    return loss, winner


def evaluate(agent: DQNAgent, env: GomokuEnv, games: int = 10) -> float:
    wins = 0
    for _ in range(games):
        env.reset()
        state = env.board_to_tensor()
        while not env.done:
            legal = env.legal_actions()
            action = agent.select_action(state, legal, explore=False)
            env.step(action)
            state = env.board_to_tensor()
        if env.winner == 1:
            wins += 1
    return wins / games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000, help="Number of self-play episodes")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("gomoku_agent.pt"))
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = GomokuEnv(board_size=args.board_size)
    agent = DQNAgent(env, device=args.device)

    stats = []
    for episode in range(1, args.episodes + 1):
        loss, _ = self_play_episode(agent, env)
        stats.append(loss)
        if episode % args.eval_interval == 0:
            win_rate = evaluate(agent, env, args.eval_games)
            print(f"Episode {episode}: avg loss={np.mean(stats[-args.eval_interval:]):.4f}, win_rate={win_rate:.2f}")
            torch.save(agent.online_net.state_dict(), args.checkpoint)

    torch.save(agent.online_net.state_dict(), args.checkpoint)
    print(f"Training finished. Model saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
