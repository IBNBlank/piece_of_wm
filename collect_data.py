"""Collect random real-environment transitions for world-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.common import configure_logging, seed_everything
from utils.data import create_replay_buffer, save_replay_buffer
from utils.env import make_env, reset_env, space_shapes


def collect_random_data(env_id: str, episodes: int, max_steps: int, seed: int):
    env = make_env(env_id)
    try:
        obs_shape, action_shape = space_shapes(env)
        replay_buffer = create_replay_buffer(episodes * max_steps, obs_shape, action_shape, seed=seed)
        env.action_space.seed(seed)
        rewards: list[float] = []
        for episode in range(episodes):
            obs = reset_env(env, seed + episode)
            total_reward = 0.0
            for _ in range(max_steps):
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                replay_buffer.add(obs, action, next_obs, float(reward), terminated, truncated)
                total_reward += float(reward)
                obs = next_obs
                if terminated or truncated:
                    break
            rewards.append(total_reward)
        return replay_buffer, rewards
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pendulum-random"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must both be positive.")
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    replay_buffer, rewards = collect_random_data(args.env_id, args.episodes, args.max_steps, args.seed)
    save_replay_buffer(
        replay_buffer,
        args.output_dir,
        env_id=args.env_id,
        extra_metadata={"collection_policy": "random", "episode_rewards": rewards, "seed": args.seed},
    )
    logger.info("Saved %d transitions from %d episodes to %s", replay_buffer.num_stored, len(rewards), args.output_dir)
    logger.info("Random reward mean: %.2f", float(np.mean(rewards)))


if __name__ == "__main__":
    main()
