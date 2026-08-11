"""Run PETS online training: fit a world model, plan with CEM, and add real data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_data import collect_random_data
from pets.model import (
    ModelTrainingConfig,
    PETSConfig,
    build_dynamics_model,
    build_model_env,
    build_model_trainer,
    build_pets_agent,
    train_dynamics_model,
)
from utils.common import configure_logging, plot_episode_rewards, plot_training_history, resolve_device, save_dynamics_model, seed_everything
from utils.data import grow_replay_buffer, load_replay_buffer, save_replay_buffer
from utils.env import make_env, reset_env, space_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional random/offline seed dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pets-online"))
    parser.add_argument("--initial-episodes", type=int, default=1)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--planning-horizon", type=int, default=15)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-population-size", type=int, default=256)
    parser.add_argument("--num-particles", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _action_from_agent(agent, observation: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    action = agent.act(observation)
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(action_shape)


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.max_steps <= 0 or args.initial_episodes < 0:
        raise ValueError("--trials and --max-steps must be positive; --initial-episodes cannot be negative.")
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    if args.data_dir is not None:
        replay_buffer, metadata = load_replay_buffer(args.data_dir, seed=args.seed)
        if metadata.get("env_id") and metadata["env_id"] != args.env_id:
            raise ValueError(f"Dataset was collected from {metadata['env_id']}, not {args.env_id}.")
        initial_rewards = list(metadata.get("episode_rewards", []))
    else:
        if args.initial_episodes == 0:
            raise ValueError("Set --initial-episodes to at least 1 or provide --data-dir.")
        replay_buffer, initial_rewards = collect_random_data(
            args.env_id, args.initial_episodes, args.max_steps, args.seed
        )
    replay_buffer = grow_replay_buffer(
        replay_buffer, replay_buffer.num_stored + args.trials * args.max_steps, seed=args.seed
    )

    env = make_env(args.env_id)
    try:
        obs_shape, action_shape = space_shapes(env)
        if replay_buffer.num_stored == 0:
            raise ValueError("Online PETS needs initial data. Set --initial-episodes to at least 1 or provide --data-dir.")
        if replay_buffer.obs.shape[1:] != obs_shape or replay_buffer.action.shape[1:] != action_shape:
            raise ValueError("Dataset observation/action shapes do not match the selected environment.")

        pets_config = PETSConfig(
            ensemble_size=args.ensemble_size,
            hidden_size=args.hidden_size,
            planning_horizon=args.planning_horizon,
            cem_iterations=args.cem_iterations,
            cem_population_size=args.cem_population_size,
            num_particles=args.num_particles,
        )
        training_config = ModelTrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
        dynamics_model = build_dynamics_model(env, pets_config, device)
        model_env = build_model_env(env, dynamics_model, device=device, seed=args.seed)
        agent = build_pets_agent(env, model_env, pets_config, device)
        trainer = build_model_trainer(dynamics_model, training_config)
        history = None
        rewards = initial_rewards.copy()

        for trial in range(args.trials):
            trainer, history = train_dynamics_model(
                dynamics_model, replay_buffer, training_config, trainer=trainer, history=history
            )
            observation = reset_env(env, args.seed + trial + 1)
            agent.reset()
            episode_reward = 0.0
            for _ in range(args.max_steps):
                action = _action_from_agent(agent, observation, action_shape)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_observation = np.asarray(next_observation, dtype=np.float32)
                replay_buffer.add(observation, action, next_observation, float(reward), terminated, truncated)
                observation = next_observation
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
            logger.info("trial=%d reward=%.2f transitions=%d", trial, episode_reward, replay_buffer.num_stored)

        save_replay_buffer(
            replay_buffer,
            args.output_dir / "data",
            env_id=args.env_id,
            extra_metadata={"collection_policy": "PETS-MPC", "episode_rewards": rewards, "seed": args.seed},
        )
        save_dynamics_model(
            dynamics_model,
            args.output_dir / "model",
            metadata={"pets": pets_config, "training": training_config, "env_id": args.env_id},
        )
        if history is not None:
            try:
                plot_training_history(history, args.output_dir / "training.png")
                plot_episode_rewards(rewards, args.output_dir / "rewards.png")
            except ModuleNotFoundError as error:
                logger.warning("Training completed, but plots were skipped: %s", error)
        logger.info("Saved model and %d transitions to %s", replay_buffer.num_stored, args.output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
