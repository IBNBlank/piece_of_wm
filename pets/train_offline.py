"""Train a PETS dynamics ensemble from a saved offline replay buffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from pets.model import ModelTrainingConfig, PETSConfig, build_dynamics_model, train_dynamics_model
from utils.common import configure_logging, plot_training_history, resolve_device, save_dynamics_model, seed_everything
from utils.data import load_replay_buffer
from utils.env import make_env, space_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing replay_buffer.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pets-offline"))
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Torch device, defaults to CUDA when available")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    replay_buffer, metadata = load_replay_buffer(args.data_dir, seed=args.seed)
    dataset_env_id = metadata.get("env_id")
    if dataset_env_id and dataset_env_id != args.env_id:
        raise ValueError(f"Dataset was collected from {dataset_env_id}, not --env-id {args.env_id}.")

    env = make_env(args.env_id)
    try:
        obs_shape, action_shape = space_shapes(env)
        if replay_buffer.obs.shape[1:] != obs_shape or replay_buffer.action.shape[1:] != action_shape:
            raise ValueError("Dataset observation/action shapes do not match the selected environment.")
        pets_config = PETSConfig(ensemble_size=args.ensemble_size, hidden_size=args.hidden_size)
        training_config = ModelTrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
        dynamics_model = build_dynamics_model(env, pets_config, device)
        _, history = train_dynamics_model(dynamics_model, replay_buffer, training_config)
        save_dynamics_model(
            dynamics_model,
            args.output_dir,
            metadata={"pets": pets_config, "training": training_config, "env_id": args.env_id},
        )
        try:
            plot_training_history(history, args.output_dir / "training.png")
        except ModuleNotFoundError as error:
            logger.warning("Model saved, but training plot was skipped: %s", error)
        logger.info("Trained on %d transitions; checkpoint saved to %s", replay_buffer.num_stored, args.output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
