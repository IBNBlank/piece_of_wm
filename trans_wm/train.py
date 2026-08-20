"""Train Trans-WM from a directory of sequence-preserving rollout files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import torch

from trans_wm import TrainingConfig, WorldModel, WorldModelConfig, WorldModelTrainer
from utils.common import configure_logging, seed_everything
from utils.replay_buffer import EpisodeBatch


LOGGER = logging.getLogger("piece_of_wm.trans_wm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/trans_wm"))
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=100, help="Number of rollouts to train.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--epochs-per-rollout", type=int, default=10, help="Optimization epochs per rollout."
    )
    parser.add_argument(
        "--checkpoint-rollouts", type=int, default=10, help="Checkpoint interval in rollouts."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--cnn-channels", default="32,64,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-ema", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--vae-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--vae-kl-weight", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    _validate_positive_args(args)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rollout_files = _rollout_files(args.data_dir)
    _validate_dataset_metadata(args.data_dir, args.num_envs, args.max_steps)
    first_batch = _load_rollout(rollout_files[0])
    observation_shape, action_shape = _model_shapes(first_batch)
    del first_batch
    model_config = WorldModelConfig(
        observation_shape=observation_shape,
        action_shape=action_shape,
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        cnn_channels=_parse_channels(args.cnn_channels),
        dropout=args.dropout,
        gamma=args.gamma,
        target_ema=args.target_ema,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        vae_reconstruction_weight=args.vae_reconstruction_weight,
        vae_kl_weight=args.vae_kl_weight,
    )
    model = WorldModel(model_config).to(device)
    trainer = WorldModelTrainer(model, training_config)
    start_rollout = 0
    if args.resume is not None:
        start_rollout = _restore_checkpoint(
            args.resume, model, trainer, model_config, training_config, rng, device
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(args.output_dir, model_config, training_config, rollout_files, args)
    metrics_path = args.output_dir / "metrics.jsonl"
    LOGGER.info(
        "Training on %d rollout files with image shape %s and action shape %s on %s",
        len(rollout_files),
        observation_shape,
        action_shape,
        device,
    )
    for rollout_index in range(start_rollout + 1, args.rollouts + 1):
        rollout_path = rollout_files[int(rng.integers(len(rollout_files)))]
        current_batch = _load_rollout(rollout_path)
        metrics = {}
        for _ in range(args.epochs_per_rollout):
            metrics = trainer.train_transitions(current_batch, args.batch_size, rng)
        record = {
            "rollout": rollout_index,
            "rollout_file": str(rollout_path),
            **metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        LOGGER.info(
            "rollout=%d/%d file=%s total=%.6f obs=%.6f reward=%.6f value=%.6f vae_recon=%.6f vae_kl=%.6f",
            rollout_index,
            args.rollouts,
            rollout_path.name,
            metrics["total"],
            metrics["observation"],
            metrics["reward"],
            metrics["value"],
            metrics["vae_reconstruction"],
            metrics["vae_kl"],
        )
        if rollout_index % args.checkpoint_rollouts == 0 or rollout_index == args.rollouts:
            _save_checkpoint(
                args.output_dir / "checkpoint.pt",
                model,
                trainer,
                model_config,
                training_config,
                rng,
                rollout_index,
            )
    LOGGER.info("Checkpoint saved to %s", args.output_dir / "checkpoint.pt")


def _load_rollout(path: Path) -> EpisodeBatch:
    required = ("obs", "action", "reward", "terminated", "truncated", "lengths")
    with np.load(path) as data:
        if missing := [name for name in required if name not in data]:
            raise ValueError(f"Rollout {path} is missing arrays: {missing}")
        arrays = {name: np.array(data[name], copy=True) for name in required}
        images = np.array(data["images"], copy=True) if "images" in data else None
    batch = EpisodeBatch(
        arrays["obs"],
        arrays["action"],
        arrays["reward"],
        arrays["terminated"],
        arrays["truncated"],
        arrays["lengths"],
        images,
    )
    if batch.images is None:
        raise ValueError(f"Rollout {path} has no images; Trans-WM requires image data.")
    if batch.images.shape[:2] != batch.obs.shape[:2]:
        raise ValueError(f"Images in {path} are not aligned with observations.")
    return batch


def _rollout_files(data_dir: Path) -> list[Path]:
    metadata_path = data_dir / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    filenames = metadata.get("rollout_files")
    if not isinstance(filenames, list) or not filenames:
        raise ValueError("dataset.json must contain a non-empty rollout_files list.")
    paths = [data_dir / str(filename) for filename in filenames]
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"Rollout files not found: {missing}")
    return paths


def _validate_dataset_metadata(data_dir: Path, num_envs: int, max_steps: int) -> None:
    metadata = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    if metadata.get("num_envs") != num_envs:
        raise ValueError(
            f"Dataset num_envs={metadata.get('num_envs')!r}, expected --num-envs {num_envs}."
        )
    if metadata.get("max_steps") != max_steps:
        raise ValueError(
            f"Dataset max_steps={metadata.get('max_steps')!r}, expected --max-steps {max_steps}."
        )


def _model_shapes(batch: EpisodeBatch) -> tuple[tuple[int, int, int], tuple[int, ...]]:
    if batch.images is None or batch.images.ndim != 5:
        raise ValueError("Images must have shape [batch, time, height, width, channels].")
    height, width, channels = batch.images.shape[2:]
    return (channels, height, width), tuple(batch.action.shape[2:])


def _parse_channels(value: str) -> tuple[int, ...]:
    try:
        channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--cnn-channels must be comma-separated integers.") from error
    if not channels or any(channel <= 0 for channel in channels):
        raise ValueError("--cnn-channels must contain positive integers.")
    return channels


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in (
        "rollouts",
        "num_envs",
        "max_steps",
        "batch_size",
        "epochs_per_rollout",
        "checkpoint_rollouts",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")


def _write_config(
    output_dir: Path,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rollout_files: list[Path],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "data_dir": str(args.data_dir),
        "num_rollout_files": len(rollout_files),
        "batch_size": args.batch_size,
        "rollouts": args.rollouts,
        "epochs_per_rollout": args.epochs_per_rollout,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_checkpoint(
    path: Path,
    model: WorldModel,
    trainer: WorldModelTrainer,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rng: np.random.Generator,
    rollout: int,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "rollout": rollout,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _restore_checkpoint(
    path: Path,
    model: WorldModel,
    trainer: WorldModelTrainer,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match CLI configuration.")
    if checkpoint["training_config"] != asdict(training_config):
        raise ValueError("Checkpoint training configuration does not match CLI configuration.")
    model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    rng.bit_generator.state = checkpoint["numpy_rng"]
    torch.set_rng_state(checkpoint["torch_rng"])
    return int(checkpoint["rollout"])


if __name__ == "__main__":
    main()
