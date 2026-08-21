"""Train Trans-WM-LE from a directory of sequence-preserving rollout files."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from eval import run_online_episode
from trans_wm_le import (
    TrainingConfig,
    WorldModel,
    WorldModelConfig,
    WorldModelTrainer,
    discounted_returns,
)
from utils.common import configure_logging, seed_everything
from utils.env import make_env
from utils.particle_policy import ParticlePolicy
from utils.replay_buffer import EpisodeBatch, RolloutReplayBuffer


LOGGER = logging.getLogger("piece_of_wm.trans_wm_le")
VALUE_GAMMA = 0.95
RECENT_CHECKPOINTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/trans_wm_le"))
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=100, help="Number of rollouts to train.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
        help="Rollout files held in RAM; defaults to the complete dataset.",
    )
    parser.add_argument("--sample-rollouts", type=int, default=2)
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--value-rollouts", type=int, default=2)
    parser.add_argument("--value-epochs", type=int, default=1)
    parser.add_argument("--particle-updates", type=int, default=4)
    parser.add_argument("--particle-sigma", type=float, default=0.1)
    parser.add_argument("--planning-horizon", type=int, default=1)
    parser.add_argument("--evaluation-rollouts", type=int, default=10)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument(
        "--epochs-per-rollout", type=int, default=10, help="Optimization epochs per rollout."
    )
    parser.add_argument(
        "--checkpoint-rollouts", type=int, default=10, help="Checkpoint interval in rollouts."
    )
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--observation-dim", type=int, default=128)
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
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    parser.add_argument("--sigreg-frequencies", type=int, default=17)
    parser.add_argument("--sigreg-max-frequency", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    _validate_positive_args(args)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device is not None else default_device)
    rollout_files = _rollout_files(args.data_dir)
    _validate_dataset_metadata(args.data_dir, args.num_envs, args.max_steps)
    replay_buffer = RolloutReplayBuffer(
        args.output_dir,
        source_dir=args.data_dir,
        max_rollouts=args.replay_capacity,
        seed=args.seed,
    )
    first_batch = replay_buffer.sample(1)
    observation_shape, action_shape = _model_shapes(first_batch)
    del first_batch
    model_config = WorldModelConfig(
        observation_shape=observation_shape,
        action_shape=action_shape,
        observation_dim=args.observation_dim,
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
        jepa_weight=args.jepa_weight,
        sigreg_weight=args.sigreg_weight,
        sigreg_projections=args.sigreg_projections,
        sigreg_frequencies=args.sigreg_frequencies,
        sigreg_max_frequency=args.sigreg_max_frequency,
    )
    model = WorldModel(model_config).to(device)
    trainer = WorldModelTrainer(model, training_config)
    validation_batch = replay_buffer.sample(args.sample_rollouts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(
        args.output_dir,
        model_config,
        training_config,
        rollout_files,
        replay_buffer.num_rollouts,
        args,
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    policy = ParticlePolicy()
    policy_generator = torch.Generator(device=device).manual_seed(args.seed)
    evaluation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    value_env = make_env(args.env_id, render_mode="rgb_array")
    start_rollout = 0
    best_online_return = -float("inf")
    checks_without_improvement = 0
    if args.resume is not None:
        resume_path = _resolve_resume_checkpoint(args.resume)
        start_rollout, best_online_return, checks_without_improvement = _restore_checkpoint(
            resume_path,
            model,
            trainer,
            model_config,
            training_config,
            rng,
            replay_buffer,
            policy_generator,
            evaluation_generator,
            device,
        )
        LOGGER.info("Resumed training from %s at rollout %d.", resume_path, start_rollout)
    LOGGER.info(
        "Training from %d RAM-resident rollouts (%d transitions), sampling %d rollouts per update, "
        "with image shape %s and action shape %s on %s",
        replay_buffer.num_rollouts,
        replay_buffer.num_stored,
        args.sample_rollouts,
        observation_shape,
        action_shape,
        device,
    )
    progress = tqdm(
        range(start_rollout + 1, args.rollouts + 1),
        total=args.rollouts,
        initial=start_rollout,
        desc="Training Trans-WM-LE",
        unit="rollout",
    )
    latest_checkpoint: Path | None = None
    with ExitStack() as stack:
        stack.callback(value_env.close)
        stack.callback(progress.close)
        stack.enter_context(logging_redirect_tqdm())
        for rollout_index in progress:
            current_batch = replay_buffer.sample(args.sample_rollouts)
            metrics = {}
            for _ in range(args.epochs_per_rollout):
                metrics = trainer.train_transitions(current_batch, args.batch_size, rng)
            value_losses = []
            online_returns = []
            for value_rollout in range(args.value_rollouts):
                online = run_online_episode(
                    "trans_wm_le", model, value_env, policy, args.max_steps,
                    args.particle_updates, args.particle_sigma, args.planning_horizon,
                    args.seed + rollout_index * args.value_rollouts + value_rollout,
                    policy_generator, record_frames=False, return_training_data=True,
                )
                targets = discounted_returns(online["rewards"], VALUE_GAMMA)
                value_metrics = {}
                for _ in range(args.value_epochs):
                    value_metrics = trainer.train_value_rollout(online["latents"], targets)
                value_losses.append(value_metrics["value"])
                online_returns.append(online["return"])
            metrics["value"] = float(np.mean(value_losses))
            metrics["value_return"] = float(np.mean(online_returns))
            record = {
                "rollout": rollout_index,
                "sample_rollouts": args.sample_rollouts,
                **metrics,
            }
            progress.set_postfix(
                total=f"{metrics['total']:.4f}",
                value=f"{metrics['value']:.4f}",
                value_return=f"{metrics['value_return']:.2f}",
            )
            should_stop = False
            is_best = False
            should_evaluate = (
                rollout_index % args.checkpoint_rollouts == 0 or rollout_index == args.rollouts
            )
            if should_evaluate:
                validation = _evaluate_validation(
                    trainer, validation_batch, args.validation_batch_size, args.seed, device
                )
                record.update({f"validation_{name}": value for name, value in validation.items()})
                evaluation_return = _evaluate_policy(
                    "trans_wm_le",
                    model,
                    rollout_index,
                    value_env,
                    policy,
                    args,
                    evaluation_generator,
                )
                record["evaluation_return"] = evaluation_return
                if evaluation_return > best_online_return:
                    best_online_return = evaluation_return
                    checks_without_improvement = 0
                    is_best = True
                else:
                    checks_without_improvement += 1
                LOGGER.info(
                    "evaluation rollout=%d return=%.3f validation_total=%.6f "
                    "best_return=%.3f checks_without_improvement=%d/%d",
                    rollout_index,
                    evaluation_return,
                    validation["total"],
                    best_online_return,
                    checks_without_improvement,
                    args.early_stop_patience,
                )
                should_stop = checks_without_improvement >= args.early_stop_patience
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if is_best:
                _save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model,
                    trainer,
                    model_config,
                    training_config,
                    rng,
                    replay_buffer,
                    policy_generator,
                    evaluation_generator,
                    rollout_index,
                    best_online_return,
                    checks_without_improvement,
                )
            if should_evaluate:
                latest_checkpoint = _save_rolling_checkpoint(
                    args.output_dir,
                    model,
                    trainer,
                    model_config,
                    training_config,
                    rng,
                    replay_buffer,
                    policy_generator,
                    evaluation_generator,
                    rollout_index,
                    best_online_return,
                    checks_without_improvement,
                )
            if should_stop:
                LOGGER.info("Early stopping at rollout %d.", rollout_index)
                break
    if latest_checkpoint is not None:
        LOGGER.info("Latest checkpoint saved to %s", latest_checkpoint)


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
        raise ValueError(f"Rollout {path} has no images; Trans-WM-LE requires image data.")
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
        "sample_rollouts",
        "evaluation_rollouts",
        "validation_batch_size",
        "epochs_per_rollout",
        "checkpoint_rollouts",
        "early_stop_patience",
        "value_rollouts",
        "value_epochs",
        "particle_updates",
        "planning_horizon",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.replay_capacity is not None and args.replay_capacity <= 0:
        raise ValueError("--replay-capacity must be positive when provided.")
    if args.particle_sigma < 0.0:
        raise ValueError("--particle-sigma must be non-negative.")


def _write_config(
    output_dir: Path,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rollout_files: list[Path],
    replay_rollouts: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "data_dir": str(args.data_dir),
        "num_rollout_files": len(rollout_files),
        "batch_size": args.batch_size,
        "replay_capacity": replay_rollouts,
        "sample_rollouts": args.sample_rollouts,
        "env_id": args.env_id,
        "value_rollouts": args.value_rollouts,
        "value_epochs": args.value_epochs,
        "particle_updates": args.particle_updates,
        "particle_sigma": args.particle_sigma,
        "planning_horizon": args.planning_horizon,
        "evaluation_rollouts": args.evaluation_rollouts,
        "value_gamma": VALUE_GAMMA,
        "validation_batch_size": args.validation_batch_size,
        "rollouts": args.rollouts,
        "epochs_per_rollout": args.epochs_per_rollout,
        "early_stop_patience": args.early_stop_patience,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _evaluate_validation(
    trainer: WorldModelTrainer,
    batch: EpisodeBatch,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed + 1)
        return trainer.evaluate_transitions(
            batch, batch_size, np.random.default_rng(seed + 1)
        )


def _evaluate_policy(
    model_name: str,
    model: WorldModel,
    rollout: int,
    env: object,
    policy: ParticlePolicy,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> float:
    was_training = model.training
    model.eval()
    returns = []
    try:
        for evaluation_rollout in range(args.evaluation_rollouts):
            online = run_online_episode(
                model_name,
                model,
                env,
                policy,
                args.max_steps,
                args.particle_updates,
                args.particle_sigma,
                args.planning_horizon,
                args.seed + 1_000_000 + rollout * args.evaluation_rollouts + evaluation_rollout,
                generator,
                record_frames=False,
            )
            returns.append(online["return"])
    finally:
        model.train(was_training)
    return float(np.mean(returns))


def _save_rolling_checkpoint(
    output_dir: Path,
    model: WorldModel,
    trainer: WorldModelTrainer,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rng: np.random.Generator,
    replay_buffer: RolloutReplayBuffer,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    rollout: int,
    best_online_return: float,
    checks_without_improvement: int,
) -> Path:
    path = output_dir / f"checkpoint_{rollout:06d}.pt"
    _save_checkpoint(
        path,
        model,
        trainer,
        model_config,
        training_config,
        rng,
        replay_buffer,
        policy_generator,
        evaluation_generator,
        rollout,
        best_online_return,
        checks_without_improvement,
    )
    checkpoints = sorted(
        (
            candidate
            for candidate in output_dir.glob("checkpoint_*.pt")
            if candidate.stem.removeprefix("checkpoint_").isdigit()
        ),
        key=lambda candidate: int(candidate.stem.removeprefix("checkpoint_")),
    )
    for stale_path in checkpoints[:-RECENT_CHECKPOINTS]:
        stale_path.unlink()
    return path


def _save_checkpoint(
    path: Path,
    model: WorldModel,
    trainer: WorldModelTrainer,
    model_config: WorldModelConfig,
    training_config: TrainingConfig,
    rng: np.random.Generator,
    replay_buffer: RolloutReplayBuffer,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    rollout: int,
    best_online_return: float,
    checks_without_improvement: int,
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "rollout": rollout,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "value_optimizer": trainer.value_optimizer.state_dict(),
            "architecture_version": 3,
            "checkpoint_format_version": 2,
            "numpy_rng": rng.bit_generator.state,
            "replay_rng": replay_buffer.rng_state(),
            "policy_generator_rng": policy_generator.get_state(),
            "evaluation_generator_rng": evaluation_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_online_return": best_online_return,
            "checks_without_improvement": checks_without_improvement,
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
    replay_buffer: RolloutReplayBuffer,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture_version") != 3:
        raise ValueError("Checkpoint predates Transformer latent dynamics; retrain from scratch.")
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match CLI configuration.")
    if checkpoint["training_config"] != asdict(training_config):
        raise ValueError("Checkpoint training configuration does not match CLI configuration.")
    model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
    rng.bit_generator.state = checkpoint["numpy_rng"]
    if checkpoint.get("checkpoint_format_version") == 2:
        replay_buffer.load_rng_state(checkpoint["replay_rng"])
        policy_generator.set_state(checkpoint["policy_generator_rng"].cpu())
        evaluation_generator.set_state(checkpoint["evaluation_generator_rng"].cpu())
    else:
        LOGGER.warning(
            "Checkpoint predates complete resume-state support; replay, policy, evaluation, "
            "and early-stop state will restart from their configured seeds."
        )
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    return (
        int(checkpoint["rollout"]),
        float(checkpoint.get("best_online_return", -float("inf"))),
        int(checkpoint.get("checks_without_improvement", 0)),
    )


def _resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoints = sorted(
        (
            candidate
            for candidate in path.glob("checkpoint_*.pt")
            if candidate.stem.removeprefix("checkpoint_").isdigit()
        ),
        key=lambda candidate: int(candidate.stem.removeprefix("checkpoint_")),
    )
    if checkpoints:
        return checkpoints[-1]
    legacy_path = path / "checkpoint.pt"
    if legacy_path.is_file():
        return legacy_path
    raise FileNotFoundError(f"No training checkpoints found in {path}")


if __name__ == "__main__":
    main()
