"""Train Trans-WM-LE from a directory of sequence-preserving rollout files."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
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
)
from utils import training_runtime
from utils.common import configure_logging, seed_everything
from utils.env import make_env
from utils.particle_policy import ParticlePolicy
from utils.replay_buffer import EpisodeBatch, OfflineRolloutDataset, RolloutReplayBuffer


LOGGER = logging.getLogger("piece_of_wm.trans_wm_le")
ARCHITECTURE_VERSION = 5
_evaluate_validation = training_runtime.evaluate_validation
_resolve_resume_checkpoint = training_runtime.resolve_resume_checkpoint
_run_pretraining = partial(
    training_runtime.run_pretraining,
    architecture_version=ARCHITECTURE_VERSION,
    description="Pretraining Trans-WM-LE",
    logger=LOGGER,
)
_save_checkpoint = partial(
    training_runtime.save_checkpoint, architecture_version=ARCHITECTURE_VERSION
)
_save_rolling_checkpoint = partial(
    training_runtime.save_rolling_checkpoint, architecture_version=ARCHITECTURE_VERSION
)
_restore_checkpoint = partial(
    training_runtime.restore_checkpoint, architecture_version=ARCHITECTURE_VERSION
)
_load_pretrained_checkpoint = partial(
    training_runtime.load_pretrained_checkpoint, architecture_version=ARCHITECTURE_VERSION
)
_read_pretrained_checkpoint = partial(
    training_runtime.read_pretrained_checkpoint, architecture_version=ARCHITECTURE_VERSION
)


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
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--particle-updates", type=int, default=5)
    parser.add_argument("--particle-sigma", type=float, default=0.1)
    parser.add_argument("--particle-temperature", type=float, default=2.0)
    parser.add_argument("--planning-horizon", type=int, default=10)
    parser.add_argument("--evaluation-rollouts", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10, help="Offline pretraining epochs.")
    parser.add_argument(
        "--checkpoint-epochs", type=int, default=1, help="Pretraining checkpoint interval."
    )
    parser.add_argument(
        "--epochs-per-rollout", type=int, default=10, help="Optimization epochs per rollout."
    )
    parser.add_argument(
        "--checkpoint-rollouts", type=int, default=10, help="Checkpoint interval in rollouts."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrain", action="store_true", help="Train only the world model.")
    parser.add_argument("--target-ema", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.2)
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
    if args.pretrain:
        replay_buffer = OfflineRolloutDataset(
            args.data_dir, max_rollouts=args.replay_capacity, seed=args.seed
        )
    else:
        replay_buffer = RolloutReplayBuffer(
            args.output_dir,
            source_dir=args.data_dir,
            max_rollouts=args.replay_capacity,
            seed=args.seed,
        )
    first_batch = replay_buffer.sample(1)
    observation_shape, action_shape = _model_shapes(first_batch)
    del first_batch
    pretrained_checkpoint = None
    if args.pretrained_checkpoint is not None:
        pretrained_checkpoint = _read_pretrained_checkpoint(
            args.pretrained_checkpoint, device
        )
        model_config = _model_config_from_checkpoint(
            pretrained_checkpoint["model_config"], observation_shape, action_shape
        )
    else:
        model_config = WorldModelConfig(
            observation_shape=observation_shape,
            action_shape=action_shape,
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
        planning_horizon=args.planning_horizon,
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
    policy_generator = torch.Generator(device=device).manual_seed(args.seed)
    evaluation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    if args.pretrained_checkpoint is not None:
        _load_pretrained_checkpoint(
            args.pretrained_checkpoint,
            model,
            trainer,
            model_config,
            training_config,
            device,
            checkpoint=pretrained_checkpoint,
        )
        LOGGER.info("Initialized world model from %s.", args.pretrained_checkpoint)
    if args.pretrain:
        _run_pretraining(
            args,
            model,
            trainer,
            model_config,
            training_config,
            replay_buffer,
            validation_batch,
            rng,
            policy_generator,
            evaluation_generator,
            metrics_path,
            device,
        )
        return

    policy = ParticlePolicy(
        num_particles=args.num_particles, horizon=args.planning_horizon
    )
    eval_env = make_env(args.env_id, render_mode="rgb_array")
    start_rollout = 0
    best_online_return = -float("inf")
    if args.resume is not None:
        resume_path = _resolve_resume_checkpoint(args.resume)
        checkpoint_state = _restore_checkpoint(
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
        start_rollout = checkpoint_state.rollout
        best_online_return = checkpoint_state.best_online_return
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
        stack.callback(eval_env.close)
        stack.callback(progress.close)
        stack.enter_context(logging_redirect_tqdm())
        for rollout_index in progress:
            current_batch = replay_buffer.sample(args.sample_rollouts)
            metrics = {}
            for _ in range(args.epochs_per_rollout):
                metrics = trainer.train_transitions(current_batch, args.batch_size, rng)
            record = {
                "rollout": rollout_index,
                "sample_rollouts": args.sample_rollouts,
                **metrics,
            }
            progress.set_postfix(
                total=f"{metrics['total']:.4f}",
            )
            is_best = False
            should_evaluate = (
                rollout_index % args.checkpoint_rollouts == 0 or rollout_index == args.rollouts
            )
            if should_evaluate:
                validation = _evaluate_validation(
                    trainer, validation_batch, args.batch_size, args.seed, device
                )
                record.update({f"validation_{name}": value for name, value in validation.items()})
                evaluation_return = _evaluate_policy(
                    "trans_wm_le",
                    model,
                    rollout_index,
                    eval_env,
                    policy,
                    args,
                    evaluation_generator,
                )
                record["evaluation_return"] = evaluation_return
                if evaluation_return > best_online_return:
                    best_online_return = evaluation_return
                    is_best = True
                LOGGER.info(
                    "evaluation rollout=%d return=%.3f validation_total=%.6f "
                    "best_return=%.3f",
                    rollout_index,
                    evaluation_return,
                    validation["total"],
                    best_online_return,
                )
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
                    0,
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
                    0,
                )
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


def _model_config_from_checkpoint(
    payload: dict[str, object],
    observation_shape: tuple[int, int, int],
    action_shape: tuple[int, ...],
) -> WorldModelConfig:
    values = dict(payload)
    checkpoint_observation_shape = tuple(values.pop("observation_shape"))
    checkpoint_action_shape = tuple(values.pop("action_shape"))
    if checkpoint_observation_shape != observation_shape:
        raise ValueError("Pretrained observation shape does not match the dataset.")
    if checkpoint_action_shape != action_shape:
        raise ValueError("Pretrained action shape does not match the dataset.")
    values["cnn_channels"] = tuple(values["cnn_channels"])
    return WorldModelConfig(
        observation_shape=observation_shape,
        action_shape=action_shape,
        **values,
    )


def _validate_positive_args(args: argparse.Namespace) -> None:
    if args.resume is not None and args.pretrained_checkpoint is not None:
        raise ValueError("--resume and --pretrained-checkpoint are mutually exclusive.")
    for name in (
        "rollouts",
        "num_envs",
        "max_steps",
        "batch_size",
        "sample_rollouts",
        "num_particles",
        "evaluation_rollouts",
        "epochs",
        "checkpoint_epochs",
        "epochs_per_rollout",
        "checkpoint_rollouts",
        "particle_updates",
        "planning_horizon",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.replay_capacity is not None and args.replay_capacity <= 0:
        raise ValueError("--replay-capacity must be positive when provided.")
    if args.particle_sigma < 0.0:
        raise ValueError("--particle-sigma must be non-negative.")
    if args.particle_temperature <= 0.0:
        raise ValueError("--particle-temperature must be positive.")


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
        "phase": "pretrain" if args.pretrain else "training",
        "pretrained_checkpoint": (
            None if args.pretrained_checkpoint is None else str(args.pretrained_checkpoint)
        ),
        "data_dir": str(args.data_dir),
        "num_rollout_files": len(rollout_files),
        "batch_size": args.batch_size,
        "replay_capacity": replay_rollouts,
        "env_id": args.env_id,
        "num_particles": args.num_particles,
        "particle_updates": args.particle_updates,
        "particle_sigma": args.particle_sigma,
        "particle_temperature": args.particle_temperature,
        "evaluation_rollouts": args.evaluation_rollouts,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    if args.pretrain:
        payload.update(
            epochs=args.epochs,
            checkpoint_epochs=args.checkpoint_epochs,
        )
    else:
        payload.update(
            rollouts=args.rollouts,
            sample_rollouts=args.sample_rollouts,
            epochs_per_rollout=args.epochs_per_rollout,
            checkpoint_rollouts=args.checkpoint_rollouts,
        )
    (output_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                args.particle_temperature,
                args.seed + 1_000_000 + rollout * args.evaluation_rollouts + evaluation_rollout,
                generator,
                record_frames=False,
            )
            returns.append(online["return"])
    finally:
        model.train(was_training)
    return float(np.mean(returns))


if __name__ == "__main__":
    main()
