"""Shared validation, pretraining, and checkpoint runtime for world models."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


RECENT_CHECKPOINTS = 2
VALIDATION_METRIC_VERSION = 1


@dataclass(frozen=True)
class CheckpointState:
    rollout: int
    best_online_return: float
    checks_without_improvement: int
    best_validation_loss: float
    epoch: int = 0


def evaluate_validation(
    trainer: Any,
    batch: Any,
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


def run_pretraining(
    args: Any,
    model: Any,
    trainer: Any,
    model_config: Any,
    training_config: Any,
    replay_buffer: Any,
    validation_batch: Any,
    rng: np.random.Generator,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    metrics_path: Path,
    device: torch.device,
    *,
    architecture_version: int,
    description: str,
    logger: logging.Logger,
) -> None:
    state = CheckpointState(0, -float("inf"), 0, float("inf"))
    if args.resume is not None:
        resume_path = resolve_resume_checkpoint(args.resume)
        state = restore_checkpoint(
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
            architecture_version=architecture_version,
            expected_phase="pretrain",
        )
        logger.info("Resumed pretraining from %s at epoch %d.", resume_path, state.epoch)

    best_validation_loss = state.best_validation_loss
    progress = tqdm(
        range(state.epoch + 1, args.epochs + 1),
        total=args.epochs,
        initial=state.epoch,
        desc=description,
        unit="epoch",
    )
    latest_checkpoint: Path | None = None
    with ExitStack() as stack:
        stack.callback(progress.close)
        stack.enter_context(logging_redirect_tqdm())
        for epoch in progress:
            updates_per_epoch = (
                replay_buffer.num_stored + args.batch_size - 1
            ) // args.batch_size
            with tqdm(
                total=updates_per_epoch,
                desc=f"Epoch {epoch}/{args.epochs}",
                unit="update",
                leave=False,
            ) as update_progress:
                def report_update(update_metrics: dict[str, float]) -> None:
                    update_progress.update()
                    update_progress.set_postfix(total=f"{update_metrics['total']:.4f}")

                metrics = trainer.train_epoch(
                    replay_buffer.batches,
                    args.batch_size,
                    rng,
                    on_update=report_update,
                )
            progress.set_postfix(total=f"{metrics['total']:.4f}")
            record = {"phase": "pretrain", "epoch": epoch, **metrics}
            should_validate = (
                epoch % args.checkpoint_epochs == 0 or epoch == args.epochs
            )
            if should_validate:
                validation = evaluate_validation(
                    trainer, validation_batch, args.batch_size, args.seed, device
                )
                record.update({f"validation_{name}": value for name, value in validation.items()})
                is_best = validation["total"] < best_validation_loss
                if is_best:
                    best_validation_loss = validation["total"]
                logger.info(
                    "pretrain validation epoch=%d total=%.6f best=%.6f",
                    epoch,
                    validation["total"],
                    best_validation_loss,
                )
                if is_best:
                    save_checkpoint(
                        args.output_dir / "checkpoint_best.pt",
                        model,
                        trainer,
                        model_config,
                        training_config,
                        rng,
                        replay_buffer,
                        policy_generator,
                        evaluation_generator,
                        epoch,
                        -float("inf"),
                        0,
                        architecture_version=architecture_version,
                        phase="pretrain",
                        best_validation_loss=best_validation_loss,
                    )
                latest_checkpoint = save_rolling_checkpoint(
                    args.output_dir,
                    model,
                    trainer,
                    model_config,
                    training_config,
                    rng,
                    replay_buffer,
                    policy_generator,
                    evaluation_generator,
                    epoch,
                    -float("inf"),
                    0,
                    architecture_version=architecture_version,
                    phase="pretrain",
                    best_validation_loss=best_validation_loss,
                )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    if latest_checkpoint is not None:
        logger.info("Latest pretraining checkpoint saved to %s", latest_checkpoint)


def save_rolling_checkpoint(
    output_dir: Path,
    model: Any,
    trainer: Any,
    model_config: Any,
    training_config: Any,
    rng: np.random.Generator,
    replay_buffer: Any,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    rollout: int,
    best_online_return: float,
    checks_without_improvement: int,
    *,
    architecture_version: int,
    phase: str = "training",
    best_validation_loss: float = float("inf"),
) -> Path:
    path = output_dir / f"checkpoint_{rollout:06d}.pt"
    save_checkpoint(
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
        architecture_version=architecture_version,
        phase=phase,
        best_validation_loss=best_validation_loss,
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


def save_checkpoint(
    path: Path,
    model: Any,
    trainer: Any,
    model_config: Any,
    training_config: Any,
    rng: np.random.Generator,
    replay_buffer: Any,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    rollout: int,
    best_online_return: float,
    checks_without_improvement: int,
    *,
    architecture_version: int,
    phase: str = "training",
    best_validation_loss: float = float("inf"),
) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            ("epoch" if phase == "pretrain" else "rollout"): rollout,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "architecture_version": architecture_version,
            "checkpoint_format_version": 2,
            "numpy_rng": rng.bit_generator.state,
            "replay_rng": replay_buffer.rng_state(),
            "policy_generator_rng": policy_generator.get_state(),
            "evaluation_generator_rng": evaluation_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_online_return": best_online_return,
            "checks_without_improvement": checks_without_improvement,
            "phase": phase,
            "best_validation_loss": best_validation_loss,
            "validation_metric_version": (
                VALIDATION_METRIC_VERSION if phase == "pretrain" else None
            ),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def restore_checkpoint(
    path: Path,
    model: Any,
    trainer: Any,
    model_config: Any,
    training_config: Any,
    rng: np.random.Generator,
    replay_buffer: Any,
    policy_generator: torch.Generator,
    evaluation_generator: torch.Generator,
    device: torch.device,
    *,
    architecture_version: int,
    expected_phase: str = "training",
) -> CheckpointState:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture_version") != architecture_version:
        raise ValueError("Checkpoint uses the removed value head; retrain from scratch.")
    phase = checkpoint.get("phase", "training")
    if phase != expected_phase:
        raise ValueError(
            f"Checkpoint phase is {phase!r}, expected {expected_phase!r}. "
            "Use --pretrained-checkpoint to initialize training from pretraining."
        )
    if expected_phase == "pretrain" and "epoch" not in checkpoint:
        raise ValueError(
            "This pretraining checkpoint uses the old rollout-based schedule and cannot "
            "be resumed as epoch-based training. Use it with --pretrained-checkpoint instead."
        )
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match CLI configuration.")
    if checkpoint["training_config"] != asdict(training_config):
        raise ValueError("Checkpoint training configuration does not match CLI configuration.")
    model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    rng.bit_generator.state = checkpoint["numpy_rng"]
    if checkpoint.get("checkpoint_format_version") == 2:
        replay_buffer.load_rng_state(checkpoint["replay_rng"])
        policy_generator.set_state(checkpoint["policy_generator_rng"].cpu())
        evaluation_generator.set_state(checkpoint["evaluation_generator_rng"].cpu())
    else:
        logging.getLogger(__name__).warning(
            "Checkpoint predates complete resume-state support; replay, policy, evaluation, "
            "and RNG state will restart from their configured seeds."
        )
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
    if (
        expected_phase == "pretrain"
        and checkpoint.get("validation_metric_version") != VALIDATION_METRIC_VERSION
    ):
        logging.getLogger(__name__).warning(
            "Pretraining checkpoint uses the sampled validation metric; "
            "the full-epoch validation baseline will be established at the next check."
        )
        best_validation_loss = float("inf")
    return CheckpointState(
        rollout=int(checkpoint.get("rollout", 0)),
        best_online_return=float(checkpoint.get("best_online_return", -float("inf"))),
        checks_without_improvement=int(checkpoint.get("checks_without_improvement", 0)),
        best_validation_loss=best_validation_loss,
        epoch=int(checkpoint.get("epoch", 0)),
    )


def load_pretrained_checkpoint(
    path: Path,
    model: Any,
    trainer: Any,
    model_config: Any,
    training_config: Any,
    device: torch.device,
    *,
    architecture_version: int,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    if checkpoint is None:
        checkpoint = read_pretrained_checkpoint(
            path, device, architecture_version=architecture_version
        )
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Pretraining model configuration does not match CLI configuration.")
    if checkpoint["training_config"] != asdict(training_config):
        raise ValueError("Pretraining optimizer configuration does not match CLI configuration.")
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Pretraining checkpoint world-model state is incomplete.")
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])


def read_pretrained_checkpoint(
    path: Path,
    device: torch.device,
    *,
    architecture_version: int,
) -> dict[str, Any]:
    if path.is_dir():
        best_path = path / "checkpoint_best.pt"
        path = best_path if best_path.is_file() else resolve_resume_checkpoint(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture_version") != architecture_version:
        raise ValueError("Pretraining checkpoint is incompatible with this architecture.")
    if checkpoint.get("phase") != "pretrain":
        raise ValueError("--pretrained-checkpoint requires a pretraining checkpoint.")
    return checkpoint


def resolve_resume_checkpoint(path: Path) -> Path:
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
