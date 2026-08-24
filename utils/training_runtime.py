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

from utils.common import interactive_progress_enabled


RECENT_CHECKPOINTS = 2


@dataclass(frozen=True)
class CheckpointState:
    rollout: int
    best_online_return: float
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
    description: str,
    logger: logging.Logger,
) -> None:
    state = CheckpointState(0, -float("inf"), float("inf"))
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
            expected_phase="pretrain",
        )
        logger.info("Resumed pretraining from %s at epoch %d.", resume_path, state.epoch)

    best_validation_loss = state.best_validation_loss
    show_progress = interactive_progress_enabled()
    progress = tqdm(
        range(state.epoch + 1, args.epochs + 1),
        total=args.epochs,
        initial=state.epoch,
        desc=description,
        unit="epoch",
        disable=not show_progress,
        dynamic_ncols=show_progress,
    )
    latest_checkpoint: Path | None = None
    with ExitStack() as stack:
        stack.callback(progress.close)
        if show_progress:
            stack.enter_context(logging_redirect_tqdm())
        for epoch in progress:
            updates_per_epoch = (
                replay_buffer.num_stored + args.batch_size - 1
            ) // args.batch_size
            report_interval = max(1, (updates_per_epoch + 9) // 10)
            update_index = 0

            def report_update(update_metrics: dict[str, float]) -> None:
                nonlocal update_index
                update_index += 1
                if show_progress:
                    progress.set_postfix(
                        epoch=f"{epoch}/{args.epochs}",
                        update=f"{update_index}/{updates_per_epoch}",
                        total=f"{update_metrics['total']:.4f}",
                    )
                elif update_index % report_interval == 0 or update_index == updates_per_epoch:
                    logger.info(
                        "pretrain epoch=%d/%d update=%d/%d total=%.4f",
                        epoch,
                        args.epochs,
                        update_index,
                        updates_per_epoch,
                        update_metrics["total"],
                    )

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
    *,
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
    *,
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
            "numpy_rng": rng.bit_generator.state,
            "replay_rng": replay_buffer.rng_state(),
            "policy_generator_rng": policy_generator.get_state(),
            "evaluation_generator_rng": evaluation_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_online_return": best_online_return,
            "phase": phase,
            "best_validation_loss": best_validation_loss,
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
    expected_phase: str = "training",
) -> CheckpointState:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    phase = checkpoint["phase"]
    if phase != expected_phase:
        raise ValueError(
            f"Checkpoint phase is {phase!r}, expected {expected_phase!r}. "
            "Use --pretrained-checkpoint to initialize training from pretraining."
        )
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match CLI configuration.")
    if checkpoint["training_config"] != asdict(training_config):
        raise ValueError("Checkpoint training configuration does not match CLI configuration.")
    model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    rng.bit_generator.state = checkpoint["numpy_rng"]
    replay_buffer.load_rng_state(checkpoint["replay_rng"])
    policy_generator.set_state(checkpoint["policy_generator_rng"].cpu())
    evaluation_generator.set_state(checkpoint["evaluation_generator_rng"].cpu())
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    return CheckpointState(
        rollout=int(checkpoint["rollout"]) if phase == "training" else 0,
        best_online_return=float(checkpoint["best_online_return"]),
        best_validation_loss=float(checkpoint["best_validation_loss"]),
        epoch=int(checkpoint["epoch"]) if phase == "pretrain" else 0,
    )


def load_pretrained_checkpoint(
    path: Path,
    model: Any,
    model_config: Any,
    device: torch.device,
    *,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    """Initializes model weights without carrying pretraining optimizer state forward."""
    if checkpoint is None:
        checkpoint = read_pretrained_checkpoint(path, device)
    if checkpoint["model_config"] != asdict(model_config):
        raise ValueError("Pretraining model configuration does not match CLI configuration.")
    model.load_state_dict(checkpoint["model"], strict=True)


def read_pretrained_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if path.is_dir():
        best_path = path / "checkpoint_best.pt"
        path = best_path if best_path.is_file() else resolve_resume_checkpoint(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["phase"] != "pretrain":
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
    raise FileNotFoundError(f"No training checkpoints found in {path}")
