"""Logging, reproducibility, plotting, and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def configure_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    return logging.getLogger("pets")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_scores: list[float] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)

    def callback(self, _model: Any, _calls: int, _epoch: int, loss: float, val_score: Any, _best: Any) -> None:
        self.train_losses.append(float(loss))
        self.val_scores.append(float("nan") if val_score is None else float(val_score.mean().item()))


def save_dynamics_model(model: Any, output_dir: str | Path, *, metadata: Any | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    if metadata is not None:
        payload = asdict(metadata) if is_dataclass(metadata) else metadata
        (output_dir / "model_config.json").write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=lambda value: asdict(value) if is_dataclass(value) else str(value),
            )
            + "\n",
            encoding="utf-8",
        )
    return output_dir / "model.pth"


def plot_training_history(history: TrainingHistory, output_path: str | Path) -> Path:
    """Writes model loss and validation score plots without requiring a notebook."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(history.train_losses)
    axes[0].set_ylabel("train loss")
    axes[0].grid(True)
    axes[1].plot(history.val_scores)
    axes[1].set_xlabel("training epoch")
    axes[1].set_ylabel("validation score")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_episode_rewards(rewards: Sequence[float], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rewards, "o-")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
