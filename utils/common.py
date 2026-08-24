"""Logging and reproducibility helpers."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch


def configure_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    return logging.getLogger("piece_of_wm")


def interactive_progress_enabled() -> bool:
    """Return whether carriage-return progress rendering is supported."""
    return sys.stderr.isatty()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
