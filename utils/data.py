"""Replay-buffer construction and portable dataset persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from mbrl.util.replay_buffer import ReplayBuffer


DATA_FILE = "replay_buffer.npz"
METADATA_FILE = "dataset.json"


def create_replay_buffer(
    capacity: int,
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    *,
    seed: int | None = None,
) -> ReplayBuffer:
    if capacity <= 0:
        raise ValueError("Replay buffer capacity must be positive.")
    return ReplayBuffer(
        capacity=capacity,
        obs_shape=obs_shape,
        action_shape=action_shape,
        rng=np.random.default_rng(seed),
    )


def save_replay_buffer(
    replay_buffer: ReplayBuffer,
    output_dir: str | Path,
    *,
    env_id: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Saves transitions plus shape and environment metadata to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_buffer.save(output_dir)
    metadata: dict[str, Any] = {
        "env_id": env_id,
        "capacity": replay_buffer.capacity,
        "num_transitions": replay_buffer.num_stored,
        "observation_shape": list(replay_buffer.obs.shape[1:]),
        "action_shape": list(replay_buffer.action.shape[1:]),
        "format": "mbrl-replay-buffer-v1",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir / DATA_FILE


def load_replay_buffer(
    data_dir: str | Path,
    *,
    capacity: int | None = None,
    seed: int | None = None,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    """Loads a saved replay buffer, allocating enough capacity for all transitions."""
    data_dir = Path(data_dir)
    data_path = data_dir / DATA_FILE
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    with np.load(data_path) as data:
        obs_shape = tuple(data["obs"].shape[1:])
        action_shape = tuple(data["action"].shape[1:])
        stored = len(data["obs"])
    metadata_path = data_dir / METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    requested_capacity = capacity if capacity is not None else int(metadata.get("capacity", stored))
    replay_buffer = create_replay_buffer(
        max(requested_capacity, stored), obs_shape, action_shape, seed=seed
    )
    replay_buffer.load(data_dir)
    return replay_buffer, metadata


def grow_replay_buffer(replay_buffer: ReplayBuffer, capacity: int, *, seed: int | None = None) -> ReplayBuffer:
    """Returns a larger buffer containing all current transitions in their current order."""
    if capacity <= replay_buffer.capacity:
        return replay_buffer
    grown = create_replay_buffer(
        capacity,
        tuple(replay_buffer.obs.shape[1:]),
        tuple(replay_buffer.action.shape[1:]),
        seed=seed,
    )
    batch = replay_buffer.get_all()
    grown.add_batch(*batch.astuple())
    return grown
