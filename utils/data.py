"""Small persistence helpers for collected episode rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.replay_buffer import EpisodeBatch


METADATA_FILE = "dataset.json"
ROLLOUT_FILE_TEMPLATE = "rollout_{rollout_index:06d}.npz"


def save_episode_batch(batch: EpisodeBatch, output_path: str | Path) -> Path:
    """Writes one ``NUM_ENVS``-episode batch in the world-model sequence layout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs": batch.obs,
        "action": batch.action,
        "reward": batch.reward,
        "terminated": batch.terminated,
        "truncated": batch.truncated,
        "lengths": batch.lengths,
    }
    if batch.images is not None:
        arrays["images"] = batch.images
    np.savez_compressed(output_path, **arrays)
    return output_path


def save_dataset_metadata(
    output_dir: str | Path,
    *,
    env_id: str,
    rollout_files: list[str],
    num_envs: int,
    max_steps: int,
    num_transitions: int,
    seed: int,
    images_collected: bool,
) -> Path:
    """Writes the small index needed to reload a collected rollout dataset."""
    output_dir = Path(output_dir)
    metadata = {
        "format": "episode-rollout-v1",
        "env_id": env_id,
        "rollout_files": rollout_files,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "num_transitions": num_transitions,
        "seed": seed,
        "images_collected": images_collected,
    }
    metadata_path = output_dir / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path
